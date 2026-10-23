"""Command Line Interface for lore-mcp operations and evaluation harness (Phase 11)."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, cast

import yaml

from lore_mcp.application.evaluation_service import EvaluationRunner
from lore_mcp.composition import build_application_services
from lore_mcp.domain.evaluation import EvaluationConfig, EvaluationReport


def _load_eval_config(config_path: str) -> EvaluationConfig:
    """Load and parse evaluation configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config at {path}: expected mapping")
    raw_dict = cast(dict[str, Any], data)
    return EvaluationConfig.model_validate(raw_dict)


def print_evaluation_summary(report: EvaluationReport) -> None:
    """Print formatted terminal tables displaying evaluation metrics and leaks."""
    m = report.metrics
    cfg = report.config

    print("\n" + "=" * 70)
    print(" LORE MCP RETRIEVAL QUALITY EVALUATION REPORT (PHASE 11)")
    print("=" * 70)
    print(f" Timestamp    : {report.timestamp}")
    print(f" Pipeline     : {cfg.pipeline.upper()} (Fusion: {cfg.fusion_mode})")
    print(f" Series       : {cfg.series}")
    print(f" Dataset      : {cfg.dataset_path}")
    print(f" Total Cases  : {m.total_cases}")
    print("-" * 70)
    print(" CORE RETRIEVAL METRICS (ANSWERABLE QUERIES):")
    print(f"  Recall@5    : {m.recall_at_5 * 100:.1f}%")
    print(f"  Recall@10   : {m.recall_at_10 * 100:.1f}%")
    print(f"  MRR         : {m.mrr:.4f}")
    print(f"  nDCG@5      : {m.ndcg_at_5:.4f}")
    print(f"  nDCG@10     : {m.ndcg_at_10:.4f}")
    print("-" * 70)
    print(" LEAK DETECTION & SAFETY (ZERO TOLERANCE):")
    leak_status = (
        "PASS" if m.total_cross_series_leaks == 0 else "FAIL (LEAKS DETECTED)"
    )
    spoil_status = (
        "PASS" if m.total_spoiler_leaks == 0 else "FAIL (SPOILERS DETECTED)"
    )
    print(f"  Cross-series leaks : {m.total_cross_series_leaks} [{leak_status}]")
    print(f"  Spoiler leaks      : {m.total_spoiler_leaks} [{spoil_status}]")
    print("-" * 70)
    print(" TIMELINE & OPERATIONAL METRICS:")
    if m.mean_chronology_tau is not None:
        tau_str = f"{m.mean_chronology_tau:.4f}"
    else:
        tau_str = "N/A"
    print(f"  Chronology Tau (τ) : {tau_str}")
    print(f"  Fallback Accuracy  : {m.fallback_accuracy * 100:.1f}%")
    print(f"  Cache Hit Rate     : {m.cache_hit_rate * 100:.1f}%")
    print(f"  Latency p50 / p90  : {m.latency_p50_ms:.1f}ms / {m.latency_p90_ms:.1f}ms")
    print(f"  Latency p95 / p99  : {m.latency_p95_ms:.1f}ms / {m.latency_p99_ms:.1f}ms")
    print("-" * 70)

    if m.by_case_type:
        print(" CATEGORY BREAKDOWN:")
        print(
            f"  {'Category':<16} {'Cases':<6} {'Recall@5':<10} {'Recall@10':<10} "
            f"{'MRR':<8} {'nDCG@10':<8} {'Leaks':<6}"
        )
        for cat, stats in sorted(m.by_case_type.items()):
            cross_l = stats.get("cross_series_leaks", 0)
            spoil_l = stats.get("spoiler_leaks", 0)
            leaks = int(cross_l + spoil_l)
            print(
                f"  {cat:<16} {int(stats.get('total_cases', 0)):<6} "
                f"{stats.get('recall_at_5', 0)*100:6.1f}%    "
                f"{stats.get('recall_at_10', 0)*100:6.1f}%    "
                f"{stats.get('mrr', 0):6.4f} "
                f"{stats.get('ndcg_at_10', 0):6.4f} "
                f"{leaks:<6}"
            )
    print("=" * 70 + "\n")


async def run_eval_command(
    config_file: str,
    series_override: str | None = None,
    output_path: str | None = None,
) -> int:
    """Execute evaluation benchmark and export results."""
    cfg = _load_eval_config(config_file)
    if series_override:
        cfg = cfg.model_copy(update={"series": series_override})

    services = build_application_services(overrides=cfg)
    try:
        runner = EvaluationRunner(retrieval_service=services.retrieval_service)
        report = await runner.run(cfg)

        print_evaluation_summary(report)

        out_file = Path(output_path or "eval/results.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        print(f"Report saved to: {out_file.resolve()}\n")

        # Zero tolerance check for leaks
        has_cross_leaks = report.metrics.total_cross_series_leaks > 0
        has_spoil_leaks = report.metrics.total_spoiler_leaks > 0
        if has_cross_leaks or has_spoil_leaks:
            print("ERROR: Evaluation failed due to privacy / spoiler leaks!")
            return 1
        return 0
    finally:
        if services.owns_client:
            await services.qdrant_client.close()


def diff_reports(
    baseline_path: str,
    current_path: str,
    tolerance: float = 0.05,
) -> int:
    """Compare baseline and current evaluation reports and detect regressions."""
    b_file = Path(baseline_path)
    c_file = Path(current_path)

    if not b_file.exists():
        print(f"Baseline file does not exist: {b_file}")
        return 1
    if not c_file.exists():
        print(f"Current evaluation file does not exist: {c_file}")
        return 1

    with open(b_file, encoding="utf-8") as f:
        b_raw = json.load(f)
    with open(c_file, encoding="utf-8") as f:
        c_raw = json.load(f)

    baseline = EvaluationReport.model_validate(b_raw)
    current = EvaluationReport.model_validate(c_raw)

    bm = baseline.metrics
    cm = current.metrics

    print("\n" + "=" * 70)
    print(" RETRIEVAL REGRESSION DIFF REPORT")
    print("=" * 70)
    print(f" Baseline: {baseline_path} ({baseline.timestamp})")
    print(f" Current : {current_path} ({current.timestamp})")
    print(f" Allowed tolerance: -{tolerance * 100:.1f}%")
    print("-" * 70)
    print(
        f" {'Metric':<22} {'Baseline':<12} {'Current':<12} "
        f"{'Delta':<10} {'Status':<10}"
    )

    regressions: list[str] = []

    metrics_to_compare: list[tuple[str, float, float]] = [
        ("Recall@5", bm.recall_at_5, cm.recall_at_5),
        ("Recall@10", bm.recall_at_10, cm.recall_at_10),
        ("MRR", bm.mrr, cm.mrr),
        ("nDCG@5", bm.ndcg_at_5, cm.ndcg_at_5),
        ("nDCG@10", bm.ndcg_at_10, cm.ndcg_at_10),
        ("Fallback Accuracy", bm.fallback_accuracy, cm.fallback_accuracy),
    ]

    for name, b_val, c_val in metrics_to_compare:
        delta = c_val - b_val
        is_regression = delta < -tolerance
        if is_regression:
            status = "REGRESSION"
        elif delta >= 0:
            status = "PASS"
        else:
            status = "WARN"
        if is_regression:
            regressions.append(
                f"{name} dropped from {b_val:.4f} to {c_val:.4f} (delta {delta:+.4f})"
            )
        print(
            f" {name:<22} {b_val:<12.4f} {c_val:<12.4f} {delta:<+10.4f} {status:<10}"
        )

    print("-" * 70)
    # Check leak regressions
    cross_leaks = cm.total_cross_series_leaks
    spoiler_leaks = cm.total_spoiler_leaks
    if cross_leaks > 0:
        regressions.append(f"Cross-series leaks detected: {cross_leaks}")
    if spoiler_leaks > 0:
        regressions.append(f"Spoiler leaks detected: {spoiler_leaks}")

    c_diff = cross_leaks - bm.total_cross_series_leaks
    c_diff_str = f"{'+' if c_diff > 0 else ''}{c_diff}"
    c_status = "FAIL" if cross_leaks > 0 else "PASS"
    print(
        f" {'Cross-series leaks':<22} {bm.total_cross_series_leaks:<12} "
        f"{cross_leaks:<12} {c_diff_str:<10} {c_status:<10}"
    )

    s_diff = spoiler_leaks - bm.total_spoiler_leaks
    s_diff_str = f"{'+' if s_diff > 0 else ''}{s_diff}"
    s_status = "FAIL" if spoiler_leaks > 0 else "PASS"
    print(
        f" {'Spoiler leaks':<22} {bm.total_spoiler_leaks:<12} "
        f"{spoiler_leaks:<12} {s_diff_str:<10} {s_status:<10}"
    )
    print("=" * 70)

    if regressions:
        print("\nCI EVALUATION GATING FAILED:")
        for r in regressions:
            print(f"  - {r}")
        print()
        return 1

    print("\nCI EVALUATION GATING PASSED: No leaks and all metrics within tolerance.\n")
    return 0


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for lore command."""
    parser = argparse.ArgumentParser(
        prog="lore",
        description="Lore MCP Command Line Interface",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # eval subparser
    eval_parser = subparsers.add_parser("eval", help="Evaluation commands")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_action", required=True)

    # lore eval run
    run_parser = eval_subparsers.add_parser("run", help="Run retrieval evaluation")
    run_parser.add_argument(
        "--config",
        default="eval/config.yaml",
        help="Path to evaluation config YAML",
    )
    run_parser.add_argument(
        "--series",
        default=None,
        help="Target series identifier override",
    )
    run_parser.add_argument(
        "--output",
        default="eval/results.json",
        help="Path to save evaluation JSON results",
    )

    # lore eval diff
    diff_parser = eval_subparsers.add_parser("diff", help="Diff evaluation reports")
    diff_parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline evaluation JSON report",
    )
    diff_parser.add_argument(
        "--current",
        required=True,
        help="Path to current evaluation JSON report",
    )
    diff_parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Maximum allowable metric regression (e.g. 0.05 for 5%)",
    )

    args = parser.parse_args(argv)

    if args.subcommand == "eval":
        if args.eval_action == "run":
            exit_code = asyncio.run(
                run_eval_command(
                    config_file=args.config,
                    series_override=args.series,
                    output_path=args.output,
                )
            )
            sys.exit(exit_code)
        elif args.eval_action == "diff":
            exit_code = diff_reports(
                baseline_path=args.baseline,
                current_path=args.current,
                tolerance=args.tolerance,
            )
            sys.exit(exit_code)


if __name__ == "__main__":
    main()
