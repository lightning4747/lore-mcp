"""Unit tests for evaluation regression gating and CLI report diffing (Phase 11)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from lore_mcp.domain.evaluation import (
    AggregateMetrics,
    EvaluationConfig,
    EvaluationReport,
)
from lore_mcp.interfaces.cli.main import (
    _load_eval_config,
    diff_reports,
    main,
    print_evaluation_summary,
    run_eval_command,
)


def _create_report(
    recall_at_5: float = 0.8,
    recall_at_10: float = 0.9,
    mrr: float = 0.75,
    ndcg_at_5: float = 0.7,
    ndcg_at_10: float = 0.8,
    fallback_accuracy: float = 1.0,
    total_cross_series_leaks: int = 0,
    total_spoiler_leaks: int = 0,
    mean_chronology_tau: float | None = 0.85,
    by_case_type: dict[str, dict[str, float]] | None = None,
) -> EvaluationReport:
    metrics = AggregateMetrics(
        total_cases=10,
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        mrr=mrr,
        ndcg_at_5=ndcg_at_5,
        ndcg_at_10=ndcg_at_10,
        total_cross_series_leaks=total_cross_series_leaks,
        total_spoiler_leaks=total_spoiler_leaks,
        mean_chronology_tau=mean_chronology_tau,
        fallback_accuracy=fallback_accuracy,
        cache_hit_rate=0.5,
        latency_p50_ms=10.0,
        latency_p90_ms=20.0,
        latency_p95_ms=25.0,
        latency_p99_ms=30.0,
        by_case_type=by_case_type
        or {
            "simple_fact": {
                "total_cases": 5.0,
                "recall_at_5": 0.8,
                "recall_at_10": 1.0,
                "mrr": 0.9,
                "ndcg_at_10": 0.95,
                "cross_series_leaks": 0.0,
                "spoiler_leaks": 0.0,
            }
        },
    )
    return EvaluationReport(
        timestamp="2026-10-23T12:00:00Z",
        config=EvaluationConfig(),
        metrics=metrics,
        case_results=[],
    )


def _save_report(report: EvaluationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))


@pytest.mark.unit
def test_diff_reports_identical(tmp_path: Path) -> None:
    b_file = tmp_path / "baseline.json"
    c_file = tmp_path / "current.json"
    report = _create_report()
    _save_report(report, b_file)
    _save_report(report, c_file)

    exit_code = diff_reports(str(b_file), str(c_file), tolerance=0.05)
    assert exit_code == 0


@pytest.mark.unit
def test_diff_reports_improvement(tmp_path: Path) -> None:
    b_file = tmp_path / "baseline.json"
    c_file = tmp_path / "current.json"
    _save_report(_create_report(recall_at_5=0.7, mrr=0.6), b_file)
    _save_report(_create_report(recall_at_5=0.85, mrr=0.8), c_file)

    exit_code = diff_reports(str(b_file), str(c_file), tolerance=0.05)
    assert exit_code == 0


@pytest.mark.unit
def test_diff_reports_within_tolerance(tmp_path: Path) -> None:
    b_file = tmp_path / "baseline.json"
    c_file = tmp_path / "current.json"
    # Drop from 0.80 to 0.77 is -0.03, within tolerance 0.05
    _save_report(_create_report(recall_at_5=0.80), b_file)
    _save_report(_create_report(recall_at_5=0.77), c_file)

    exit_code = diff_reports(str(b_file), str(c_file), tolerance=0.05)
    assert exit_code == 0


@pytest.mark.unit
def test_diff_reports_metric_regression(tmp_path: Path) -> None:
    b_file = tmp_path / "baseline.json"
    c_file = tmp_path / "current.json"
    # Drop from 0.80 to 0.70 is -0.10, exceeds tolerance 0.05
    _save_report(_create_report(recall_at_5=0.80), b_file)
    _save_report(_create_report(recall_at_5=0.70), c_file)

    exit_code = diff_reports(str(b_file), str(c_file), tolerance=0.05)
    assert exit_code == 1


@pytest.mark.unit
def test_diff_reports_cross_series_leak_detected(tmp_path: Path) -> None:
    b_file = tmp_path / "baseline.json"
    c_file = tmp_path / "current.json"
    _save_report(_create_report(total_cross_series_leaks=0), b_file)
    _save_report(_create_report(total_cross_series_leaks=1), c_file)

    exit_code = diff_reports(str(b_file), str(c_file), tolerance=0.05)
    assert exit_code == 1


@pytest.mark.unit
def test_diff_reports_spoiler_leak_detected(tmp_path: Path) -> None:
    b_file = tmp_path / "baseline.json"
    c_file = tmp_path / "current.json"
    _save_report(_create_report(total_spoiler_leaks=0), b_file)
    _save_report(_create_report(total_spoiler_leaks=2), c_file)

    exit_code = diff_reports(str(b_file), str(c_file), tolerance=0.05)
    assert exit_code == 1


@pytest.mark.unit
def test_diff_reports_missing_files(tmp_path: Path) -> None:
    valid_file = tmp_path / "exists.json"
    _save_report(_create_report(), valid_file)
    missing_file = tmp_path / "does_not_exist.json"

    assert diff_reports(str(missing_file), str(valid_file)) == 1
    assert diff_reports(str(valid_file), str(missing_file)) == 1


@pytest.mark.unit
def test_diff_reports_invalid_json(tmp_path: Path) -> None:
    valid_file = tmp_path / "valid.json"
    _save_report(_create_report(), valid_file)
    bad_file = tmp_path / "corrupt.json"
    bad_file.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        diff_reports(str(bad_file), str(valid_file))


@pytest.mark.unit
def test_print_evaluation_summary(capsys: pytest.CaptureFixture[str]) -> None:
    report = _create_report(
        recall_at_5=0.8,
        total_cross_series_leaks=0,
        total_spoiler_leaks=0,
        mean_chronology_tau=0.91,
    )
    print_evaluation_summary(report)
    captured = capsys.readouterr().out
    assert "LORE MCP RETRIEVAL QUALITY EVALUATION REPORT" in captured
    assert "Recall@5    : 80.0%" in captured
    assert "Cross-series leaks : 0 [PASS]" in captured
    assert "Chronology Tau (τ) : 0.9100" in captured
    assert "simple_fact" in captured


@pytest.mark.unit
def test_print_evaluation_summary_with_leaks_and_no_tau(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _create_report(
        total_cross_series_leaks=1,
        total_spoiler_leaks=1,
        mean_chronology_tau=None,
    )
    print_evaluation_summary(report)
    captured = capsys.readouterr().out
    assert "Cross-series leaks : 1 [FAIL (LEAKS DETECTED)]" in captured
    assert "Spoiler leaks      : 1 [FAIL (SPOILERS DETECTED)]" in captured
    assert "Chronology Tau (τ) : N/A" in captured


@pytest.mark.unit
def test_load_eval_config(tmp_path: Path) -> None:
    cfg_file = tmp_path / "test_cfg.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "pipeline": "hybrid",
                "fusion_mode": "client",
                "enable_rerank": True,
                "rrf_k": 60,
                "max_results": 10,
                "dataset_path": "eval/golden/loki.yaml",
                "series": "loki",
                "tolerance": 0.05,
            },
            f,
        )

    cfg = _load_eval_config(str(cfg_file))
    assert cfg.pipeline == "hybrid"
    assert cfg.series == "loki"

    # Nonexistent config
    with pytest.raises(FileNotFoundError):
        _load_eval_config(str(tmp_path / "nonexistent.yaml"))

    # Invalid config structure (list instead of dict)
    invalid_file = tmp_path / "invalid.yaml"
    with open(invalid_file, "w", encoding="utf-8") as f:
        f.write("- item1\n- item2\n")
    with pytest.raises(ValueError, match="expected mapping"):
        _load_eval_config(str(invalid_file))


@pytest.mark.unit
def test_cli_main_diff_success(tmp_path: Path) -> None:
    b_file = tmp_path / "baseline.json"
    c_file = tmp_path / "current.json"
    report = _create_report()
    _save_report(report, b_file)
    _save_report(report, c_file)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "eval",
                "diff",
                "--baseline",
                str(b_file),
                "--current",
                str(c_file),
                "--tolerance",
                "0.05",
            ]
        )
    assert exc_info.value.code == 0


@pytest.mark.unit
def test_cli_main_diff_failure(tmp_path: Path) -> None:
    b_file = tmp_path / "baseline.json"
    c_file = tmp_path / "current.json"
    _save_report(_create_report(recall_at_5=0.9), b_file)
    _save_report(_create_report(recall_at_5=0.5), c_file)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "eval",
                "diff",
                "--baseline",
                str(b_file),
                "--current",
                str(c_file),
            ]
        )
    assert exc_info.value.code == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_eval_command_success(tmp_path: Path) -> None:
    cfg_file = tmp_path / "cfg.yaml"
    out_file = tmp_path / "out.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump({"series": "loki"}, f)

    mock_report = _create_report(total_cross_series_leaks=0, total_spoiler_leaks=0)

    with (
        patch("lore_mcp.interfaces.cli.main.build_application_services") as mock_build,
        patch(
            "lore_mcp.interfaces.cli.main.EvaluationRunner.run",
            new_callable=AsyncMock,
            return_value=mock_report,
        ),
    ):
        mock_services = AsyncMock()
        mock_services.owns_client = False
        mock_build.return_value = mock_services

        exit_code = await run_eval_command(
            config_file=str(cfg_file),
            output_path=str(out_file),
        )
        assert exit_code == 0
        assert out_file.exists()
        with open(out_file, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["timestamp"] == mock_report.timestamp


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_eval_command_with_series_override(tmp_path: Path) -> None:
    cfg_file = tmp_path / "cfg.yaml"
    out_file = tmp_path / "out.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump({"series": "loki"}, f)

    mock_report = _create_report()
    captured_config: list[EvaluationConfig] = []

    async def fake_run(cfg: EvaluationConfig) -> EvaluationReport:
        captured_config.append(cfg)
        return mock_report

    with (
        patch("lore_mcp.interfaces.cli.main.build_application_services") as mock_build,
        patch(
            "lore_mcp.interfaces.cli.main.EvaluationRunner.run",
            side_effect=fake_run,
        ),
    ):
        mock_services = AsyncMock()
        mock_services.owns_client = False
        mock_build.return_value = mock_services

        exit_code = await run_eval_command(
            config_file=str(cfg_file),
            series_override="custom_series",
            output_path=str(out_file),
        )
        assert exit_code == 0
        assert len(captured_config) == 1
        assert captured_config[0].series == "custom_series"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_eval_command_leak_failure(tmp_path: Path) -> None:
    cfg_file = tmp_path / "cfg.yaml"
    out_file = tmp_path / "out.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump({"series": "loki"}, f)

    mock_report = _create_report(total_cross_series_leaks=1)

    with (
        patch("lore_mcp.interfaces.cli.main.build_application_services") as mock_build,
        patch(
            "lore_mcp.interfaces.cli.main.EvaluationRunner.run",
            new_callable=AsyncMock,
            return_value=mock_report,
        ),
    ):
        mock_services = AsyncMock()
        mock_services.owns_client = True
        mock_services.qdrant_client = AsyncMock()
        mock_build.return_value = mock_services

        exit_code = await run_eval_command(
            config_file=str(cfg_file),
            output_path=str(out_file),
        )
        assert exit_code == 1
        mock_services.qdrant_client.close.assert_awaited_once()


@pytest.mark.unit
def test_cli_main_eval_run_dispatch(tmp_path: Path) -> None:
    cfg_file = tmp_path / "cfg.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump({"series": "loki"}, f)

    with patch(
        "lore_mcp.interfaces.cli.main.run_eval_command",
        new_callable=AsyncMock,
        return_value=0,
    ) as mock_run:
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "eval",
                    "run",
                    "--config",
                    str(cfg_file),
                    "--series",
                    "test_series",
                ]
            )
        assert exc_info.value.code == 0
        mock_run.assert_awaited_once_with(
            config_file=str(cfg_file),
            series_override="test_series",
            output_path="eval/results.json",
        )
