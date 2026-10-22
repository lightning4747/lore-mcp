"""Automated evaluation runner for lore-mcp retrieval pipelines.

Evaluates Pure Dense, Pure Sparse (BM25), and Hybrid (RRF + Reranker)
search against the canonical golden dataset across Hit@1, Hit@3, Hit@5,
Mean Reciprocal Rank (MRR), and Negative Scope Precision.
"""

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lore_mcp.adapters.qdrant.client import create_async_qdrant_client
from lore_mcp.adapters.qdrant.index import QdrantVectorIndex
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.config import get_settings
from lore_mcp.domain.search import SearchHit, SearchQuery
from lore_mcp.domain.types import SeriesId

# Confidence/sufficiency thresholds below which results are treated as out-of-scope
DENSE_CONFIDENCE_THRESHOLD = 0.30
SPARSE_CONFIDENCE_THRESHOLD = 6.5


class GoldenQuery(BaseModel):
    """Canonical benchmark question with expected ground truth."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    category: Literal["entity", "event", "dialogue", "negative"]
    query: str
    series: str
    target_tool: str | None = None
    target_entity: str | None = None
    expected_speaker: str | None = None
    expected_episode: int | None = None
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)


class QueryEvaluationResult(BaseModel):
    """Per-query evaluation breakdown and diagnostic data."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    category: str
    query: str
    pipeline: str
    expected_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    scores: list[float]
    first_relevant_rank: int | None  # 1-indexed, None if not found in top-K
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float
    is_negative: bool
    negative_rejected: bool  # True if 0 hits or below sufficiency threshold


class PipelineMetrics(BaseModel):
    """Aggregate metrics for a retrieval pipeline."""

    model_config = ConfigDict(frozen=True)

    pipeline_name: str
    total_queries: int
    positive_queries: int
    negative_queries: int
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    mrr: float
    negative_scope_precision: float
    category_breakdown: dict[str, dict[str, float]]


class EvaluationReport(BaseModel):
    """Comprehensive evaluation report containing summary and detailed logs."""

    model_config = ConfigDict(frozen=True)

    timestamp: str
    dataset_path: str
    total_benchmark_questions: int
    pipelines: dict[str, PipelineMetrics]
    failures: list[QueryEvaluationResult]
    detailed_results: list[QueryEvaluationResult]


def is_negative_rejected(pipeline: str, hits: list[SearchHit]) -> bool:
    """Determine if a query's returned hits represent low-confidence / out-of-scope."""
    if not hits:
        return True

    top_hit = hits[0]
    if pipeline == "dense":
        return top_hit.score < DENSE_CONFIDENCE_THRESHOLD
    if pipeline == "sparse":
        return top_hit.score < SPARSE_CONFIDENCE_THRESHOLD
    if pipeline == "hybrid":
        # Check underlying signal scores
        dense_pass = (
            top_hit.dense_score is not None
            and top_hit.dense_score >= DENSE_CONFIDENCE_THRESHOLD
        )
        sparse_pass = (
            top_hit.sparse_score is not None
            and top_hit.sparse_score >= SPARSE_CONFIDENCE_THRESHOLD
        )
        return not (dense_pass or sparse_pass)

    return False


async def evaluate_single_query(
    golden: GoldenQuery,
    pipeline_name: str,
    search_fn: Callable[[SearchQuery], Awaitable[list[SearchHit]]],
    k: int = 5,
) -> QueryEvaluationResult:
    """Evaluate a single query under a specific retrieval pipeline."""
    target_ent = golden.target_entity if golden.category == "entity" else None
    search_query = SearchQuery(
        series_id=SeriesId(golden.series),
        query=golden.query,
        entity=target_ent,
        max_results=k,
    )
    hits: list[SearchHit] = await search_fn(search_query)

    retrieved_chunk_ids = [str(hit.chunk.chunk_id) for hit in hits[:k]]
    scores = [round(hit.score, 5) for hit in hits[:k]]

    expected_set = set(golden.expected_chunk_ids)
    is_neg = golden.category == "negative"

    first_rank: int | None = None
    if not is_neg:
        for idx, cid in enumerate(retrieved_chunk_ids, start=1):
            if cid in expected_set:
                first_rank = idx
                break

    h1 = first_rank == 1
    h3 = first_rank is not None and 1 <= first_rank <= 3
    h5 = first_rank is not None and 1 <= first_rank <= 5
    rr = (1.0 / first_rank) if first_rank is not None else 0.0

    neg_rejected = is_negative_rejected(pipeline_name, hits) if is_neg else False

    return QueryEvaluationResult(
        query_id=golden.id,
        category=golden.category,
        query=golden.query,
        pipeline=pipeline_name,
        expected_chunk_ids=golden.expected_chunk_ids,
        retrieved_chunk_ids=retrieved_chunk_ids,
        scores=scores,
        first_relevant_rank=first_rank,
        hit_at_1=h1,
        hit_at_3=h3,
        hit_at_5=h5,
        reciprocal_rank=rr,
        is_negative=is_neg,
        negative_rejected=neg_rejected,
    )


def compute_metrics(
    pipeline_name: str, results: list[QueryEvaluationResult]
) -> PipelineMetrics:
    """Aggregate per-query results into benchmark pipeline metrics."""
    pos_results = [r for r in results if not r.is_negative]
    neg_results = [r for r in results if r.is_negative]

    total_pos = len(pos_results)
    total_neg = len(neg_results)

    h1 = sum(1 for r in pos_results if r.hit_at_1) / total_pos if total_pos else 0.0
    h3 = sum(1 for r in pos_results if r.hit_at_3) / total_pos if total_pos else 0.0
    h5 = sum(1 for r in pos_results if r.hit_at_5) / total_pos if total_pos else 0.0
    mrr = sum(r.reciprocal_rank for r in pos_results) / total_pos if total_pos else 0.0

    neg_prec = (
        sum(1 for r in neg_results if r.negative_rejected) / total_neg
        if total_neg
        else 0.0
    )

    categories = {"entity", "event", "dialogue"}
    breakdown: dict[str, dict[str, float]] = {}
    for cat in categories:
        cat_pos = [r for r in pos_results if r.category == cat]
        cat_count = len(cat_pos)
        if cat_count > 0:
            cat_h1 = sum(1 for r in cat_pos if r.hit_at_1) / cat_count
            cat_h3 = sum(1 for r in cat_pos if r.hit_at_3) / cat_count
            cat_h5 = sum(1 for r in cat_pos if r.hit_at_5) / cat_count
            cat_mrr = sum(r.reciprocal_rank for r in cat_pos) / cat_count
            breakdown[cat] = {
                "count": float(cat_count),
                "hit_at_1": round(cat_h1, 4),
                "hit_at_3": round(cat_h3, 4),
                "hit_at_5": round(cat_h5, 4),
                "mrr": round(cat_mrr, 4),
            }

    return PipelineMetrics(
        pipeline_name=pipeline_name,
        total_queries=len(results),
        positive_queries=total_pos,
        negative_queries=total_neg,
        hit_rate_at_1=round(h1, 4),
        hit_rate_at_3=round(h3, 4),
        hit_rate_at_5=round(h5, 4),
        mrr=round(mrr, 4),
        negative_scope_precision=round(neg_prec, 4),
        category_breakdown=breakdown,
    )


def print_comparison_table(metrics: dict[str, PipelineMetrics]) -> None:
    """Render a clean CLI summary table comparing Dense, Sparse, and Hybrid."""
    header = (
        f"{'Pipeline':<12} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | "
        f"{'MRR':<8} | {'Neg Precision':<14} | {'Status'}"
    )
    divider = "-" * len(header)
    print("\n" + "=" * len(header))
    print(" LORE-MCP RETRIEVAL EVALUATION BENCHMARK SUMMARY")
    print("=" * len(header))
    print(header)
    print(divider)

    for name, m in metrics.items():
        status = "PASSED" if m.hit_rate_at_5 >= 0.85 else "WARN"
        print(
            f"{name:<12} | {m.hit_rate_at_1 * 100:>6.1f}% | "
            f"{m.hit_rate_at_3 * 100:>6.1f}% | {m.hit_rate_at_5 * 100:>6.1f}% | "
            f"{m.mrr:>8.4f} | {m.negative_scope_precision * 100:>12.1f}% | {status}"
        )
    print(divider + "\n")

    print("CATEGORY BREAKDOWN (Hit@1 / Hit@3 / MRR):")
    cat_header = (
        f"{'Pipeline':<12} | {'Entity (n=8)':<20} | {'Event (n=8)':<20} | "
        f"{'Dialogue (n=8)':<20}"
    )
    print(cat_header)
    print("-" * len(cat_header))
    for name, m in metrics.items():
        cb = m.category_breakdown
        ent = (
            f"{cb['entity']['hit_at_1'] * 100:.0f}% / "
            f"{cb['entity']['hit_at_3'] * 100:.0f}% / "
            f"{cb['entity']['mrr']:.3f}"
            if "entity" in cb
            else "N/A"
        )
        evt = (
            f"{cb['event']['hit_at_1'] * 100:.0f}% / "
            f"{cb['event']['hit_at_3'] * 100:.0f}% / "
            f"{cb['event']['mrr']:.3f}"
            if "event" in cb
            else "N/A"
        )
        dia = (
            f"{cb['dialogue']['hit_at_1'] * 100:.0f}% / "
            f"{cb['dialogue']['hit_at_3'] * 100:.0f}% / "
            f"{cb['dialogue']['mrr']:.3f}"
            if "dialogue" in cb
            else "N/A"
        )
        print(f"{name:<12} | {ent:<20} | {evt:<20} | {dia:<20}")
    print("-" * len(cat_header) + "\n")


async def run_evaluation(
    dataset_path: Path, output_path: Path | None = None
) -> EvaluationReport:
    """Execute complete evaluation suite across all pipelines."""
    print(f"Loading golden dataset from '{dataset_path}'...")
    with open(dataset_path, encoding="utf-8") as f:
        raw_items = json.load(f)

    golden_queries = [GoldenQuery.model_validate(item) for item in raw_items]
    print(f"Loaded {len(golden_queries)} canonical benchmark questions.")

    settings = get_settings()
    client = create_async_qdrant_client(settings.qdrant)
    index = QdrantVectorIndex(client, settings.qdrant)
    retriever = HybridRetriever(index)

    pipelines = {
        "Dense": index.search_dense,
        "Sparse": index.search_sparse,
        "Hybrid": retriever.retrieve,
    }

    all_results: list[QueryEvaluationResult] = []
    pipeline_metrics: dict[str, PipelineMetrics] = {}

    try:
        for name, search_fn in pipelines.items():
            print(
                f"Evaluating pipeline '{name}' across {len(golden_queries)} queries..."
            )
            pipe_results: list[QueryEvaluationResult] = []
            for query in golden_queries:
                res = await evaluate_single_query(
                    query, name.lower(), search_fn, k=5
                )
                pipe_results.append(res)
                all_results.append(res)

            metrics = compute_metrics(name, pipe_results)
            pipeline_metrics[name] = metrics

    finally:
        await client.close()

    print_comparison_table(pipeline_metrics)

    failures = [r for r in all_results if not r.is_negative and not r.hit_at_5]
    if failures:
        print(
            f"Notice: {len(failures)} positive queries missed top-5 across pipelines:"
        )
        for fail in failures:
            print(
                f"  - [{fail.pipeline}] {fail.query_id}: {fail.query[:60]}... "
                f"(Retrieved: {fail.retrieved_chunk_ids[:2]})"
            )
        print()

    target_output = output_path or Path("eval/results.json")
    report = EvaluationReport(
        timestamp=datetime.now(UTC).isoformat(),
        dataset_path=str(dataset_path),
        total_benchmark_questions=len(golden_queries),
        pipelines=pipeline_metrics,
        failures=failures,
        detailed_results=all_results,
    )

    with open(target_output, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    print(f"Saved evaluation results and failure logs to '{target_output}'.")
    return report


def main() -> None:
    """CLI entrypoint for running retrieval evaluation."""
    parser = argparse.ArgumentParser(
        description="Run evaluation against lore-mcp golden dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/golden_dataset.json"),
        help="Path to golden dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results.json"),
        help="Path to write evaluation results JSON",
    )
    args = parser.parse_args()

    asyncio.run(run_evaluation(dataset_path=args.dataset, output_path=args.output))


if __name__ == "__main__":
    main()
