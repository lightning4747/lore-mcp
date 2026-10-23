"""Application evaluation service for retrieval quality and Phase 11 compliance."""

import math
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import yaml

from lore_mcp.application.retrieval_service import LoreRetrievalService
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.evaluation import (
    AggregateMetrics,
    CaseEvaluationResult,
    EvaluationConfig,
    EvaluationReport,
    EvidenceMatch,
    GoldenCase,
)
from lore_mcp.domain.search import Evidence, EvidenceBundle, SearchHit


def compute_evidence_relevance(
    item: Evidence | SearchHit,
    case: GoldenCase,
) -> int:
    """Compute graded relevance score (0, 1, or 2) for an item against a test case.

    Relevance 2 (Exact / Strong):
    - Entity ID match: item's entity matches one in expected_evidence.entity_ids.
    - Scene match: item's season and episode (and scene_index if specified) match.

    Relevance 1 (Partial):
    - Dialogue quote match: verbatim quote appears in item text.
    - Chronology keyword match: any chronological step phrase appears in item text.

    Relevance 0 (Irrelevant):
    - Unanswerable cases or non-matching content.
    """
    if case.case_type == "unanswerable":
        return 0

    expected = case.expected_evidence

    # Extract fields based on whether item is Evidence or SearchHit
    item_entity_ids: list[str] = []
    item_season: int | None = None
    item_episode: int | None = None
    item_scene_index: int | None = None
    content: str = ""

    if isinstance(item, Evidence):
        if item.entity_id is not None:
            item_entity_ids.append(str(item.entity_id))
        item_season = item.season
        item_episode = item.episode
        item_scene_index = item.scene_index
        content = item.content
    else:
        chunk: Chunk = item.chunk
        item_entity_ids.extend([str(eid) for eid in chunk.entity_ids])
        item_season = chunk.season
        item_episode = chunk.episode
        item_scene_index = chunk.scene_index
        content = chunk.text

    # 1. Check exact entity match (Grade 2)
    expected_entities = {str(eid).lower() for eid in expected.entity_ids}
    if expected_entities and any(
        eid.lower() in expected_entities for eid in item_entity_ids
    ):
        return 2

    # 2. Check exact scene match (Grade 2)
    if expected.scenes and item_season is not None and item_episode is not None:
        for scene in expected.scenes:
            if scene.season == item_season and scene.episode == item_episode:
                if scene.scene_index is None or scene.scene_index == item_scene_index:
                    return 2

    # 3. Check dialogue quotation match (Grade 1)
    content_lower = content.lower()
    for quote in expected.dialogue_quotes:
        if quote.lower() in content_lower:
            return 1

    # 4. Check chronological step match (Grade 1)
    for step in expected.chronological_order:
        # Check if prominent subphrases of step appear in content
        tokens = [t.lower() for t in step.split() if len(t) > 3]
        if tokens and sum(1 for t in tokens if t in content_lower) >= 2:
            return 1

    return 0


def detect_cross_series_leaks(
    items: Sequence[Evidence | SearchHit],
    expected_series: str,
) -> int:
    """Count items from another series leaking into results (must be zero)."""
    leaks = 0
    expected_clean = expected_series.strip().lower()
    for item in items:
        series_id = (
            str(item.series_id).lower()
            if isinstance(item, Evidence)
            else str(item.chunk.series_id).lower()
        )
        if series_id != expected_clean:
            leaks += 1
    return leaks


def detect_spoiler_leaks(
    items: Sequence[Evidence | SearchHit],
    season_bound: int | None,
    episode_bound: int | None,
) -> int:
    """Count items leaking beyond temporal/narrative boundary (must be zero)."""
    if season_bound is None and episode_bound is None:
        return 0

    leaks = 0
    for item in items:
        item_season = (
            item.season if isinstance(item, Evidence) else item.chunk.season
        )
        item_episode = (
            item.episode if isinstance(item, Evidence) else item.chunk.episode
        )

        if item_season is None and item_episode is None:
            continue

        if season_bound is not None:
            if item_season is not None and item_season > season_bound:
                leaks += 1
                continue
            if (
                item_season == season_bound
                and episode_bound is not None
                and item_episode is not None
                and item_episode > episode_bound
            ):
                leaks += 1
                continue

        if (
            season_bound is None
            and episode_bound is not None
            and item_episode is not None
            and item_episode > episode_bound
        ):
            leaks += 1
            continue

    return leaks


def calculate_dcg(relevances: Sequence[int], k: int) -> float:
    """Compute Discounted Cumulative Gain up to rank k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        gain = float((1 << rel) - 1)  # 2^rel - 1
        discount = math.log2(i + 2)  # log2(rank + 1), 1-indexed -> i + 2
        dcg += gain / discount
    return dcg


def calculate_ndcg(
    relevances: Sequence[int],
    k: int,
    ideal_relevances: Sequence[int] | None = None,
) -> float:
    """Compute Normalized Discounted Cumulative Gain up to rank k."""
    if not relevances:
        return 0.0

    actual_dcg = calculate_dcg(relevances, k)
    if ideal_relevances is None:
        sorted_ideal = sorted(relevances, reverse=True)
    else:
        sorted_ideal = sorted(ideal_relevances, reverse=True)

    ideal_dcg = calculate_dcg(sorted_ideal, k)
    if ideal_dcg <= 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def calculate_kendall_tau(
    retrieved_ranks: Sequence[int],
    chronological_orders: Sequence[int],
) -> float | None:
    """Compute Kendall's tau rank correlation between retrieval and narrative order."""
    n = len(retrieved_ranks)
    if n < 2 or len(chronological_orders) != n:
        return 1.0 if n == 1 else None

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            rank_diff = retrieved_ranks[i] - retrieved_ranks[j]
            order_diff = chronological_orders[i] - chronological_orders[j]
            prod = rank_diff * order_diff
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1

    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return 1.0
    return (concordant - discordant) / float(total_pairs)


def calculate_percentiles(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Calculate p50, p90, p95, and p99 percentiles from a sequence of float values."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def get_p(p: float) -> float:
        idx = math.ceil(p * n) - 1
        return sorted_vals[max(0, min(n - 1, idx))]

    return get_p(0.50), get_p(0.90), get_p(0.95), get_p(0.99)


def evaluate_single_case(
    case: GoldenCase,
    items: Sequence[Evidence | SearchHit],
    latency_ms: float,
    cache_hit: bool,
    status: str = "ok",
) -> CaseEvaluationResult:
    """Evaluate an individual test case against retrieved items."""
    relevances: list[int] = []
    matches: list[EvidenceMatch] = []
    valid_orders: list[int] = []
    order_ranks: list[int] = []

    for rank_idx, item in enumerate(items, start=1):
        rel = compute_evidence_relevance(item, case)
        relevances.append(rel)

        chunk_id = (
            str(item.chunk_id)
            if isinstance(item, Evidence)
            else str(item.chunk.chunk_id)
        )
        series_id = (
            str(item.series_id)
            if isinstance(item, Evidence)
            else str(item.chunk.series_id)
        )
        narrative_order = (
            item.narrative_order
            if isinstance(item, Evidence)
            else item.chunk.narrative_order
        )
        if narrative_order is not None and rel > 0:
            valid_orders.append(narrative_order)
            order_ranks.append(rank_idx)

        matches.append(
            EvidenceMatch(
                rank=rank_idx,
                chunk_id=chunk_id,
                series_id=series_id,
                relevance_score=rel,
                entity_ids=[
                    str(eid)
                    for eid in (
                        [item.entity_id]
                        if isinstance(item, Evidence) and item.entity_id
                        else item.chunk.entity_ids
                        if isinstance(item, SearchHit)
                        else []
                    )
                ],
                season=item.season if isinstance(item, Evidence) else item.chunk.season,
                episode=(
                    item.episode if isinstance(item, Evidence) else item.chunk.episode
                ),
                scene_index=(
                    item.scene_index
                    if isinstance(item, Evidence)
                    else item.chunk.scene_index
                ),
                narrative_order=narrative_order,
                text_snippet=(
                    item.content[:150]
                    if isinstance(item, Evidence)
                    else item.chunk.text[:150]
                ),
            )
        )

    # 1. Recall@5 and Recall@10
    hit_at_5 = any(r >= 1 for r in relevances[:5])
    hit_at_10 = any(r >= 1 for r in relevances[:10])

    # 2. Reciprocal Rank
    first_relevant_rank: int | None = None
    for idx, r in enumerate(relevances, start=1):
        if r >= 1:
            first_relevant_rank = idx
            break
    reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    # 3. nDCG@5 and nDCG@10
    ndcg_at_5 = calculate_ndcg(relevances, 5)
    ndcg_at_10 = calculate_ndcg(relevances, 10)

    # 4. Zero tolerance Leaks
    cross_series_leaks = detect_cross_series_leaks(items, str(case.series))
    spoiler_leaks = detect_spoiler_leaks(
        items,
        case.season_bound,
        case.episode_bound,
    )

    # 5. Chronology rank correlation
    chronology_tau: float | None = None
    if case.case_type == "chronology" and len(valid_orders) >= 2:
        chronology_tau = calculate_kendall_tau(order_ranks, valid_orders)

    # 6. Fallback Behavior
    fallback_correct = False
    if case.case_type == "unanswerable":
        # System correctly indicates lack of sufficient evidence
        fallback_correct = (
            status in ("insufficient", "fallback_used")
            or len(items) == 0
            or all(r == 0 for r in relevances)
        )
    else:
        # System returns positive status and relevant hits
        fallback_correct = (
            status in ("ok", "cached") and len(items) > 0 and hit_at_10
        )

    return CaseEvaluationResult(
        case_id=case.case_id,
        case_type=case.case_type,
        query=case.query,
        hit_at_5=hit_at_5,
        hit_at_10=hit_at_10,
        reciprocal_rank=reciprocal_rank,
        ndcg_at_5=ndcg_at_5,
        ndcg_at_10=ndcg_at_10,
        cross_series_leaks=cross_series_leaks,
        spoiler_leaks=spoiler_leaks,
        chronology_tau=chronology_tau,
        fallback_correct=fallback_correct,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        matches=matches,
    )


def compute_aggregate_metrics(
    case_results: Sequence[CaseEvaluationResult],
) -> AggregateMetrics:
    """Aggregate individual case results into overall and per-category statistics."""
    total_cases = len(case_results)
    if total_cases == 0:
        return AggregateMetrics(
            total_cases=0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            mrr=0.0,
            ndcg_at_5=0.0,
            ndcg_at_10=0.0,
            total_cross_series_leaks=0,
            total_spoiler_leaks=0,
            mean_chronology_tau=None,
            fallback_accuracy=0.0,
            cache_hit_rate=0.0,
            latency_p50_ms=0.0,
            latency_p90_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            by_case_type={},
        )

    answerable_cases = [r for r in case_results if r.case_type != "unanswerable"]
    num_answerable = len(answerable_cases)

    # Retrieval metrics evaluated over answerable queries
    recall_at_5 = (
        sum(1.0 for r in answerable_cases if r.hit_at_5) / num_answerable
        if num_answerable > 0
        else 0.0
    )
    recall_at_10 = (
        sum(1.0 for r in answerable_cases if r.hit_at_10) / num_answerable
        if num_answerable > 0
        else 0.0
    )
    mrr = (
        sum(r.reciprocal_rank for r in answerable_cases) / num_answerable
        if num_answerable > 0
        else 0.0
    )
    ndcg_at_5 = (
        sum(r.ndcg_at_5 for r in answerable_cases) / num_answerable
        if num_answerable > 0
        else 0.0
    )
    ndcg_at_10 = (
        sum(r.ndcg_at_10 for r in answerable_cases) / num_answerable
        if num_answerable > 0
        else 0.0
    )

    # Leaks (evaluated over all queries)
    total_cross_series_leaks = sum(r.cross_series_leaks for r in case_results)
    total_spoiler_leaks = sum(r.spoiler_leaks for r in case_results)

    # Chronology Kendall's tau
    tau_vals = [
        r.chronology_tau for r in case_results if r.chronology_tau is not None
    ]
    mean_tau = sum(tau_vals) / len(tau_vals) if tau_vals else None

    # Fallback and Cache
    fallback_accuracy = (
        sum(1.0 for r in case_results if r.fallback_correct) / total_cases
    )
    cache_hit_rate = sum(1.0 for r in case_results if r.cache_hit) / total_cases

    # Latencies
    latencies = [r.latency_ms for r in case_results]
    p50, p90, p95, p99 = calculate_percentiles(latencies)

    # Category breakdown
    by_category: dict[str, dict[str, float]] = {}
    distinct_types = {r.case_type for r in case_results}
    for ctype in distinct_types:
        group = [r for r in case_results if r.case_type == ctype]
        g_len = len(group)
        if g_len == 0:
            continue
        by_category[ctype] = {
            "total_cases": float(g_len),
            "recall_at_5": sum(1.0 for r in group if r.hit_at_5) / g_len,
            "recall_at_10": sum(1.0 for r in group if r.hit_at_10) / g_len,
            "mrr": sum(r.reciprocal_rank for r in group) / g_len,
            "ndcg_at_5": sum(r.ndcg_at_5 for r in group) / g_len,
            "ndcg_at_10": sum(r.ndcg_at_10 for r in group) / g_len,
            "cross_series_leaks": float(sum(r.cross_series_leaks for r in group)),
            "spoiler_leaks": float(sum(r.spoiler_leaks for r in group)),
            "fallback_accuracy": sum(1.0 for r in group if r.fallback_correct) / g_len,
        }

    return AggregateMetrics(
        total_cases=total_cases,
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        mrr=mrr,
        ndcg_at_5=ndcg_at_5,
        ndcg_at_10=ndcg_at_10,
        total_cross_series_leaks=total_cross_series_leaks,
        total_spoiler_leaks=total_spoiler_leaks,
        mean_chronology_tau=mean_tau,
        fallback_accuracy=fallback_accuracy,
        cache_hit_rate=cache_hit_rate,
        latency_p50_ms=p50,
        latency_p90_ms=p90,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
        by_case_type=by_category,
    )


class EvaluationRunner:
    """Execution engine for running benchmark datasets through application services."""

    def __init__(self, retrieval_service: LoreRetrievalService) -> None:
        self._retrieval_service = retrieval_service

    def load_dataset(self, path: Path | str) -> list[GoldenCase]:
        """Load and validate golden YAML cases."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Golden dataset not found at {file_path}")
        with open(file_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, list):
            raise ValueError(f"Golden dataset at {file_path} must be a list of cases")
        cases: list[GoldenCase] = []
        raw_list = cast(list[object], raw)
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            cases.append(GoldenCase.model_validate(item))
        return cases

    async def run(self, config: EvaluationConfig) -> EvaluationReport:
        """Execute the full evaluation pipeline and generate a comprehensive report."""
        cases = self.load_dataset(config.dataset_path)
        if config.series:
            cases = [c for c in cases if str(c.series) == config.series]

        case_results: list[CaseEvaluationResult] = []

        for case in cases:
            start_time = time.perf_counter()
            bundle: EvidenceBundle = await self._retrieval_service.search(
                query=case.query,
                series=str(case.series),
                max_season=case.season_bound,
                max_episode=case.episode_bound,
                max_results=config.max_results,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            cache_hit = any("cache" in notice.lower() for notice in bundle.notices)

            result = evaluate_single_case(
                case=case,
                items=bundle.evidence,
                latency_ms=elapsed_ms,
                cache_hit=cache_hit,
                status=bundle.status,
            )
            case_results.append(result)

        metrics = compute_aggregate_metrics(case_results)

        return EvaluationReport(
            timestamp=datetime.now(UTC).isoformat(),
            config=config,
            metrics=metrics,
            case_results=case_results,
        )
