"""Unit tests for retrieval quality metrics and leak detection (Phase 11.3 & 11.6)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from lore_mcp.application.evaluation_service import (
    EvaluationRunner,
    calculate_dcg,
    calculate_kendall_tau,
    calculate_ndcg,
    calculate_percentiles,
    compute_aggregate_metrics,
    compute_evidence_relevance,
    detect_cross_series_leaks,
    detect_spoiler_leaks,
    evaluate_single_case,
)
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.evaluation import (
    CaseEvaluationResult,
    EvaluationConfig,
    ExpectedEvidence,
    GoldenCase,
    SceneLocator,
)
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import Evidence, EvidenceBundle
from lore_mcp.domain.types import ChunkId, EntityId, SeriesId


def _create_test_chunk(
    chunk_id: str = "c1",
    series_id: str = "loki",
    text: str = "Sample lore text",
    entity_ids: list[str] | None = None,
    season: int | None = None,
    episode: int | None = None,
    scene_index: int | None = None,
    narrative_order: int | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=ChunkId(chunk_id),
        series_id=SeriesId(series_id),
        text=text,
        content_hash="hash-abc",
        entity_ids=[EntityId(e) for e in (entity_ids or [])],
        season=season,
        episode=episode,
        scene_index=scene_index,
        narrative_order=narrative_order,
        provenance=Provenance(
            source_url="https://example.com",
            source_type="transcript",
            retrieval_timestamp=datetime.now(UTC),
        ),
    )


def _create_test_evidence(
    chunk_id: str = "c1",
    series_id: str = "loki",
    content: str = "Sample evidence content",
    entity_id: str | None = None,
    season: int | None = None,
    episode: int | None = None,
    scene_index: int | None = None,
    narrative_order: int | None = None,
) -> Evidence:
    return Evidence(
        chunk_id=ChunkId(chunk_id),
        series_id=SeriesId(series_id),
        content=content,
        entity_id=EntityId(entity_id) if entity_id else None,
        season=season,
        episode=episode,
        scene_index=scene_index,
        narrative_order=narrative_order,
        score=0.95,
        provenance=Provenance(
            source_url="https://example.com",
            source_type="transcript",
            retrieval_timestamp=datetime.now(UTC),
        ),
    )


def test_compute_evidence_relevance_entity_match() -> None:
    case = GoldenCase(
        case_id="c1",
        query="Who is Sylvie?",
        series=SeriesId("loki"),
        case_type="alias_entity",
        expected_evidence=ExpectedEvidence(entity_ids=[EntityId("sylvie")]),
    )

    ev_match = _create_test_evidence(entity_id="sylvie")
    ev_mismatch = _create_test_evidence(entity_id="mobius")

    assert compute_evidence_relevance(ev_match, case) == 2
    assert compute_evidence_relevance(ev_mismatch, case) == 0


def test_compute_evidence_relevance_scene_match() -> None:
    case = GoldenCase(
        case_id="c2",
        query="What happened at Pompeii?",
        series=SeriesId("loki"),
        case_type="simple_fact",
        expected_evidence=ExpectedEvidence(
            scenes=[SceneLocator(season=1, episode=2, scene_index=3)]
        ),
    )

    ev_exact = _create_test_evidence(season=1, episode=2, scene_index=3)
    ev_wrong_scene = _create_test_evidence(season=1, episode=2, scene_index=4)
    ev_wrong_ep = _create_test_evidence(season=1, episode=3, scene_index=3)

    assert compute_evidence_relevance(ev_exact, case) == 2
    assert compute_evidence_relevance(ev_wrong_scene, case) == 0
    assert compute_evidence_relevance(ev_wrong_ep, case) == 0


def test_compute_evidence_relevance_dialogue_quote() -> None:
    case = GoldenCase(
        case_id="c3",
        query="Love metaphor quote",
        series=SeriesId("loki"),
        case_type="dialogue",
        expected_evidence=ExpectedEvidence(
            dialogue_quotes=["Love is a dagger", "makes you bleed"]
        ),
    )

    ev_quote = _create_test_evidence(
        content="He said that love is a dagger that can hurt."
    )
    ev_none = _create_test_evidence(content="He gave a speech about leadership.")

    assert compute_evidence_relevance(ev_quote, case) == 1
    assert compute_evidence_relevance(ev_none, case) == 0


def test_detect_cross_series_leaks() -> None:
    e1 = _create_test_evidence(series_id="loki")
    e2 = _create_test_evidence(series_id="wandavision")
    e3 = _create_test_evidence(series_id="loki")

    assert detect_cross_series_leaks([e1, e3], "loki") == 0
    assert detect_cross_series_leaks([e1, e2, e3], "loki") == 1
    assert detect_cross_series_leaks([e2], "loki") == 1


def test_detect_spoiler_leaks() -> None:
    e_s1e1 = _create_test_evidence(season=1, episode=1)
    e_s1e2 = _create_test_evidence(season=1, episode=2)
    e_s1e4 = _create_test_evidence(season=1, episode=4)
    e_s2e1 = _create_test_evidence(season=2, episode=1)

    # Max season=1, episode=2:
    assert detect_spoiler_leaks([e_s1e1, e_s1e2], season_bound=1, episode_bound=2) == 0
    assert detect_spoiler_leaks([e_s1e1, e_s1e4], season_bound=1, episode_bound=2) == 1
    assert (
        detect_spoiler_leaks([e_s1e1, e_s1e4, e_s2e1], season_bound=1, episode_bound=2)
        == 2
    )

    # No bounds: 0 leaks
    assert (
        detect_spoiler_leaks(
            [e_s1e1, e_s1e4, e_s2e1], season_bound=None, episode_bound=None
        )
        == 0
    )


def test_calculate_dcg_and_ndcg() -> None:
    # Perfect ranking: [2, 2, 1, 0]
    perfect_rels = [2, 2, 1, 0]
    dcg_perfect = calculate_dcg(perfect_rels, 4)
    assert dcg_perfect > 0.0

    ndcg_perfect = calculate_ndcg(perfect_rels, 4)
    assert pytest.approx(ndcg_perfect, 0.001) == 1.0

    # Suboptimal ranking: [0, 1, 2, 2]
    subopt_rels = [0, 1, 2, 2]
    ndcg_subopt = calculate_ndcg(subopt_rels, 4)
    assert 0.0 < ndcg_subopt < 1.0

    # Empty
    assert calculate_ndcg([], 5) == 0.0
    assert calculate_ndcg([0, 0, 0], 3) == 0.0


def test_calculate_kendall_tau() -> None:
    # Identical ordering: tau = 1.0
    ranks = [1, 2, 3, 4]
    orders = [10, 20, 30, 40]
    assert calculate_kendall_tau(ranks, orders) == 1.0

    # Inverted ordering: tau = -1.0
    rev_orders = [40, 30, 20, 10]
    assert calculate_kendall_tau(ranks, rev_orders) == -1.0

    # Single item
    assert calculate_kendall_tau([1], [10]) == 1.0
    assert calculate_kendall_tau([], []) is None


def test_calculate_percentiles() -> None:
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    p50, p90, p95, p99 = calculate_percentiles(latencies)

    assert p50 == 50.0
    assert p90 == 90.0
    assert p95 == 100.0
    assert p99 == 100.0

    empty_p = calculate_percentiles([])
    assert empty_p == (0.0, 0.0, 0.0, 0.0)


def test_evaluate_single_case_metrics() -> None:
    case = GoldenCase(
        case_id="loki-fact-001",
        case_type="simple_fact",
        series=SeriesId("loki"),
        query="Who created the TVA?",
        expected_evidence=ExpectedEvidence(entity_ids=[EntityId("he-who-remains")]),
    )

    hit1 = _create_test_evidence(chunk_id="c1", entity_id="other")
    hit2 = _create_test_evidence(chunk_id="c2", entity_id="he-who-remains")
    hit3 = _create_test_evidence(chunk_id="c3", entity_id="other")

    res = evaluate_single_case(
        case=case,
        items=[hit1, hit2, hit3],
        latency_ms=45.0,
        cache_hit=False,
        status="ok",
    )

    assert res.case_id == "loki-fact-001"
    assert res.hit_at_5 is True
    assert res.hit_at_10 is True
    assert res.reciprocal_rank == 0.5  # First hit is at rank 2
    assert res.cross_series_leaks == 0
    assert res.spoiler_leaks == 0
    assert res.fallback_correct is True


def test_compute_aggregate_metrics() -> None:
    r1 = CaseEvaluationResult(
        case_id="q1",
        case_type="simple_fact",
        query="query 1",
        hit_at_5=True,
        hit_at_10=True,
        reciprocal_rank=1.0,
        ndcg_at_5=1.0,
        ndcg_at_10=1.0,
        cross_series_leaks=0,
        spoiler_leaks=0,
        chronology_tau=None,
        fallback_correct=True,
        latency_ms=10.0,
        cache_hit=True,
    )
    r2 = CaseEvaluationResult(
        case_id="q2",
        case_type="dialogue",
        query="query 2",
        hit_at_5=False,
        hit_at_10=True,
        reciprocal_rank=0.1,
        ndcg_at_5=0.2,
        ndcg_at_10=0.6,
        cross_series_leaks=0,
        spoiler_leaks=0,
        chronology_tau=None,
        fallback_correct=True,
        latency_ms=30.0,
        cache_hit=False,
    )
    r3 = CaseEvaluationResult(
        case_id="q3",
        case_type="unanswerable",
        query="query 3",
        hit_at_5=False,
        hit_at_10=False,
        reciprocal_rank=0.0,
        ndcg_at_5=0.0,
        ndcg_at_10=0.0,
        cross_series_leaks=0,
        spoiler_leaks=0,
        chronology_tau=None,
        fallback_correct=True,
        latency_ms=5.0,
        cache_hit=False,
    )

    metrics = compute_aggregate_metrics([r1, r2, r3])

    assert metrics.total_cases == 3
    # Answerable cases: r1 and r2
    assert metrics.recall_at_5 == 0.5
    assert metrics.recall_at_10 == 1.0
    assert metrics.mrr == 0.55  # (1.0 + 0.1) / 2
    assert metrics.total_cross_series_leaks == 0
    assert metrics.total_spoiler_leaks == 0
    assert metrics.fallback_accuracy == 1.0
    assert pytest.approx(metrics.cache_hit_rate, 0.01) == 0.333
    assert "simple_fact" in metrics.by_case_type
    assert "dialogue" in metrics.by_case_type
    assert "unanswerable" in metrics.by_case_type


@pytest.mark.asyncio
async def test_evaluation_runner_execution() -> None:
    mock_retrieval_svc = AsyncMock()
    mock_bundle = EvidenceBundle(
        query="Who created the TVA?",
        series_id=SeriesId("loki"),
        status="ok",
        evidence=[
            _create_test_evidence(
                chunk_id="chunk-1",
                entity_id="he-who-remains",
                season=1,
                episode=6,
            )
        ],
        notices=["Cache hit"],
    )
    mock_retrieval_svc.search.return_value = mock_bundle

    runner = EvaluationRunner(retrieval_service=mock_retrieval_svc)
    config = EvaluationConfig(
        dataset_path="eval/golden/loki.yaml",
        series="loki",
        max_results=5,
    )

    report = await runner.run(config)

    assert report.metrics.total_cases == 42
    assert report.metrics.recall_at_5 > 0.0
    assert mock_retrieval_svc.search.call_count == 42
