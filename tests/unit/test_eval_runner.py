"""Unit tests for the automated evaluation runner and metrics calculations."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from eval.run_eval import (
    GoldenQuery,
    QueryEvaluationResult,
    compute_metrics,
    evaluate_single_query,
    is_negative_rejected,
)

from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import SearchHit
from lore_mcp.domain.types import ChunkId, EntityId, SeriesId


def _create_dummy_chunk(chunk_id: str, text: str = "test chunk") -> Chunk:
    return Chunk(
        chunk_id=ChunkId(chunk_id),
        series_id=SeriesId("loki"),
        text=text,
        content_hash="hash-123",
        entity_ids=[EntityId("loki")],
        provenance=Provenance(
            source_url="https://example.com",
            source_type="transcript",
            retrieval_timestamp=datetime.now(UTC),
        ),
    )


def test_is_negative_rejected_dense() -> None:
    hit_low = SearchHit(chunk=_create_dummy_chunk("c1"), score=0.25)
    hit_high = SearchHit(chunk=_create_dummy_chunk("c2"), score=0.65)

    assert is_negative_rejected("dense", []) is True
    assert is_negative_rejected("dense", [hit_low]) is True
    assert is_negative_rejected("dense", [hit_high]) is False


def test_is_negative_rejected_sparse() -> None:
    hit_low = SearchHit(chunk=_create_dummy_chunk("c1"), score=4.5)
    hit_high = SearchHit(chunk=_create_dummy_chunk("c2"), score=12.0)

    assert is_negative_rejected("sparse", []) is True
    assert is_negative_rejected("sparse", [hit_low]) is True
    assert is_negative_rejected("sparse", [hit_high]) is False


def test_is_negative_rejected_hybrid() -> None:
    hit_rejected = SearchHit(
        chunk=_create_dummy_chunk("c1"),
        score=0.03,
        dense_score=0.20,
        sparse_score=4.0,
    )
    hit_dense_pass = SearchHit(
        chunk=_create_dummy_chunk("c2"),
        score=0.03,
        dense_score=0.55,
        sparse_score=4.0,
    )
    hit_sparse_pass = SearchHit(
        chunk=_create_dummy_chunk("c3"),
        score=0.03,
        dense_score=0.20,
        sparse_score=15.0,
    )

    assert is_negative_rejected("hybrid", []) is True
    assert is_negative_rejected("hybrid", [hit_rejected]) is True
    assert is_negative_rejected("hybrid", [hit_dense_pass]) is False
    assert is_negative_rejected("hybrid", [hit_sparse_pass]) is False


@pytest.mark.asyncio
async def test_evaluate_single_query_positive_hit() -> None:
    golden = GoldenQuery(
        id="test-001",
        category="dialogue",
        query="Burdened with glorious purpose",
        series="loki",
        expected_chunk_ids=["chunk-target"],
    )

    mock_search = AsyncMock(
        return_value=[
            SearchHit(chunk=_create_dummy_chunk("chunk-other-1"), score=0.9),
            SearchHit(chunk=_create_dummy_chunk("chunk-target"), score=0.8),
        ]
    )

    result = await evaluate_single_query(golden, "hybrid", mock_search, k=5)

    assert result.query_id == "test-001"
    assert result.first_relevant_rank == 2
    assert result.hit_at_1 is False
    assert result.hit_at_3 is True
    assert result.hit_at_5 is True
    assert result.reciprocal_rank == 0.5
    assert result.is_negative is False


@pytest.mark.asyncio
async def test_evaluate_single_query_negative_rejected() -> None:
    golden = GoldenQuery(
        id="test-neg-001",
        category="negative",
        query="Batman batmobile Gotham",
        series="loki",
        expected_chunk_ids=[],
    )

    mock_search = AsyncMock(
        return_value=[
            SearchHit(chunk=_create_dummy_chunk("chunk-unrelated"), score=0.22)
        ]
    )

    result = await evaluate_single_query(golden, "dense", mock_search, k=5)

    assert result.query_id == "test-neg-001"
    assert result.is_negative is True
    assert result.negative_rejected is True
    assert result.first_relevant_rank is None
    assert result.reciprocal_rank == 0.0


def test_compute_metrics_calculation() -> None:
    r1 = QueryEvaluationResult(
        query_id="q1",
        category="entity",
        query="Query 1",
        pipeline="dense",
        expected_chunk_ids=["c1"],
        retrieved_chunk_ids=["c1"],
        scores=[0.9],
        first_relevant_rank=1,
        hit_at_1=True,
        hit_at_3=True,
        hit_at_5=True,
        reciprocal_rank=1.0,
        is_negative=False,
        negative_rejected=False,
    )
    r2 = QueryEvaluationResult(
        query_id="q2",
        category="event",
        query="Query 2",
        pipeline="dense",
        expected_chunk_ids=["c2"],
        retrieved_chunk_ids=["other", "c2"],
        scores=[0.9, 0.8],
        first_relevant_rank=2,
        hit_at_1=False,
        hit_at_3=True,
        hit_at_5=True,
        reciprocal_rank=0.5,
        is_negative=False,
        negative_rejected=False,
    )
    r3 = QueryEvaluationResult(
        query_id="q3",
        category="negative",
        query="Query 3",
        pipeline="dense",
        expected_chunk_ids=[],
        retrieved_chunk_ids=["other"],
        scores=[0.2],
        first_relevant_rank=None,
        hit_at_1=False,
        hit_at_3=False,
        hit_at_5=False,
        reciprocal_rank=0.0,
        is_negative=True,
        negative_rejected=True,
    )

    metrics = compute_metrics("Dense", [r1, r2, r3])

    assert metrics.pipeline_name == "Dense"
    assert metrics.total_queries == 3
    assert metrics.positive_queries == 2
    assert metrics.negative_queries == 1
    assert metrics.hit_rate_at_1 == 0.5
    assert metrics.hit_rate_at_3 == 1.0
    assert metrics.hit_rate_at_5 == 1.0
    assert metrics.mrr == 0.75  # (1.0 + 0.5) / 2
    assert metrics.negative_scope_precision == 1.0
    assert "entity" in metrics.category_breakdown
    assert "event" in metrics.category_breakdown


def test_golden_dataset_structure() -> None:
    dataset_file = Path("eval/golden_dataset.json")
    assert dataset_file.exists()

    with open(dataset_file, encoding="utf-8") as f:
        import json

        data = json.load(f)

    assert len(data) >= 20
    queries = [GoldenQuery.model_validate(item) for item in data]
    assert len(queries) == 30

    categories = {q.category for q in queries}
    assert categories == {"entity", "event", "dialogue", "negative"}
