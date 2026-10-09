"""Unit tests for Reciprocal Rank Fusion pure domain function [S3]
and server-side fusion configuration [S2].
"""

from datetime import UTC, datetime
from typing import Any

import pytest
from qdrant_client import AsyncQdrantClient, models

from lore_mcp.adapters.qdrant.index import QdrantVectorIndex
from lore_mcp.adapters.qdrant.initializer import QdrantInitializer
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.config import QdrantSettings
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import (
    SearchHit,
    SearchQuery,
    reciprocal_rank_fusion,
)
from lore_mcp.domain.types import ChunkId, EntityId, SeriesId
from lore_mcp.testing.fakes import FakeVectorIndex


def make_chunk(
    chunk_id: str,
    series_id: str = "loki",
    text: str = "Test chunk text",
    season: int | None = 1,
    episode: int | None = 1,
) -> Chunk:
    return Chunk(
        chunk_id=ChunkId(chunk_id),
        series_id=SeriesId(series_id),
        text=text,
        season=season,
        episode=episode,
        content_hash=f"hash_{chunk_id}",
        entity_ids=[EntityId("loki")],
        provenance=Provenance(
            source_type="transcript",
            source_url="https://example.com",
            retrieval_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )


def make_hit(
    chunk_id: str,
    score: float = 0.5,
    dense_score: float | None = None,
    sparse_score: float | None = None,
    rerank_score: float | None = None,
    text: str = "Sample text",
) -> SearchHit:
    return SearchHit(
        chunk=make_chunk(chunk_id=chunk_id, text=text),
        score=score,
        dense_score=dense_score,
        sparse_score=sparse_score,
        rerank_score=rerank_score,
    )


# ============================================================================
# 1. Pure Domain Function Tests: Reciprocal Rank Fusion [S3]
# ============================================================================


@pytest.mark.unit
def test_rrf_empty_inputs() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []
    assert reciprocal_rank_fusion([[], []]) == []


@pytest.mark.unit
def test_rrf_invalid_k_raises() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        reciprocal_rank_fusion([], k=0)

    with pytest.raises(ValueError, match="must be positive"):
        reciprocal_rank_fusion([], k=-5)


@pytest.mark.unit
def test_rrf_formula_and_ranking_order() -> None:
    # List 1: c1 (rank 1), c2 (rank 2)
    # List 2: c2 (rank 1), c3 (rank 2)
    # With default k=60:
    # c1: 1/(60+1) = 1/61 ≈ 0.016393
    # c2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.016129 + 0.016393 = 0.032522
    # c3: 1/(60+2) = 1/62 ≈ 0.016129
    # Ordering must be: c2, c1, c3
    list1 = [make_hit("c1"), make_hit("c2")]
    list2 = [make_hit("c2"), make_hit("c3")]

    fused = reciprocal_rank_fusion([list1, list2], k=60)

    assert len(fused) == 3
    assert [h.chunk.chunk_id for h in fused] == ["c2", "c1", "c3"]
    assert pytest.approx(fused[0].score, rel=1e-5) == (1 / 62 + 1 / 61)
    assert pytest.approx(fused[1].score, rel=1e-5) == (1 / 61)
    assert pytest.approx(fused[2].score, rel=1e-5) == (1 / 62)


@pytest.mark.unit
def test_rrf_removes_duplicate_chunks() -> None:
    # Same chunk c1 appears in list1, list2, and duplicates within list1
    list1 = [make_hit("c1"), make_hit("c1"), make_hit("c2")]
    list2 = [make_hit("c1"), make_hit("c3")]

    fused = reciprocal_rank_fusion([list1, list2], k=60)

    # Must contain exactly 3 unique chunks: c1, c2, c3
    chunk_ids = [h.chunk.chunk_id for h in fused]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert set(chunk_ids) == {"c1", "c2", "c3"}


@pytest.mark.unit
def test_rrf_preserves_per_signal_scores() -> None:
    # Dense list has dense_score, Sparse list has sparse_score
    dense_hit = make_hit("c1", score=0.95, dense_score=0.95, sparse_score=None)
    sparse_hit = make_hit("c1", score=12.5, dense_score=None, sparse_score=12.5)

    fused = reciprocal_rank_fusion([[dense_hit], [sparse_hit]], k=60)

    assert len(fused) == 1
    fused_hit = fused[0]
    assert fused_hit.chunk.chunk_id == "c1"
    # Preserved both per-signal scores
    assert fused_hit.dense_score == 0.95
    assert fused_hit.sparse_score == 12.5
    assert fused_hit.rerank_score is None
    # Combined score is RRF score
    assert pytest.approx(fused_hit.score, rel=1e-5) == 2 * (1 / 61)


@pytest.mark.unit
def test_rrf_preserves_chunk_payload() -> None:
    chunk = make_chunk(
        "c1",
        series_id="loki",
        text="Unique timeline text",
        season=2,
        episode=4,
    )
    hit = SearchHit(
        chunk=chunk,
        score=0.8,
        dense_score=0.8,
    )

    fused = reciprocal_rank_fusion([[hit]], k=60)
    assert len(fused) == 1
    retrieved_chunk = fused[0].chunk
    assert retrieved_chunk.chunk_id == "c1"
    assert retrieved_chunk.series_id == "loki"
    assert retrieved_chunk.text == "Unique timeline text"
    assert retrieved_chunk.season == 2
    assert retrieved_chunk.episode == 4
    assert retrieved_chunk.entity_ids == [EntityId("loki")]


@pytest.mark.unit
def test_rrf_max_results_capping() -> None:
    list1 = [make_hit(f"c{i}") for i in range(10)]
    fused = reciprocal_rank_fusion([list1], k=60, max_results=4)
    assert len(fused) == 4


# ============================================================================
# 2. Application Service Integration: HybridRetriever with RRF
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hybrid_retriever_retrieve_fuses_candidates() -> None:
    fake_index = FakeVectorIndex()
    retriever = HybridRetriever(fake_index, rrf_k=60)

    # c1 is strong for dense ("Sylvie"), c2 is strong for sparse lexical ("O.B.")
    c1 = make_chunk("c1", text="Sylvie and Loki travel to the Void")
    c2 = make_chunk("c2", text="O.B. repairs the TemPad at the TVA repair shop")
    c3 = make_chunk("c3", text="Sylvie talks with O.B. about time slipping")
    await fake_index.upsert([c1, c2, c3])

    # Query with both terms
    query = SearchQuery(
        series_id=SeriesId("loki"),
        query="Sylvie O.B.",
        max_results=3,
    )

    fused_hits = await retriever.retrieve(query)

    assert len(fused_hits) >= 2
    # Verify no duplicates
    ids = [h.chunk.chunk_id for h in fused_hits]
    assert len(ids) == len(set(ids))

    # Verify per-signal scores preserved on results that appeared in dense or sparse
    for h in fused_hits:
        assert h.score > 0.0
        # Either dense_score, sparse_score, or both must be present
        assert h.dense_score is not None or h.sparse_score is not None

    # Scores strictly descending
    scores = [h.score for h in fused_hits]
    assert scores == sorted(scores, reverse=True)


# ============================================================================
# 3. Server-side Query API Fusion Configuration [S2]
# ============================================================================


@pytest.mark.unit
def test_qdrant_settings_server_side_fusion_defaults() -> None:
    settings = QdrantSettings()
    assert settings.enable_server_side_fusion is False
    assert settings.rrf_k == 60


@pytest.mark.unit
def test_qdrant_settings_server_side_fusion_overrides() -> None:
    settings = QdrantSettings(enable_server_side_fusion=True, rrf_k=30)
    assert settings.enable_server_side_fusion is True
    assert settings.rrf_k == 30

    with pytest.raises(ValueError, match="greater than 0"):
        QdrantSettings(rrf_k=0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_qdrant_server_side_fusion_execution() -> None:
    settings = QdrantSettings(
        url=":memory:",
        chunks_collection="fusion_test_chunks",
        meta_collection="fusion_test_meta",
        embedding_model="all-MiniLM-L6-v2",
        dimension=384,
        inference_backend="fastembed",
        schema_version=1,
        enable_server_side_fusion=True,
    )
    client = AsyncQdrantClient(":memory:")
    await QdrantInitializer.ensure_collections_ready(client, settings)

    def fake_embed(raw_models: Any, is_query: bool = False, batch_size: int = 8) -> Any:
        if isinstance(raw_models, models.FusionQuery):
            yield raw_models
            return
        items = raw_models if isinstance(raw_models, list) else [raw_models]
        for item in items:
            if isinstance(item, models.Prefetch):
                nearest = getattr(item.query, "nearest", item.query)
                model_name = getattr(nearest, "model", None)
                if model_name == "Qdrant/bm25" or "bm25" in str(item.query).lower():
                    vec = models.SparseVector(indices=[0, 1], values=[1.0, 0.5])
                else:
                    vec = [0.01] * 384
                yield models.Prefetch(
                    query=vec,
                    using=item.using,
                    limit=item.limit,
                    filter=item.filter,
                )
            elif isinstance(item, models.PointStruct):
                yield models.PointStruct(
                    id=item.id,
                    vector={
                        "dense": [0.01] * 384,
                        "bm25": models.SparseVector(indices=[0, 1], values=[1.0, 0.5]),
                    },
                    payload=item.payload,
                )
            else:
                yield item

    client._model_embedder.embed_models = fake_embed  # type: ignore[assignment]
    client._embed_models = fake_embed  # type: ignore[assignment]
    index = QdrantVectorIndex(client, settings)

    c1 = make_chunk("c1", "loki", "Loki in TVA")
    c2 = make_chunk("c2", "loki", "Sylvie at Citadel")
    await index.upsert([c1, c2])

    query = SearchQuery(series_id=SeriesId("loki"), query="Sylvie Citadel")

    # index.search uses server-side Query API fusion when enable_server_side_fusion=True
    hits = await index.search(query)
    assert len(hits) >= 1
    assert all(h.chunk.series_id == "loki" for h in hits)
    for h in hits:
        assert h.score > 0.0
