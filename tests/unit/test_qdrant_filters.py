"""Unit tests for Qdrant metadata filter builder and spoiler bounds [S10, S11]."""

from datetime import UTC, datetime

import pytest
from qdrant_client import AsyncQdrantClient, models

from lore_mcp.adapters.qdrant.filters import (
    build_filter_from_query,
    build_metadata_filter,
)
from lore_mcp.adapters.qdrant.index import QdrantVectorIndex
from lore_mcp.adapters.qdrant.initializer import QdrantInitializer
from lore_mcp.config import QdrantSettings
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import SearchQuery
from lore_mcp.domain.types import ChunkId, EntityId, SeriesId


def make_test_chunk(
    chunk_id: str,
    series_id: str = "loki",
    season: int | None = 1,
    episode: int | None = 1,
    entity_ids: list[str] | None = None,
    source_type: str = "transcript",
) -> Chunk:
    return Chunk(
        chunk_id=ChunkId(chunk_id),
        series_id=SeriesId(series_id),
        text=f"Chunk {chunk_id}",
        season=season,
        episode=episode,
        scene_index=1,
        narrative_order=100,
        schema_version=1,
        ingest_version=1,
        content_hash=f"hash-{chunk_id}",
        entity_ids=[EntityId(e) for e in (entity_ids or ["loki"])],
        provenance=Provenance(
            source_url="https://example.com/test",
            source_type=source_type,
            retrieval_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )


@pytest.fixture
def filter_settings() -> QdrantSettings:
    return QdrantSettings(
        url=":memory:",
        chunks_collection="filter_test_chunks",
        meta_collection="filter_test_meta",
        embedding_model="all-MiniLM-L6-v2",
        dimension=384,
        inference_backend="fastembed",
        schema_version=1,
    )


async def make_filter_client(settings: QdrantSettings) -> AsyncQdrantClient:
    client = AsyncQdrantClient(":memory:")
    await QdrantInitializer.ensure_collections_ready(client, settings)

    def fake_embed_models(points_or_queries, is_query=False, batch_size=8):
        if is_query:
            yield [0.0] * 384
            return
        for p in points_or_queries:
            yield models.PointStruct(
                id=p.id,
                vector={
                    "dense": [0.01] * 384,
                    "bm25": models.SparseVector(indices=[0], values=[1.0]),
                },
                payload=p.payload,
            )

    client._embed_models = fake_embed_models  # type: ignore[assignment]
    return client


@pytest.mark.unit
def test_series_is_mandatory() -> None:
    with pytest.raises(ValueError, match="series_id is mandatory"):
        build_metadata_filter("")

    with pytest.raises(ValueError, match="series_id is mandatory"):
        build_metadata_filter("   ")

    # Property: series condition is always present
    flt = build_metadata_filter(SeriesId("loki"))
    assert flt.must is not None
    assert any(
        isinstance(cond, models.FieldCondition)
        and cond.key == "series"
        and cond.match == models.MatchValue(value="loki")
        for cond in flt.must
    )


@pytest.mark.unit
def test_equality_filters_structure() -> None:
    flt = build_metadata_filter(
        SeriesId("loki"),
        season=2,
        episode=5,
        entity=EntityId("mobius"),
        entity_type="character",
        source_type="transcript",
    )

    assert flt.must is not None
    keys_and_values = {}
    for cond in flt.must:
        if isinstance(cond, models.FieldCondition) and isinstance(
            cond.match, models.MatchValue
        ):
            keys_and_values[cond.key] = cond.match.value

    assert keys_and_values["series"] == "loki"
    assert keys_and_values["season"] == 2
    assert keys_and_values["episode"] == 5
    assert keys_and_values["entity_id"] == "mobius"
    assert keys_and_values["entity_type"] == "character"
    assert keys_and_values["source_type"] == "transcript"


@pytest.mark.unit
def test_spoiler_bound_filter_structure() -> None:
    # 1. Season and episode bound without unscoped
    flt_scoped = build_metadata_filter(
        SeriesId("loki"),
        max_season=2,
        max_episode=3,
        include_unscoped=False,
    )
    assert flt_scoped.must is not None
    assert isinstance(flt_scoped.should, list)
    assert len(flt_scoped.should) == 2

    # 2. Season and episode bound with unscoped
    flt_unscoped = build_metadata_filter(
        SeriesId("loki"),
        max_season=2,
        max_episode=3,
        include_unscoped=True,
    )
    assert isinstance(flt_unscoped.should, list)
    assert len(flt_unscoped.should) == 3
    assert any(isinstance(c, models.IsEmptyCondition) for c in flt_unscoped.should)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_in_memory_qdrant_spoiler_bounds_filtering(
    filter_settings: QdrantSettings,
) -> None:
    client = await make_filter_client(filter_settings)
    index = QdrantVectorIndex(client, filter_settings)

    # Ingest chunks covering multiple seasons, episodes, unscoped, and foreign series
    c_s1e6 = make_test_chunk("c1", "loki", season=1, episode=6)
    c_s2e2 = make_test_chunk("c2", "loki", season=2, episode=2)
    c_s2e3 = make_test_chunk("c3", "loki", season=2, episode=3)
    c_s2e5 = make_test_chunk("c4", "loki", season=2, episode=5)
    c_s3e1 = make_test_chunk("c5", "loki", season=3, episode=1)
    c_unscoped = make_test_chunk("c6", "loki", season=None, episode=None)
    c_starwars = make_test_chunk("c7", "star-wars", season=1, episode=1)

    await index.upsert(
        [
            c_s1e6,
            c_s2e2,
            c_s2e3,
            c_s2e5,
            c_s3e1,
            c_unscoped,
            c_starwars,
        ]
    )

    # 1. Up to S2E3, without unscoped: only c1 (S1E6), c2 (S2E2), and c3 (S2E3)
    query_scoped = SearchQuery(
        series_id=SeriesId("loki"),
        query="loki",
        max_season=2,
        max_episode=3,
        include_unscoped=False,
    )
    hits_scoped = await index.search(query_scoped)
    scoped_ids = {h.chunk.chunk_id for h in hits_scoped}
    assert scoped_ids == {ChunkId("c1"), ChunkId("c2"), ChunkId("c3")}

    # 2. Up to S2E3, with unscoped: c1, c2, c3, and c6
    query_unscoped = SearchQuery(
        series_id=SeriesId("loki"),
        query="loki",
        max_season=2,
        max_episode=3,
        include_unscoped=True,
    )
    hits_unscoped = await index.search(query_unscoped)
    unscoped_ids = {h.chunk.chunk_id for h in hits_unscoped}
    assert unscoped_ids == {
        ChunkId("c1"),
        ChunkId("c2"),
        ChunkId("c3"),
        ChunkId("c6"),
    }

    # 3. Up to S1 only (all episodes), without unscoped: only c1 (S1E6)
    query_s1 = SearchQuery(
        series_id=SeriesId("loki"),
        query="loki",
        max_season=1,
        include_unscoped=False,
    )
    hits_s1 = await index.search(query_s1)
    s1_ids = {h.chunk.chunk_id for h in hits_s1}
    assert s1_ids == {ChunkId("c1")}

    # 4. Exact equality: Season 2, Episode 5: only c4
    query_exact = SearchQuery(
        series_id=SeriesId("loki"),
        query="loki",
        season=2,
        episode=5,
    )
    hits_exact = await index.search(query_exact)
    assert len(hits_exact) == 1
    assert hits_exact[0].chunk.chunk_id == ChunkId("c4")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_entity_and_source_type_filters(
    filter_settings: QdrantSettings,
) -> None:
    client = await make_filter_client(filter_settings)
    index = QdrantVectorIndex(client, filter_settings)

    c1 = make_test_chunk(
        "c1",
        "loki",
        entity_ids=["loki", "mobius"],
        source_type="transcript",
    )
    c2 = make_test_chunk(
        "c2",
        "loki",
        entity_ids=["sylvie"],
        source_type="fandom",
    )

    await index.upsert([c1, c2])

    # Filter by entity "mobius"
    query_mobius = SearchQuery(
        series_id=SeriesId("loki"),
        query="test",
        entity="mobius",
    )
    hits_mobius = await index.search(query_mobius)
    assert len(hits_mobius) == 1
    assert hits_mobius[0].chunk.chunk_id == ChunkId("c1")

    # Filter by source_type "fandom"
    query_fandom = SearchQuery(
        series_id=SeriesId("loki"),
        query="test",
        source_type="fandom",
    )
    hits_fandom = await index.search(query_fandom)
    assert len(hits_fandom) == 1
    assert hits_fandom[0].chunk.chunk_id == ChunkId("c2")


@pytest.mark.unit
def test_build_filter_from_query_helper() -> None:
    query = SearchQuery(
        series_id=SeriesId("loki"),
        query="test",
        season=1,
        episode=2,
        entity="tva",
        entity_type="organization",
        source_type="transcript",
    )
    flt = build_filter_from_query(query)
    assert flt.must is not None

    keys = [cond.key for cond in flt.must if isinstance(cond, models.FieldCondition)]
    assert "series" in keys
    assert "season" in keys
    assert "episode" in keys
    assert "entity_id" in keys
    assert "entity_type" in keys
    assert "source_type" in keys
