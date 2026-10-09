"""Unit tests for metadata filtering at retrieval time, spoiler bounds,
and unscoped chunk handling across dense, sparse, and hybrid retrievers.
"""

from datetime import UTC, datetime

import pytest

from lore_mcp.application.dense import DenseRetriever
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.application.sparse import SparseRetriever
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import SearchQuery
from lore_mcp.domain.types import ChunkId, EntityId, SeriesId
from lore_mcp.testing.fakes import FakeVectorIndex


def make_chunk(
    chunk_id: str,
    series_id: str = "loki",
    text: str = "Test chunk text",
    season: int | None = 1,
    episode: int | None = 1,
    entity_ids: list[str] | None = None,
    source_type: str = "transcript",
) -> Chunk:
    return Chunk(
        chunk_id=ChunkId(chunk_id),
        series_id=SeriesId(series_id),
        text=text,
        content_hash=f"hash_{chunk_id}",
        season=season,
        episode=episode,
        scene_index=1,
        narrative_order=10,
        entity_ids=[EntityId(e) for e in (entity_ids or [])],
        provenance=Provenance(
            source_type=source_type,
            source_url=f"https://example.com/{chunk_id}",
            license="CC-BY-SA-3.0",
            attribution_text="Test Wiki",
            retrieval_timestamp=datetime.now(UTC),
        ),
    )


@pytest.fixture
def fake_index() -> FakeVectorIndex:
    return FakeVectorIndex()


@pytest.fixture
def dense_retriever(fake_index: FakeVectorIndex) -> DenseRetriever:
    return DenseRetriever(index=fake_index)


@pytest.fixture
def sparse_retriever(fake_index: FakeVectorIndex) -> SparseRetriever:
    return SparseRetriever(index=fake_index)


@pytest.fixture
def hybrid_retriever(fake_index: FakeVectorIndex) -> HybridRetriever:
    return HybridRetriever(index=fake_index)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exact_season_and_episode_filtering(
    fake_index: FakeVectorIndex,
    dense_retriever: DenseRetriever,
    sparse_retriever: SparseRetriever,
) -> None:
    c1 = make_chunk("c1", season=1, episode=1, text="Mobius interrogates Loki")
    c2 = make_chunk("c2", season=1, episode=2, text="Mobius and Loki track variant")
    c3 = make_chunk("c3", season=2, episode=1, text="Mobius notices time slipping")
    await fake_index.upsert([c1, c2, c3])

    query = SearchQuery(series_id=SeriesId("loki"), query="Mobius", season=1, episode=2)

    dense_hits = await dense_retriever.retrieve(query)
    assert len(dense_hits) == 1
    assert dense_hits[0].chunk.chunk_id == ChunkId("c2")

    sparse_hits = await sparse_retriever.retrieve(query)
    assert len(sparse_hits) == 1
    assert sparse_hits[0].chunk.chunk_id == ChunkId("c2")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_entity_and_source_type_filtering(
    fake_index: FakeVectorIndex,
    dense_retriever: DenseRetriever,
    sparse_retriever: SparseRetriever,
) -> None:
    c1 = make_chunk(
        "c1",
        text="Sylvie fights guards",
        entity_ids=["sylvie"],
        source_type="transcript",
    )
    c2 = make_chunk(
        "c2",
        text="Sylvie character overview",
        entity_ids=["sylvie"],
        source_type="fandom",
    )
    c3 = make_chunk(
        "c3",
        text="Loki character overview",
        entity_ids=["loki"],
        source_type="fandom",
    )
    await fake_index.upsert([c1, c2, c3])

    # Filter by entity and source_type
    query = SearchQuery(
        series_id=SeriesId("loki"),
        query="Sylvie",
        entity="sylvie",
        source_type="fandom",
    )

    dense_hits = await dense_retriever.retrieve(query)
    assert len(dense_hits) == 1
    assert dense_hits[0].chunk.chunk_id == ChunkId("c2")

    sparse_hits = await sparse_retriever.retrieve(query)
    assert len(sparse_hits) == 1
    assert sparse_hits[0].chunk.chunk_id == ChunkId("c2")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_spoiler_bounds_exclude_future_seasons_and_episodes(
    fake_index: FakeVectorIndex,
    dense_retriever: DenseRetriever,
) -> None:
    c_s1e6 = make_chunk("c1", season=1, episode=6, text="Loki meets Citadel")
    c_s2e2 = make_chunk("c2", season=2, episode=2, text="Loki in London 1977")
    c_s2e3 = make_chunk("c3", season=2, episode=3, text="Loki in Chicago 1893")
    c_s2e6 = make_chunk("c4", season=2, episode=6, text="Loki takes the throne")
    c_s3e1 = make_chunk("c5", season=3, episode=1, text="Loki future timeline")
    await fake_index.upsert([c_s1e6, c_s2e2, c_s2e3, c_s2e6, c_s3e1])

    # User watched up to Season 2, Episode 2
    query = SearchQuery(
        series_id=SeriesId("loki"),
        query="Loki",
        max_season=2,
        max_episode=2,
    )

    hits = await dense_retriever.retrieve(query)
    hit_ids = {h.chunk.chunk_id for h in hits}
    # Should include S1E6 and S2E2, but exclude S2E3, S2E6, and S3E1
    assert hit_ids == {ChunkId("c1"), ChunkId("c2")}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_spoiler_bounds_with_max_season_only(
    fake_index: FakeVectorIndex,
    dense_retriever: DenseRetriever,
) -> None:
    c_s1e6 = make_chunk("c1", season=1, episode=6, text="Loki meets Citadel")
    c_s2e5 = make_chunk("c2", season=2, episode=5, text="Loki in London 1977")
    c_s3e1 = make_chunk("c3", season=3, episode=1, text="Loki future timeline")
    await fake_index.upsert([c_s1e6, c_s2e5, c_s3e1])

    query = SearchQuery(
        series_id=SeriesId("loki"),
        query="Loki",
        max_season=2,
    )

    hits = await dense_retriever.retrieve(query)
    hit_ids = {h.chunk.chunk_id for h in hits}
    assert hit_ids == {ChunkId("c1"), ChunkId("c2")}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_spoiler_bounds_exclude_unscoped_by_default(
    fake_index: FakeVectorIndex,
    dense_retriever: DenseRetriever,
    sparse_retriever: SparseRetriever,
) -> None:
    c_scoped = make_chunk("c1", season=1, episode=1, text="Loki arrested by TVA")
    c_unscoped = make_chunk(
        "c2", season=None, episode=None, text="General Loki lore without season"
    )
    await fake_index.upsert([c_scoped, c_unscoped])

    # Bound set, include_unscoped is False by default
    query = SearchQuery(
        series_id=SeriesId("loki"),
        query="Loki",
        max_season=1,
        max_episode=1,
        include_unscoped=False,
    )

    dense_hits = await dense_retriever.retrieve(query)
    assert len(dense_hits) == 1
    assert dense_hits[0].chunk.chunk_id == ChunkId("c1")

    sparse_hits = await sparse_retriever.retrieve(query)
    assert len(sparse_hits) == 1
    assert sparse_hits[0].chunk.chunk_id == ChunkId("c1")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_spoiler_bounds_include_unscoped_when_requested(
    fake_index: FakeVectorIndex,
    dense_retriever: DenseRetriever,
    sparse_retriever: SparseRetriever,
) -> None:
    c_scoped = make_chunk("c1", season=1, episode=1, text="Loki arrested by TVA")
    c_unscoped = make_chunk(
        "c2", season=None, episode=None, text="General Loki lore without season"
    )
    await fake_index.upsert([c_scoped, c_unscoped])

    # Bound set, include_unscoped is True
    query = SearchQuery(
        series_id=SeriesId("loki"),
        query="Loki",
        max_season=1,
        max_episode=1,
        include_unscoped=True,
    )

    dense_hits = await dense_retriever.retrieve(query)
    dense_ids = {h.chunk.chunk_id for h in dense_hits}
    assert dense_ids == {ChunkId("c1"), ChunkId("c2")}

    sparse_hits = await sparse_retriever.retrieve(query)
    sparse_ids = {h.chunk.chunk_id for h in sparse_hits}
    assert sparse_ids == {ChunkId("c1"), ChunkId("c2")}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hybrid_parallel_retrieval_applies_all_filters(
    fake_index: FakeVectorIndex,
    hybrid_retriever: HybridRetriever,
) -> None:
    c1 = make_chunk(
        "c1",
        season=1,
        episode=2,
        entity_ids=["mobius"],
        source_type="transcript",
        text="Mobius drinks soda in the archives",
    )
    c2 = make_chunk(
        "c2",
        season=2,
        episode=1,
        entity_ids=["mobius"],
        source_type="transcript",
        text="Mobius time slips in season 2",
    )
    c3 = make_chunk(
        "c3",
        season=1,
        episode=2,
        entity_ids=["loki"],
        source_type="transcript",
        text="Loki drinks water in the archives",
    )
    await fake_index.upsert([c1, c2, c3])

    query = SearchQuery(
        series_id=SeriesId("loki"),
        query="Mobius archives soda",
        season=1,
        episode=2,
        entity="mobius",
        source_type="transcript",
    )

    dense_hits, sparse_hits = await hybrid_retriever.retrieve_parallel(query)
    assert len(dense_hits) == 1
    assert dense_hits[0].chunk.chunk_id == ChunkId("c1")
    assert len(sparse_hits) == 1
    assert sparse_hits[0].chunk.chunk_id == ChunkId("c1")
