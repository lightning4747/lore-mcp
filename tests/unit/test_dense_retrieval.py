"""Unit tests for dense vector retrieval service,
series isolation, and ranking.
"""

from datetime import UTC, datetime

import pytest

from lore_mcp.application.dense import DenseRetriever
from lore_mcp.application.errors import InvalidArgument
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import SearchQuery
from lore_mcp.domain.types import ChunkId, EntityId, SeriesId
from lore_mcp.testing.fakes import FakeVectorIndex


def make_chunk(
    chunk_id: str,
    series_id: str,
    text: str,
    season: int | None = None,
    episode: int | None = None,
    entity_ids: list[str] | None = None,
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
            source_type="fandom",
            source_url=f"https://wiki.example.com/{chunk_id}",
            license="CC-BY-SA-3.0",
            attribution_text="Test Wiki",
            retrieval_timestamp=datetime.now(UTC),
        ),
    )


@pytest.fixture
def fake_index() -> FakeVectorIndex:
    return FakeVectorIndex()


@pytest.fixture
def retriever(fake_index: FakeVectorIndex) -> DenseRetriever:
    return DenseRetriever(index=fake_index)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dense_retrieval_mandatory_series_isolation(
    fake_index: FakeVectorIndex, retriever: DenseRetriever
) -> None:
    c_loki = make_chunk("c1", "loki", "Sylvie and Loki meet He Who Remains")
    c_andor = make_chunk("c2", "andor", "Cassian escapes the Imperial prison")
    c_leak = make_chunk("c3", "star-wars", "Loki in a completely wrong series")
    await fake_index.upsert([c_loki, c_andor, c_leak])

    query = SearchQuery(series_id=SeriesId("loki"), query="Sylvie")
    hits = await retriever.retrieve(query)

    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == ChunkId("c1")
    assert hits[0].chunk.series_id == SeriesId("loki")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dense_retrieval_ranked_candidates_with_cosine_scores(
    fake_index: FakeVectorIndex, retriever: DenseRetriever
) -> None:
    c1 = make_chunk("c1", "loki", "Loki talks to Mobius about time slipping")
    c2 = make_chunk("c2", "loki", "Loki weaves the branches of the multiverse")
    await fake_index.upsert([c1, c2])

    query = SearchQuery(series_id=SeriesId("loki"), query="Mobius")
    hits = await retriever.retrieve(query)

    assert len(hits) == 2
    # Verify scores exist and are ranked descending
    assert hits[0].dense_score is not None
    assert hits[1].dense_score is not None
    assert hits[0].score >= hits[1].score
    assert hits[0].score == hits[0].dense_score
    assert hits[0].chunk.chunk_id == ChunkId("c1")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dense_retrieval_preserves_metadata(
    fake_index: FakeVectorIndex, retriever: DenseRetriever
) -> None:
    chunk = make_chunk(
        "c-detailed",
        "loki",
        "Loki prunes himself in the TVA loom chamber",
        season=2,
        episode=4,
        entity_ids=["loki", "mobius", "temporal-loom"],
    )
    await fake_index.upsert([chunk])

    query = SearchQuery(series_id=SeriesId("loki"), query="prune")
    hits = await retriever.retrieve(query)
    assert len(hits) == 1

    retrieved = hits[0].chunk
    assert retrieved.chunk_id == ChunkId("c-detailed")
    assert retrieved.series_id == SeriesId("loki")
    assert retrieved.season == 2
    assert retrieved.episode == 4
    assert retrieved.scene_index == 1
    assert retrieved.narrative_order == 10
    assert retrieved.entity_ids == [
        EntityId("loki"),
        EntityId("mobius"),
        EntityId("temporal-loom"),
    ]
    assert retrieved.provenance.license == "CC-BY-SA-3.0"
    assert retrieved.provenance.attribution_text == "Test Wiki"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dense_retrieval_respects_max_results(
    fake_index: FakeVectorIndex, retriever: DenseRetriever
) -> None:
    chunks = [
        make_chunk(f"c{i}", "loki", f"Loki timeline sequence {i}") for i in range(10)
    ]
    await fake_index.upsert(chunks)

    query = SearchQuery(series_id=SeriesId("loki"), query="sequence", max_results=3)
    hits = await retriever.retrieve(query)

    assert len(hits) == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dense_retrieval_empty_query_raises_invalid_argument(
    retriever: DenseRetriever,
) -> None:
    query = SearchQuery.model_construct(
        series_id=SeriesId("loki"),
        query="   ",
        max_results=5,
    )
    with pytest.raises(InvalidArgument) as exc_info:
        await retriever.retrieve(query)
    assert exc_info.value.argument_name == "query"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dense_retrieval_empty_results(
    fake_index: FakeVectorIndex, retriever: DenseRetriever
) -> None:
    hits = await retriever.retrieve(
        SearchQuery(series_id=SeriesId("unknown-series"), query="test")
    )
    assert hits == []
