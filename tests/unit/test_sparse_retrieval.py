"""Unit tests for sparse BM25 vector retrieval service and parallel execution."""

from datetime import UTC, datetime

import pytest

from lore_mcp.application.errors import InvalidArgument
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.application.sparse import SparseRetriever
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
def sparse_retriever(fake_index: FakeVectorIndex) -> SparseRetriever:
    return SparseRetriever(index=fake_index)


@pytest.fixture
def hybrid_retriever(fake_index: FakeVectorIndex) -> HybridRetriever:
    return HybridRetriever(index=fake_index)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sparse_retrieval_mandatory_series_isolation(
    fake_index: FakeVectorIndex, sparse_retriever: SparseRetriever
) -> None:
    c_loki = make_chunk("c1", "loki", "Sylvie encounters the TVA in the Citadel")
    c_andor = make_chunk("c2", "andor", "Cassian encounters the Imperial forces")
    c_leak = make_chunk("c3", "star-wars", "Sylvie in another galaxy")
    await fake_index.upsert([c_loki, c_andor, c_leak])

    query = SearchQuery(series_id=SeriesId("loki"), query="Sylvie Citadel")
    hits = await sparse_retriever.retrieve(query)

    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == ChunkId("c1")
    assert hits[0].chunk.series_id == SeriesId("loki")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sparse_retrieval_lexical_scoring_and_ranking(
    fake_index: FakeVectorIndex, sparse_retriever: SparseRetriever
) -> None:
    c1 = make_chunk("c1", "loki", "Mobius drinks Josta soda at the TVA archives")
    c2 = make_chunk("c2", "loki", "Mobius talks to Loki about jet skis")
    c3 = make_chunk("c3", "loki", "Sylvie wields her blade against the timeline")
    await fake_index.upsert([c1, c2, c3])

    # Query with multiple matching terms for c1 and one term for c2
    query = SearchQuery(series_id=SeriesId("loki"), query="Mobius Josta soda")
    hits = await sparse_retriever.retrieve(query)

    # c3 has 0 term overlap so it must not be in lexical hits
    assert len(hits) == 2
    assert hits[0].chunk.chunk_id == ChunkId("c1")
    assert hits[1].chunk.chunk_id == ChunkId("c2")

    # Scores must be populated and sorted descending
    assert hits[0].sparse_score is not None
    assert hits[1].sparse_score is not None
    assert hits[0].score > hits[1].score
    assert hits[0].score == hits[0].sparse_score


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sparse_retrieval_preserves_metadata(
    fake_index: FakeVectorIndex, sparse_retriever: SparseRetriever
) -> None:
    chunk = make_chunk(
        "c-detailed",
        "loki",
        "Victor Timely unveils the Throughput Multiplier device",
        season=2,
        episode=3,
        entity_ids=["victor-timely", "temporal-loom"],
    )
    await fake_index.upsert([chunk])

    query = SearchQuery(series_id=SeriesId("loki"), query="Victor Timely Multiplier")
    hits = await sparse_retriever.retrieve(query)
    assert len(hits) == 1

    retrieved = hits[0].chunk
    assert retrieved.chunk_id == ChunkId("c-detailed")
    assert retrieved.series_id == SeriesId("loki")
    assert retrieved.season == 2
    assert retrieved.episode == 3
    assert retrieved.scene_index == 1
    assert retrieved.narrative_order == 10
    assert retrieved.entity_ids == [
        EntityId("victor-timely"),
        EntityId("temporal-loom"),
    ]
    assert retrieved.provenance.license == "CC-BY-SA-3.0"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sparse_retrieval_respects_max_results(
    fake_index: FakeVectorIndex, sparse_retriever: SparseRetriever
) -> None:
    chunks = [
        make_chunk(f"c{i}", "loki", f"Timeline variant branch number {i}")
        for i in range(10)
    ]
    await fake_index.upsert(chunks)

    query = SearchQuery(
        series_id=SeriesId("loki"), query="variant branch", max_results=4
    )
    hits = await sparse_retriever.retrieve(query)

    assert len(hits) == 4


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sparse_retrieval_empty_query_raises_invalid_argument(
    sparse_retriever: SparseRetriever,
) -> None:
    query = SearchQuery.model_construct(
        series_id=SeriesId("loki"),
        query="   ",
        max_results=5,
    )
    with pytest.raises(InvalidArgument) as exc_info:
        await sparse_retriever.retrieve(query)
    assert exc_info.value.argument_name == "query"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sparse_retrieval_empty_results(
    fake_index: FakeVectorIndex, sparse_retriever: SparseRetriever
) -> None:
    c1 = make_chunk("c1", "loki", "Loki in TVA")
    await fake_index.upsert([c1])

    hits = await sparse_retriever.retrieve(
        SearchQuery(series_id=SeriesId("loki"), query="completely unmatched terms")
    )
    assert hits == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hybrid_retriever_runs_dense_and_sparse_in_parallel(
    fake_index: FakeVectorIndex, hybrid_retriever: HybridRetriever
) -> None:
    c1 = make_chunk("c1", "loki", "Loki weaves the temporal branches")
    c2 = make_chunk("c2", "loki", "He Who Remains sits at the end of time")
    await fake_index.upsert([c1, c2])

    query = SearchQuery(series_id=SeriesId("loki"), query="temporal branches")
    dense_hits, sparse_hits = await hybrid_retriever.retrieve_parallel(query)

    assert len(dense_hits) > 0
    assert len(sparse_hits) > 0
    assert all(h.dense_score is not None for h in dense_hits)
    assert all(h.sparse_score is not None for h in sparse_hits)
    assert dense_hits[0].chunk.series_id == SeriesId("loki")
    assert sparse_hits[0].chunk.series_id == SeriesId("loki")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hybrid_retriever_empty_query_validation(
    hybrid_retriever: HybridRetriever,
) -> None:
    query = SearchQuery.model_construct(
        series_id=SeriesId("loki"),
        query="",
    )
    with pytest.raises(InvalidArgument):
        await hybrid_retriever.retrieve_parallel(query)
