"""Unit tests for QdrantVectorIndex adapter [S7, S8]."""

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient, models

from lore_mcp.adapters.qdrant.errors import QdrantOperationError
from lore_mcp.adapters.qdrant.index import (
    QdrantVectorIndex,
    resolve_embedding_model_name,
    to_point_id,
)
from lore_mcp.adapters.qdrant.initializer import QdrantInitializer
from lore_mcp.application.errors import DependencyUnavailable
from lore_mcp.config import QdrantSettings
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.identifiers import LORE_CHUNK_NAMESPACE, generate_chunk_id
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import SearchQuery
from lore_mcp.domain.types import ChunkId, EntityId, SeriesId


def make_chunk(
    chunk_id: str,
    series_id: str = "loki",
    text: str = "Test chunk text",
    season: int | None = 1,
    episode: int | None = 1,
    entity_ids: list[str] | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=ChunkId(chunk_id),
        series_id=SeriesId(series_id),
        text=text,
        season=season,
        episode=episode,
        scene_index=1,
        narrative_order=100,
        schema_version=1,
        ingest_version=1,
        content_hash="test-hash",
        entity_ids=[EntityId(e) for e in (entity_ids or ["loki"])],
        provenance=Provenance(
            source_url="https://example.com/source",
            source_type="transcript",
            license="CC-BY-SA-3.0",
            attribution_text="Test Attribution",
            retrieval_timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
            revision_id="rev-123",
        ),
    )


@pytest.fixture
def base_settings() -> QdrantSettings:
    return QdrantSettings(
        url=":memory:",
        chunks_collection="test_chunks",
        meta_collection="test_meta",
        embedding_model="all-MiniLM-L6-v2",
        dimension=384,
        inference_backend="fastembed",
        schema_version=1,
        batch_size=50,
    )


async def make_test_client(settings: QdrantSettings) -> AsyncQdrantClient:
    """In-memory Qdrant client initialized with schema and mock inference."""
    client = AsyncQdrantClient(":memory:")
    await QdrantInitializer.ensure_collections_ready(client, settings)

    # Patch local inference embedder to avoid ONNX downloads in unit tests
    def fake_embed_models(
        points_or_queries: Any, is_query: bool = False, batch_size: int = 8
    ) -> Any:
        if is_query:
            yield [0.0] * 384
            return
        for p in points_or_queries:
            yield models.PointStruct(
                id=p.id,
                vector={
                    "dense": [0.01] * 384,
                    "bm25": models.SparseVector(indices=[0, 1], values=[1.0, 0.5]),
                },
                payload=p.payload,
            )

    client._embed_models = fake_embed_models  # type: ignore[assignment]
    return client


@pytest.mark.unit
def test_deterministic_point_id_generation() -> None:
    # 1. Valid UUID string chunk IDs remain unchanged
    valid_uuid_str = str(uuid.uuid4())
    assert to_point_id(ChunkId(valid_uuid_str)) == valid_uuid_str

    # 2. Domain-generated chunk IDs (which are UUID5) remain unchanged
    generated_id = generate_chunk_id(
        SeriesId("loki"), "https://wiki/Loki", "History", 0
    )
    assert to_point_id(generated_id) == str(generated_id)

    # 3. Non-UUID strings (e.g. test fixture IDs) map to deterministic UUID5
    id1 = to_point_id(ChunkId("c1"))
    id2 = to_point_id(ChunkId("c1"))
    id_other = to_point_id(ChunkId("c2"))

    assert id1 == id2
    assert id1 != id_other
    assert uuid.UUID(id1).version == 5
    assert id1 == str(uuid.uuid5(LORE_CHUNK_NAMESPACE, "c1"))


@pytest.mark.unit
def test_resolve_embedding_model_name() -> None:
    assert (
        resolve_embedding_model_name("all-MiniLM-L6-v2", "fastembed")
        == "sentence-transformers/all-MiniLM-L6-v2"
    )
    model_full = "sentence-transformers/all-MiniLM-L6-v2"
    assert (
        resolve_embedding_model_name(model_full, "fastembed")
        == "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert (
        resolve_embedding_model_name("all-MiniLM-L6-v2", "cloud") == "all-MiniLM-L6-v2"
    )


@pytest.mark.unit
def test_build_point_structure(base_settings: QdrantSettings) -> None:
    dummy_client = AsyncMock(spec=AsyncQdrantClient)
    index = QdrantVectorIndex(dummy_client, base_settings)

    chunk = make_chunk("c1", "loki", "Content to embed")
    point = index._build_point(chunk)

    assert point.id == to_point_id(chunk.chunk_id)

    # Vector structures contain models.Document for dense and sparse models
    assert isinstance(point.vector, dict)
    dense_doc = point.vector["dense"]
    sparse_doc = point.vector["bm25"]
    assert isinstance(dense_doc, models.Document)
    assert dense_doc.text == "Content to embed"
    assert dense_doc.model == "sentence-transformers/all-MiniLM-L6-v2"

    assert isinstance(sparse_doc, models.Document)
    assert sparse_doc.text == "Content to embed"
    assert sparse_doc.model == "Qdrant/bm25"

    # Payload checks
    payload = point.payload or {}
    assert payload["series"] == "loki"
    assert payload["entity_id"] == ["loki"]
    assert payload["season"] == 1
    assert payload["episode"] == 1
    assert payload["scene_index"] == 1
    assert payload["narrative_order"] == 100
    assert payload["source_type"] == "transcript"
    assert payload["chunk_id"] == "c1"
    assert "_chunk_json" in payload


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_batching(base_settings: QdrantSettings) -> None:
    mock_client = AsyncMock(spec=AsyncQdrantClient)
    base_settings_with_batch = base_settings.model_copy(update={"batch_size": 50})
    index = QdrantVectorIndex(mock_client, base_settings_with_batch)

    # Upsert 125 chunks with batch_size=50 -> 3 batches (50, 50, 25)
    chunks = [make_chunk(f"chunk-{i}", "loki") for i in range(125)]
    await index.upsert(chunks)

    assert mock_client.upsert.call_count == 3
    first_call_points = mock_client.upsert.call_args_list[0].kwargs["points"]
    second_call_points = mock_client.upsert.call_args_list[1].kwargs["points"]
    third_call_points = mock_client.upsert.call_args_list[2].kwargs["points"]

    assert len(first_call_points) == 50
    assert len(second_call_points) == 50
    assert len(third_call_points) == 25


@pytest.mark.unit
@pytest.mark.asyncio
async def test_idempotent_indexing_and_resumability(
    base_settings: QdrantSettings,
) -> None:
    client = await make_test_client(base_settings)
    index = QdrantVectorIndex(client, base_settings)

    chunk1 = make_chunk("c1", "loki", "First version of Loki text")
    chunk2 = make_chunk("c2", "loki", "Second chunk text")

    # 1. Initial indexing
    await index.upsert([chunk1, chunk2])
    assert await index.count(SeriesId("loki")) == 2
    assert await index.count() == 2

    # 2. Resumability: check existing IDs
    existing = await index.get_existing_ids(
        [ChunkId("c1"), ChunkId("c2"), ChunkId("c3")]
    )
    assert existing == {ChunkId("c1"), ChunkId("c2")}

    # 3. Idempotent indexing: re-indexing identical chunks does not duplicate points
    await index.upsert([chunk1, chunk2])
    assert await index.count(SeriesId("loki")) == 2

    # 4. In-place update with same deterministic ID
    chunk1_updated = make_chunk("c1", "loki", "Updated Loki text")
    await index.upsert([chunk1_updated])
    assert await index.count(SeriesId("loki")) == 2

    fetched = await index.get_by_id(ChunkId("c1"))
    assert fetched is not None
    assert fetched.text == "Updated Loki text"
    assert fetched.chunk_id == "c1"
    assert fetched.provenance.license == "CC-BY-SA-3.0"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_by_id_missing_and_fallback(
    base_settings: QdrantSettings,
) -> None:
    client = await make_test_client(base_settings)
    index = QdrantVectorIndex(client, base_settings)

    # Missing chunk returns None
    assert await index.get_by_id(ChunkId("nonexistent")) is None

    # Test fallback payload reconstruction when _chunk_json is absent
    point_id = to_point_id(ChunkId("fallback-chunk"))
    await client.upsert(
        collection_name=base_settings.chunks_collection,
        points=[
            models.PointStruct(
                id=point_id,
                vector={
                    "dense": [0.01] * 384,
                    "bm25": models.SparseVector(indices=[0], values=[1.0]),
                },
                payload={
                    "series": "loki",
                    "chunk_id": "fallback-chunk",
                    "text": "Fallback text",
                    "season": 2,
                    "episode": 3,
                    "source_url": "https://example.com/fallback",
                    "source_type": "fandom",
                    "entity_id": ["mobius"],
                },
            )
        ],
    )

    reconstructed = await index.get_by_id(ChunkId("fallback-chunk"))
    assert reconstructed is not None
    assert reconstructed.chunk_id == "fallback-chunk"
    assert reconstructed.series_id == "loki"
    assert reconstructed.text == "Fallback text"
    assert reconstructed.season == 2
    assert reconstructed.episode == 3
    assert reconstructed.entity_ids == [EntityId("mobius")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_series_isolation(
    base_settings: QdrantSettings,
) -> None:
    client = await make_test_client(base_settings)
    index = QdrantVectorIndex(client, base_settings)

    c_loki = make_chunk("c-loki", "loki", "Loki text")
    c_sw = make_chunk("c-sw", "star-wars", "Star Wars text")

    await index.upsert([c_loki, c_sw])
    assert await index.count() == 2
    assert await index.count(SeriesId("loki")) == 1
    assert await index.count(SeriesId("star-wars")) == 1

    await index.delete_series(SeriesId("loki"))

    assert await index.count(SeriesId("loki")) == 0
    assert await index.count(SeriesId("star-wars")) == 1
    assert await index.count() == 1
    assert await index.get_by_id(ChunkId("c-loki")) is None
    assert await index.get_by_id(ChunkId("c-sw")) is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_filters_series(
    base_settings: QdrantSettings,
) -> None:
    client = await make_test_client(base_settings)
    index = QdrantVectorIndex(client, base_settings)

    c1 = make_chunk("c1", "loki", "Loki in TVA")
    c2 = make_chunk("c2", "star-wars", "Loki in a galaxy far away")
    await index.upsert([c1, c2])

    query = SearchQuery(series_id=SeriesId("loki"), query="Loki")
    hits = await index.search(query)

    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == "c1"
    assert hits[0].chunk.series_id == "loki"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_operation_error_wrapping(base_settings: QdrantSettings) -> None:
    mock_client = AsyncMock(spec=AsyncQdrantClient)
    mock_client.upsert.side_effect = RuntimeError("Qdrant connection dropped")
    mock_client.retrieve.side_effect = RuntimeError("Qdrant read timeout")
    mock_client.delete.side_effect = RuntimeError("Qdrant write timeout")
    mock_client.count.side_effect = RuntimeError("Qdrant count failure")
    mock_client.query_points.side_effect = RuntimeError("Qdrant search error")

    index = QdrantVectorIndex(mock_client, base_settings)
    chunk = make_chunk("c1", "loki")

    with pytest.raises(QdrantOperationError) as exc_upsert:
        await index.upsert([chunk])
    assert isinstance(exc_upsert.value, DependencyUnavailable)

    with pytest.raises(QdrantOperationError) as exc_get:
        await index.get_by_id(ChunkId("c1"))
    assert isinstance(exc_get.value, DependencyUnavailable)

    with pytest.raises(QdrantOperationError) as exc_del:
        await index.delete_series(SeriesId("loki"))
    assert isinstance(exc_del.value, DependencyUnavailable)

    with pytest.raises(QdrantOperationError) as exc_count:
        await index.count(SeriesId("loki"))
    assert isinstance(exc_count.value, DependencyUnavailable)

    with pytest.raises(QdrantOperationError) as exc_search:
        await index.search(SearchQuery(series_id=SeriesId("loki"), query="test"))
    assert isinstance(exc_search.value, DependencyUnavailable)
