"""Integration tests for Qdrant vector storage, registry, filtering,
versioned rebuild, and pruning [S4-S11].
"""

import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from qdrant_client import AsyncQdrantClient, models

from lore_mcp.adapters.qdrant.index import QdrantVectorIndex
from lore_mcp.adapters.qdrant.initializer import QdrantInitializer
from lore_mcp.adapters.qdrant.registry import QdrantSeriesRegistry
from lore_mcp.application.errors import IncompatibleData
from lore_mcp.config import QdrantSettings
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.entities import SeriesManifest
from lore_mcp.domain.identifiers import generate_chunk_id
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import SearchQuery
from lore_mcp.domain.types import ChunkId, EntityId, SeriesId


def is_docker_qdrant_running(url: str = "http://localhost:6333") -> bool:
    """Check if an external/Docker Qdrant instance is reachable."""
    try:
        resp = httpx.get(f"{url}/readyz", timeout=0.5)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture
def test_prefix() -> str:
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def qdrant_settings(test_prefix: str) -> QdrantSettings:
    use_docker = is_docker_qdrant_running()
    return QdrantSettings(
        url="http://localhost:6333" if use_docker else ":memory:",
        chunks_collection=f"{test_prefix}_chunks",
        meta_collection=f"{test_prefix}_meta",
        embedding_model="all-MiniLM-L6-v2",
        dimension=384,
        inference_backend="fastembed",
        schema_version=1,
        batch_size=50,
    )


async def create_initialized_client(settings: QdrantSettings) -> AsyncQdrantClient:
    """Create client and initialize schema for tests."""
    client = AsyncQdrantClient(settings.url)
    await QdrantInitializer.ensure_collections_ready(client, settings)

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


def make_chunk(
    chunk_id: str,
    series_id: str = "loki",
    season: int | None = 1,
    episode: int | None = 1,
    ingest_version: int = 1,
    entity_ids: list[str] | None = None,
    source_type: str = "transcript",
    text: str = "Sample lore content",
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
        ingest_version=ingest_version,
        content_hash=f"hash-{chunk_id}-v{ingest_version}",
        entity_ids=[EntityId(e) for e in (entity_ids or ["loki"])],
        provenance=Provenance(
            source_url="https://example.com/source",
            source_type=source_type,
            license="CC-BY-SA-3.0",
            attribution_text="MCU Wiki",
            retrieval_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            revision_id="rev-1",
        ),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_collection_config_dimensions_and_metadata_mismatch(
    qdrant_settings: QdrantSettings,
) -> None:
    client = await create_initialized_client(qdrant_settings)

    # 1. Verify collections and vectors configuration
    assert await client.collection_exists(qdrant_settings.chunks_collection)
    assert await client.collection_exists(qdrant_settings.meta_collection)

    info = await client.get_collection(qdrant_settings.chunks_collection)
    assert info.config is not None
    vectors = info.config.params.vectors
    assert isinstance(vectors, dict)
    assert vectors["dense"].size == 384
    assert vectors["dense"].distance == models.Distance.COSINE
    assert info.config.metadata == {
        "embedding_model": "all-MiniLM-L6-v2",
        "dimension": 384,
        "inference_backend": "fastembed",
        "schema_version": 1,
    }

    # 2. Refusal to start on metadata mismatch
    mismatched_settings = qdrant_settings.model_copy(
        update={"embedding_model": "different-model"}
    )
    with pytest.raises(IncompatibleData):
        await QdrantInitializer.validate_chunks_metadata(client, mismatched_settings)

    dim_mismatch_settings = qdrant_settings.model_copy(update={"dimension": 768})
    with pytest.raises(IncompatibleData):
        await QdrantInitializer.validate_chunks_metadata(client, dim_mismatch_settings)

    backend_mismatch_settings = qdrant_settings.model_copy(
        update={"inference_backend": "cloud"}
    )
    with pytest.raises(IncompatibleData):
        await QdrantInitializer.validate_chunks_metadata(
            client, backend_mismatch_settings
        )

    schema_mismatch_settings = qdrant_settings.model_copy(update={"schema_version": 2})
    with pytest.raises(IncompatibleData):
        await QdrantInitializer.validate_chunks_metadata(
            client, schema_mismatch_settings
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chunk_indexing_payload_preservation_and_deterministic_ids(
    qdrant_settings: QdrantSettings,
) -> None:
    client = await create_initialized_client(qdrant_settings)
    index = QdrantVectorIndex(client, qdrant_settings)

    deterministic_id = generate_chunk_id(
        SeriesId("loki"), "https://wiki/Loki", "Origin", 0
    )
    chunk = make_chunk(
        chunk_id=str(deterministic_id),
        series_id="loki",
        season=2,
        episode=6,
        ingest_version=1,
        entity_ids=["loki", "temporal-loom"],
        source_type="transcript",
        text="Loki weaves the timelines into Yggdrasil.",
    )

    # 1. Chunk indexing
    await index.upsert([chunk])
    assert await index.count(SeriesId("loki")) == 1

    # 2. Payload preservation
    fetched = await index.get_by_id(deterministic_id)
    assert fetched is not None
    assert fetched.chunk_id == deterministic_id
    assert fetched.series_id == "loki"
    assert fetched.season == 2
    assert fetched.episode == 6
    assert fetched.ingest_version == 1
    assert fetched.entity_ids == [EntityId("loki"), EntityId("temporal-loom")]
    assert fetched.provenance.license == "CC-BY-SA-3.0"
    assert fetched.provenance.attribution_text == "MCU Wiki"
    assert fetched.text == "Loki weaves the timelines into Yggdrasil."

    # 3. Idempotent indexing: second upsert leaves point count unchanged
    await index.upsert([chunk])
    assert await index.count(SeriesId("loki")) == 1

    # Content update in place with identical ID
    updated_chunk = make_chunk(
        chunk_id=str(deterministic_id),
        series_id="loki",
        season=2,
        episode=6,
        ingest_version=1,
        text="Loki takes the throne at the center of the multiverse.",
    )
    await index.upsert([updated_chunk])
    assert await index.count(SeriesId("loki")) == 1
    fetched_updated = await index.get_by_id(deterministic_id)
    assert fetched_updated is not None
    assert fetched_updated.text == (
        "Loki takes the throne at the center of the multiverse."
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metadata_filters_and_spoiler_bounds(
    qdrant_settings: QdrantSettings,
) -> None:
    client = await create_initialized_client(qdrant_settings)
    index = QdrantVectorIndex(client, qdrant_settings)

    # Ingest diverse chunks
    c_s1e6 = make_chunk("c1", "loki", season=1, episode=6, entity_ids=["loki"])
    c_s2e2 = make_chunk(
        "c2", "loki", season=2, episode=2, entity_ids=["loki", "mobius"]
    )
    c_s2e4 = make_chunk(
        "c3", "loki", season=2, episode=4, entity_ids=["loki", "victor-timely"]
    )
    c_s3e1 = make_chunk("c4", "loki", season=3, episode=1, entity_ids=["loki"])
    c_unscoped = make_chunk("c5", "loki", season=None, episode=None, entity_ids=["tva"])
    c_sw = make_chunk("c6", "star-wars", season=1, episode=1, entity_ids=["loki"])

    await index.upsert([c_s1e6, c_s2e2, c_s2e4, c_s3e1, c_unscoped, c_sw])

    # 1. Mandatory series isolation: never leaks star-wars
    query_base = SearchQuery(series_id=SeriesId("loki"), query="loki")
    hits_base = await index.search(query_base)
    assert all(h.chunk.series_id == "loki" for h in hits_base)
    assert len(hits_base) == 5

    # 2. Exact equality: season=2, episode=2
    query_exact = SearchQuery(
        series_id=SeriesId("loki"), query="loki", season=2, episode=2
    )
    hits_exact = await index.search(query_exact)
    assert [h.chunk.chunk_id for h in hits_exact] == [ChunkId("c2")]

    # 3. Entity filter: "victor-timely"
    query_entity = SearchQuery(
        series_id=SeriesId("loki"), query="loki", entity="victor-timely"
    )
    hits_entity = await index.search(query_entity)
    assert [h.chunk.chunk_id for h in hits_entity] == [ChunkId("c3")]

    # 4. Spoiler bound: max_season=2, max_episode=3 (without unscoped)
    # Expected: c1 (S1E6) and c2 (S2E2)
    query_spoiler_no_unscoped = SearchQuery(
        series_id=SeriesId("loki"),
        query="loki",
        max_season=2,
        max_episode=3,
        include_unscoped=False,
    )
    hits_spoiler = await index.search(query_spoiler_no_unscoped)
    assert {h.chunk.chunk_id for h in hits_spoiler} == {ChunkId("c1"), ChunkId("c2")}

    # 5. Spoiler bound: max_season=2, max_episode=3 (with unscoped)
    # Expected: c1, c2, and c5 (unscoped TVA)
    query_spoiler_with_unscoped = SearchQuery(
        series_id=SeriesId("loki"),
        query="loki",
        max_season=2,
        max_episode=3,
        include_unscoped=True,
    )
    hits_with_unscoped = await index.search(query_spoiler_with_unscoped)
    assert {h.chunk.chunk_id for h in hits_with_unscoped} == {
        ChunkId("c1"),
        ChunkId("c2"),
        ChunkId("c5"),
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_versioned_rebuild_prune_and_interrupted_run_safety(
    qdrant_settings: QdrantSettings,
) -> None:
    client = await create_initialized_client(qdrant_settings)
    index = QdrantVectorIndex(client, qdrant_settings)
    registry = QdrantSeriesRegistry(client, qdrant_settings)

    # Run 1: Initial Ingestion with ingest_version = 1
    chunks_v1 = [
        make_chunk("c1", "loki", ingest_version=1, text="Loki v1 - chunk 1"),
        make_chunk("c2", "loki", ingest_version=1, text="Loki v1 - chunk 2"),
        make_chunk("c3", "loki", ingest_version=1, text="Loki v1 - chunk 3"),
    ]
    await index.upsert(chunks_v1)

    manifest_v1 = SeriesManifest(
        series_id=SeriesId("loki"),
        display_name="Loki",
        chunk_count=3,
        ingest_version=1,
    )
    await registry.register_series(manifest_v1)

    assert await index.count(SeriesId("loki")) == 3
    fetched_manifest = await registry.get_series(SeriesId("loki"))
    assert fetched_manifest is not None
    assert fetched_manifest.ingest_version == 1

    # Also ingest independent series "andor" to ensure independent rebuild
    chunks_andor = [
        make_chunk("andor-1", "andor", ingest_version=1, text="Andor chunk 1"),
    ]
    await index.upsert(chunks_andor)
    assert await index.count(SeriesId("andor")) == 1

    # Run 2: Interrupted Rebuild with ingest_version = 2
    # Ingestion starts: batch 1 uploaded (c1, c2 with v2)
    partial_chunks_v2 = [
        make_chunk("c1", "loki", ingest_version=2, text="Loki v2 - chunk 1"),
        make_chunk("c2", "loki", ingest_version=2, text="Loki v2 - chunk 2"),
    ]
    await index.upsert(partial_chunks_v2)

    # Interrupted before manifest publish or prune:
    # Manifest is still v1 (interrupted-run safety: previous manifest remains active)
    active_manifest = await registry.get_series(SeriesId("loki"))
    assert active_manifest is not None
    assert active_manifest.ingest_version == 1

    # c3 (still v1) is intact and untouched
    c3_record = await index.get_by_id(ChunkId("c3"))
    assert c3_record is not None
    assert c3_record.ingest_version == 1

    # Resume Ingestion: upload remainder (c3 with v2 and a new chunk c4 with v2)
    remaining_chunks_v2 = [
        make_chunk("c3", "loki", ingest_version=2, text="Loki v2 - chunk 3"),
        make_chunk("c4", "loki", ingest_version=2, text="Loki v2 - chunk 4"),
    ]
    await index.upsert(remaining_chunks_v2)

    # Publish updated manifest
    manifest_v2 = SeriesManifest(
        series_id=SeriesId("loki"),
        display_name="Loki",
        chunk_count=4,
        ingest_version=2,
    )
    await registry.register_series(manifest_v2)
    updated_manifest = await registry.get_series(SeriesId("loki"))
    assert updated_manifest is not None
    assert updated_manifest.ingest_version == 2

    # Prune points with older ingest_version (< 2)
    await index.prune_older_versions(SeriesId("loki"), active_version=2)

    # Verify all loki chunks now have ingest_version = 2
    assert await index.count(SeriesId("loki")) == 4
    for cid in ("c1", "c2", "c3", "c4"):
        c = await index.get_by_id(ChunkId(cid))
        assert c is not None
        assert c.ingest_version == 2

    # Verify independent series "andor" was NOT affected by pruning
    assert await index.count(SeriesId("andor")) == 1
    andor_c = await index.get_by_id(ChunkId("andor-1"))
    assert andor_c is not None
    assert andor_c.ingest_version == 1
