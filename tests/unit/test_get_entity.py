"""Unit tests for get_entity MCP tool and EntityService."""

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from qdrant_client import AsyncQdrantClient
from starlette.testclient import TestClient

from lore_mcp.application.entity_service import EntityService
from lore_mcp.application.errors import DataUnavailable, InvalidArgument, InvalidScope
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.application.scope import ScopeResolver
from lore_mcp.config import QdrantSettings, RequestLimitsSettings, Settings
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.entities import Entity, EntityDetails, SeriesManifest
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.types import ChunkId, EntityId, SeriesId
from lore_mcp.interfaces.mcp import create_http_app, create_mcp_server
from lore_mcp.testing import FakeEntityRegistry, FakeSeriesRegistry, FakeVectorIndex


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        qdrant=QdrantSettings(
            url=":memory:",
            chunks_collection="test_chunks",
            meta_collection="test_meta",
            embedding_model="all-MiniLM-L6-v2",
            dimension=384,
            inference_backend="cloud",
            schema_version=1,
        ),
        limits=RequestLimitsSettings(
            default_max_results=8,
            hard_max_results=20,
            max_query_length=500,
        ),
    )


def make_chunk(
    chunk_id: str,
    series_id: str,
    text: str,
    season: int | None = 1,
    episode: int | None = 1,
    scene_index: int | None = 1,
    narrative_order: int | None = 1,
    entity_ids: list[str] | None = None,
    source_url: str = "https://example.com/source1",
) -> Chunk:
    return Chunk(
        chunk_id=ChunkId(chunk_id),
        series_id=SeriesId(series_id),
        text=text,
        season=season,
        episode=episode,
        scene_index=scene_index,
        narrative_order=narrative_order,
        content_hash=f"hash_{chunk_id}",
        entity_ids=[EntityId(e) for e in (entity_ids or [])],
        provenance=Provenance(
            source_type="transcript",
            source_url=source_url,
            retrieval_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )


@pytest.fixture
def sample_entities() -> list[Entity]:
    return [
        Entity(
            entity_id=EntityId("loki"),
            series_id=SeriesId("loki"),
            name="Loki Laufeyson",
            entity_type="character",
            aliases=["God of Mischief", "Variant L1130"],
            summary="Prince of Asgard and God of Mischief.",
            related_entities=[EntityId("mobius"), EntityId("sylvie")],
            metadata={"status": "alive", "role": "protagonist"},
        ),
        Entity(
            entity_id=EntityId("he-who-remains"),
            series_id=SeriesId("loki"),
            name="He Who Remains",
            entity_type="character",
            aliases=["Kang Variant", "Creator of TVA"],
            summary="Creator of the Time Variance Authority.",
            related_entities=[EntityId("tva")],
            metadata={"status": "deceased"},
        ),
        Entity(
            entity_id=EntityId("tva"),
            series_id=SeriesId("loki"),
            name="Time Variance Authority",
            entity_type="organization",
            aliases=["TVA"],
            summary="Bureaucratic organization preserving the Sacred Timeline.",
            related_entities=[EntityId("he-who-remains")],
        ),
    ]


@pytest.fixture
def populated_index() -> FakeVectorIndex:
    index = FakeVectorIndex()
    chunks = [
        make_chunk(
            "loki-1",
            "loki",
            "Loki: I am Loki of Asgard, and I am burdened with glorious purpose.",
            season=1,
            episode=1,
            entity_ids=["loki"],
            source_url="https://example.com/s1e1",
        ),
        make_chunk(
            "loki-2",
            "loki",
            "Mobius interviews Loki about his true motivations at the TVA.",
            season=1,
            episode=1,
            entity_ids=["loki", "mobius", "tva"],
            source_url="https://example.com/s1e1",
        ),
        make_chunk(
            "loki-3",
            "loki",
            "He Who Remains explains the multiversal war to Loki and Sylvie.",
            season=1,
            episode=6,
            entity_ids=["he-who-remains", "loki", "sylvie"],
            source_url="https://example.com/s1e6",
        ),
        make_chunk(
            "unregistered-entity-chunk",
            "loki",
            "Miss Minutes explains the Sacred Timeline to new TVA variants.",
            season=1,
            episode=1,
            entity_ids=["miss-minutes"],
            source_url="https://example.com/s1e1-cartoon",
        ),
    ]
    for c in chunks:
        index._chunks[c.chunk_id] = c
    return index


@pytest.fixture
def populated_registry() -> FakeSeriesRegistry:
    reg = FakeSeriesRegistry()
    manifest = SeriesManifest(
        series_id=SeriesId("loki"),
        display_name="Loki",
        kind="show",
        aliases=["MCU Loki"],
        seasons=[1, 2],
        entity_count=3,
        chunk_count=4,
    )
    reg._manifests[SeriesId("loki")] = manifest
    return reg


@pytest.fixture
def populated_entity_registry(sample_entities: list[Entity]) -> FakeEntityRegistry:
    return FakeEntityRegistry(sample_entities)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_entity_by_canonical_id(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
    populated_entity_registry: FakeEntityRegistry,
) -> None:
    """Resolve canonical entity by slug ID and verify evidence + provenance."""
    scope_resolver = ScopeResolver(registry=populated_registry)

    retriever = HybridRetriever(index=populated_index)
    service = EntityService(
        scope_resolver=scope_resolver,
        retriever=retriever,
        entity_registry=populated_entity_registry,
    )

    details = await service.get_entity(entity="loki", series="loki")
    assert isinstance(details, EntityDetails)
    assert details.entity.entity_id == "loki"
    assert details.entity.name == "Loki Laufeyson"
    assert "God of Mischief" in details.entity.aliases
    assert len(details.evidence) > 0

    # Ensure evidence is associated with the entity
    for ev in details.evidence:
        assert ev.entity_id == "loki"
        assert ev.series_id == "loki"
        assert ev.provenance.source_url

    # Check unique deduplicated provenance
    urls = [p.source_url for p in details.provenance]
    assert len(urls) == len(set(urls))
    assert "https://example.com/s1e1" in urls


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_entity_by_alias(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
    populated_entity_registry: FakeEntityRegistry,
) -> None:
    """Resolve entity via known alias (e.g. 'God of Mischief' or 'Kang Variant')."""
    scope_resolver = ScopeResolver(registry=populated_registry)
    retriever = HybridRetriever(index=populated_index)
    service = EntityService(
        scope_resolver=scope_resolver,
        retriever=retriever,
        entity_registry=populated_entity_registry,
    )

    # By alias "God of Mischief"
    details_loki = await service.get_entity(entity="God of Mischief", series="loki")
    assert details_loki.entity.entity_id == "loki"
    assert details_loki.entity.name == "Loki Laufeyson"

    # By alias "Kang Variant"
    details_kang = await service.get_entity(entity="Kang Variant", series="loki")
    assert details_kang.entity.entity_id == "he-who-remains"
    assert details_kang.entity.name == "He Who Remains"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_entity_by_display_name_case_insensitive(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
    populated_entity_registry: FakeEntityRegistry,
) -> None:
    """Resolve entity by display name case-insensitively."""
    scope_resolver = ScopeResolver(registry=populated_registry)
    retriever = HybridRetriever(index=populated_index)
    service = EntityService(
        scope_resolver=scope_resolver,
        retriever=retriever,
        entity_registry=populated_entity_registry,
    )

    details = await service.get_entity(entity="he who remains", series="loki")
    assert details.entity.entity_id == "he-who-remains"
    assert details.entity.name == "He Who Remains"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_entity_fallback_synthesis_from_chunks(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
    populated_entity_registry: FakeEntityRegistry,
) -> None:
    """Synthesize canonical entity from indexed chunks when unregistered."""
    scope_resolver = ScopeResolver(registry=populated_registry)

    retriever = HybridRetriever(index=populated_index)
    service = EntityService(
        scope_resolver=scope_resolver,
        retriever=retriever,
        entity_registry=populated_entity_registry,
    )

    # 'miss-minutes' is in chunks but not in entity registry
    details = await service.get_entity(entity="miss-minutes", series="loki")
    assert details.entity.entity_id == "miss-minutes"
    assert len(details.evidence) >= 1
    assert details.evidence[0].chunk_id == "unregistered-entity-chunk"
    assert len(details.provenance) == 1
    assert details.provenance[0].source_url == "https://example.com/s1e1-cartoon"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_entity_unknown_entity_raises_data_unavailable(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
    populated_entity_registry: FakeEntityRegistry,
) -> None:
    """When entity does not exist in registry or chunks, raise DataUnavailable."""
    scope_resolver = ScopeResolver(registry=populated_registry)
    retriever = HybridRetriever(index=populated_index)
    service = EntityService(
        scope_resolver=scope_resolver,
        retriever=retriever,
        entity_registry=populated_entity_registry,
    )

    with pytest.raises(DataUnavailable) as exc_info:
        await service.get_entity(entity="nonexistent-character", series="loki")
    assert "nonexistent-character" in str(exc_info.value)
    assert exc_info.value.resource_id == "nonexistent-character"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_entity_unknown_series_raises_invalid_scope(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
    populated_entity_registry: FakeEntityRegistry,
) -> None:
    """When series is unknown, ScopeResolver raises InvalidScope."""
    scope_resolver = ScopeResolver(registry=populated_registry)
    retriever = HybridRetriever(index=populated_index)
    service = EntityService(
        scope_resolver=scope_resolver,
        retriever=retriever,
        entity_registry=populated_entity_registry,
    )

    with pytest.raises(InvalidScope) as exc_info:
        await service.get_entity(entity="loki", series="unknown-franchise")
    assert "unknown-franchise" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_entity_input_validation(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
    populated_entity_registry: FakeEntityRegistry,
) -> None:
    """Invalid empty entity or invalid max_results raises InvalidArgument."""
    scope_resolver = ScopeResolver(registry=populated_registry)
    retriever = HybridRetriever(index=populated_index)
    service = EntityService(
        scope_resolver=scope_resolver,
        retriever=retriever,
        entity_registry=populated_entity_registry,
    )

    with pytest.raises(InvalidArgument) as exc1:
        await service.get_entity(entity="   ", series="loki")
    assert exc1.value.argument_name == "entity"

    with pytest.raises(InvalidArgument) as exc2:
        await service.get_entity(entity="loki", series="loki", max_results=0)
    assert exc2.value.argument_name == "max_results"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_entity_max_results_limit(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
    populated_entity_registry: FakeEntityRegistry,
) -> None:
    """max_results parameter limits the number of returned evidence items."""
    scope_resolver = ScopeResolver(registry=populated_registry)
    retriever = HybridRetriever(index=populated_index)
    service = EntityService(
        scope_resolver=scope_resolver,
        retriever=retriever,
        entity_registry=populated_entity_registry,
    )

    details = await service.get_entity(entity="loki", series="loki", max_results=1)
    assert len(details.evidence) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_entity_registered_with_no_chunks(
    populated_registry: FakeSeriesRegistry,
) -> None:
    """Registered entity with no chunks returns metadata with empty evidence."""
    empty_index = FakeVectorIndex()

    entity_reg = FakeEntityRegistry(
        [
            Entity(
                entity_id=EntityId("ravonna"),
                series_id=SeriesId("loki"),
                name="Ravonna Renslayer",
                entity_type="character",
                aliases=["Judge Renslayer"],
                summary="Judge of the Time Variance Authority.",
            )
        ]
    )
    scope_resolver = ScopeResolver(registry=populated_registry)
    retriever = HybridRetriever(index=empty_index)
    service = EntityService(
        scope_resolver=scope_resolver,
        retriever=retriever,
        entity_registry=entity_reg,
    )

    details = await service.get_entity(entity="ravonna", series="loki")
    assert details.entity.entity_id == "ravonna"
    assert details.entity.name == "Ravonna Renslayer"
    assert details.evidence == []
    assert details.provenance == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_entity_via_fastmcp_call_tool(
    test_settings: Settings,
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
    populated_entity_registry: FakeEntityRegistry,
) -> None:
    """get_entity tool is invocable directly via FastMCP call_tool."""
    memory_client = AsyncQdrantClient(":memory:")
    mcp: FastMCP = create_mcp_server(
        settings=test_settings,
        client=memory_client,
        registry=populated_registry,
        entity_registry=populated_entity_registry,
        vector_index=populated_index,
    )

    result = await mcp.call_tool(
        "get_entity",
        {"entity": "God of Mischief", "series": "loki"},
    )
    assert result.is_error is False
    assert result.structured_content is not None

    details = EntityDetails.model_validate(result.structured_content)
    assert details.entity.entity_id == "loki"
    assert details.entity.name == "Loki Laufeyson"
    assert len(details.evidence) > 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_entity_via_fastmcp_error_handling(
    test_settings: Settings,
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
    populated_entity_registry: FakeEntityRegistry,
) -> None:
    """FastMCP call_tool wraps application errors appropriately."""
    memory_client = AsyncQdrantClient(":memory:")
    mcp: FastMCP = create_mcp_server(
        settings=test_settings,
        client=memory_client,
        registry=populated_registry,
        entity_registry=populated_entity_registry,
        vector_index=populated_index,
    )

    with pytest.raises((DataUnavailable, ToolError)) as exc_info:
        await mcp.call_tool(
            "get_entity",
            {"entity": "nonexistent-entity", "series": "loki"},
        )
    assert "nonexistent-entity" in str(exc_info.value)


@pytest.mark.unit
def test_get_entity_via_streamable_http(
    test_settings: Settings,
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
    populated_entity_registry: FakeEntityRegistry,
) -> None:
    """get_entity is callable over Streamable HTTP POST /mcp."""
    memory_client = AsyncQdrantClient(":memory:")
    mcp = create_mcp_server(
        settings=test_settings,
        client=memory_client,
        registry=populated_registry,
        entity_registry=populated_entity_registry,
        vector_index=populated_index,
    )
    app = create_http_app(mcp)

    with TestClient(app) as client:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "get_entity",
                "arguments": {
                    "entity": "loki",
                    "series": "loki",
                },
            },
        }
        res = client.post(
            "/mcp",
            json=payload,
            headers={"Accept": "application/json, text/event-stream"},
        )
        assert res.status_code == 200
        content_type = res.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            lines = res.text.strip().splitlines()
            data_line = next(line for line in lines if line.startswith("data: "))
            data = json.loads(data_line.removeprefix("data: ").strip())
        else:
            data = res.json()

        assert data.get("jsonrpc") == "2.0"
        assert "result" in data
        assert "content" in data["result"]
        parsed_entity_res = json.loads(data["result"]["content"][0]["text"])
        assert parsed_entity_res["entity"]["entity_id"] == "loki"
        assert parsed_entity_res["entity"]["name"] == "Loki Laufeyson"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_entity_missing_in_registry_and_chunks_raises_data_unavailable(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
    populated_entity_registry: FakeEntityRegistry,
) -> None:
    """When entity is absent in registry and has 0 chunk matches, DataUnavailable
    is raised.
    """
    scope_resolver = ScopeResolver(registry=populated_registry)
    retriever = HybridRetriever(index=populated_index)
    service = EntityService(
        scope_resolver=scope_resolver,
        retriever=retriever,
        entity_registry=populated_entity_registry,
    )

    with pytest.raises(DataUnavailable) as exc_info:
        await service.get_entity(entity="thor", series="loki")
    assert "thor" in str(exc_info.value)
    assert exc_info.value.resource_id == "thor"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_entity_retrieval_operational_error_falls_back_to_data_unavailable(
    populated_registry: FakeSeriesRegistry,
    populated_entity_registry: FakeEntityRegistry,
) -> None:
    """Non-dependency error during fallback chunk search results in DataUnavailable."""
    from unittest.mock import AsyncMock

    from lore_mcp.ports.vector_index import VectorIndex

    mock_index = AsyncMock(spec=VectorIndex)
    mock_index.search.side_effect = RuntimeError("Vector embedding inference error")

    scope_resolver = ScopeResolver(registry=populated_registry)
    retriever = HybridRetriever(index=mock_index)
    service = EntityService(
        scope_resolver=scope_resolver,
        retriever=retriever,
        entity_registry=populated_entity_registry,
    )

    with pytest.raises(DataUnavailable) as exc_info:
        await service.get_entity(entity="thor", series="loki")
    assert "thor" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_entity_connection_error_propagates_dependency_unavailable(
    populated_registry: FakeSeriesRegistry,
    populated_entity_registry: FakeEntityRegistry,
) -> None:
    """Genuine connectivity failure propagates DependencyUnavailable."""
    from unittest.mock import AsyncMock

    from lore_mcp.application.errors import DependencyUnavailable
    from lore_mcp.ports.vector_index import VectorIndex

    mock_index = AsyncMock(spec=VectorIndex)
    mock_index.search_dense.side_effect = DependencyUnavailable(
        "qdrant", "connection refused"
    )
    mock_index.search_sparse.side_effect = DependencyUnavailable(
        "qdrant", "connection refused"
    )

    scope_resolver = ScopeResolver(registry=populated_registry)
    retriever = HybridRetriever(index=mock_index)
    service = EntityService(
        scope_resolver=scope_resolver,
        retriever=retriever,
        entity_registry=populated_entity_registry,
    )

    with pytest.raises(DependencyUnavailable):
        await service.get_entity(entity="thor", series="loki")
