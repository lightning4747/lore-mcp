"""Unit tests for get_chunk MCP tool and ChunkService [S11]."""

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from qdrant_client import AsyncQdrantClient
from starlette.testclient import TestClient

from lore_mcp.application.chunk_service import ChunkService
from lore_mcp.application.errors import DataUnavailable, InvalidArgument, InvalidScope
from lore_mcp.application.scope import ScopeResolver
from lore_mcp.config import QdrantSettings, RequestLimitsSettings, Settings
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.entities import SeriesManifest
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.types import ChunkId, EntityId, SeriesId
from lore_mcp.interfaces.mcp import create_http_app, create_mcp_server
from lore_mcp.testing import FakeSeriesRegistry, FakeVectorIndex


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
def populated_index() -> FakeVectorIndex:
    index = FakeVectorIndex()
    chunks = [
        make_chunk(
            "chunk-loki-1",
            "loki",
            "Loki: I am Loki of Asgard, and I am burdened with glorious purpose.",
            season=1,
            episode=1,
            scene_index=1,
            narrative_order=10,
            entity_ids=["loki"],
            source_url="https://example.com/loki/s1e1",
        ),
        make_chunk(
            "chunk-loki-2",
            "loki",
            "Mobius: Time passes differently here in the TVA.",
            season=1,
            episode=1,
            scene_index=2,
            narrative_order=20,
            entity_ids=["mobius", "tva"],
            source_url="https://example.com/loki/s1e1",
        ),
        make_chunk(
            "chunk-andor-1",
            "andor",
            "Cassian: To steal from the Empire, you need an inside man.",
            season=1,
            episode=1,
            scene_index=1,
            narrative_order=1,
            entity_ids=["cassian"],
            source_url="https://example.com/andor/s1e1",
        ),
    ]
    for c in chunks:
        index._chunks[c.chunk_id] = c
    return index


@pytest.fixture
def populated_registry() -> FakeSeriesRegistry:
    reg = FakeSeriesRegistry()
    m_loki = SeriesManifest(
        series_id=SeriesId("loki"),
        display_name="Loki",
        kind="show",
        aliases=["MCU Loki"],
        seasons=[1, 2],
        entity_count=3,
        chunk_count=2,
    )
    m_andor = SeriesManifest(
        series_id=SeriesId("andor"),
        display_name="Andor",
        kind="show",
        aliases=["Star Wars: Andor"],
        seasons=[1],
        entity_count=1,
        chunk_count=1,
    )
    reg._manifests[SeriesId("loki")] = m_loki
    reg._manifests[SeriesId("andor")] = m_andor
    return reg


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_chunk_success(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """Fetch a full chunk by ID and verify text expansion and attributes."""
    scope_resolver = ScopeResolver(registry=populated_registry)
    service = ChunkService(
        scope_resolver=scope_resolver,
        vector_index=populated_index,
    )

    chunk = await service.get_chunk(chunk_id="chunk-loki-1", series="loki")
    assert isinstance(chunk, Chunk)
    assert chunk.chunk_id == "chunk-loki-1"
    assert chunk.series_id == "loki"
    assert "burdened with glorious purpose" in chunk.text
    assert chunk.season == 1
    assert chunk.episode == 1
    assert chunk.scene_index == 1
    assert chunk.narrative_order == 10
    assert chunk.provenance.source_url == "https://example.com/loki/s1e1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_chunk_series_alias_resolution(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """Resolve series scope using alias when fetching chunk."""
    scope_resolver = ScopeResolver(registry=populated_registry)
    service = ChunkService(
        scope_resolver=scope_resolver,
        vector_index=populated_index,
    )

    chunk = await service.get_chunk(chunk_id="chunk-loki-1", series="MCU Loki")
    assert chunk.chunk_id == "chunk-loki-1"
    assert chunk.series_id == "loki"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_chunk_rechecks_series_scope_mismatch(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """Re-check series after fetching; reject cross-tenant chunk access [S11]."""
    scope_resolver = ScopeResolver(registry=populated_registry)
    service = ChunkService(
        scope_resolver=scope_resolver,
        vector_index=populated_index,
    )

    # 'chunk-loki-1' belongs to 'loki', but caller requested series 'andor'
    with pytest.raises(DataUnavailable) as exc_info:
        await service.get_chunk(chunk_id="chunk-loki-1", series="andor")

    assert "chunk-loki-1" in str(exc_info.value)
    assert "andor" in str(exc_info.value)
    assert exc_info.value.resource_id == "chunk-loki-1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_chunk_not_found(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """Non-existent chunk ID raises DataUnavailable."""
    scope_resolver = ScopeResolver(registry=populated_registry)
    service = ChunkService(
        scope_resolver=scope_resolver,
        vector_index=populated_index,
    )

    with pytest.raises(DataUnavailable) as exc_info:
        await service.get_chunk(chunk_id="nonexistent-chunk", series="loki")

    assert "nonexistent-chunk" in str(exc_info.value)
    assert exc_info.value.resource_id == "nonexistent-chunk"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_chunk_unknown_series(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """Unknown series raises InvalidScope."""
    scope_resolver = ScopeResolver(registry=populated_registry)
    service = ChunkService(
        scope_resolver=scope_resolver,
        vector_index=populated_index,
    )

    with pytest.raises(InvalidScope) as exc_info:
        await service.get_chunk(chunk_id="chunk-loki-1", series="unknown-series")

    assert "unknown-series" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_chunk_input_validation(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """Empty chunk_id or series raises InvalidArgument."""
    scope_resolver = ScopeResolver(registry=populated_registry)
    service = ChunkService(
        scope_resolver=scope_resolver,
        vector_index=populated_index,
    )

    with pytest.raises(InvalidArgument) as exc1:
        await service.get_chunk(chunk_id="   ", series="loki")
    assert exc1.value.argument_name == "chunk_id"

    with pytest.raises(InvalidArgument) as exc2:
        await service.get_chunk(chunk_id="chunk-loki-1", series="   ")
    assert exc2.value.argument_name == "series"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_chunk_via_fastmcp_call_tool(
    test_settings: Settings,
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """get_chunk tool is invocable directly via FastMCP call_tool."""
    memory_client = AsyncQdrantClient(":memory:")
    mcp: FastMCP = create_mcp_server(
        settings=test_settings,
        client=memory_client,
        registry=populated_registry,
        vector_index=populated_index,
    )

    result = await mcp.call_tool(
        "get_chunk",
        {"chunk_id": "chunk-loki-1", "series": "loki"},
    )
    assert result.is_error is False
    assert result.structured_content is not None

    chunk = Chunk.model_validate(result.structured_content)
    assert chunk.chunk_id == "chunk-loki-1"
    assert chunk.series_id == "loki"
    assert "glorious purpose" in chunk.text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_chunk_via_fastmcp_tenant_isolation_error(
    test_settings: Settings,
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """Cross-series chunk retrieval via call_tool triggers error wrapping [S11]."""
    memory_client = AsyncQdrantClient(":memory:")
    mcp: FastMCP = create_mcp_server(
        settings=test_settings,
        client=memory_client,
        registry=populated_registry,
        vector_index=populated_index,
    )

    with pytest.raises((DataUnavailable, ToolError)) as exc_info:
        await mcp.call_tool(
            "get_chunk",
            {"chunk_id": "chunk-loki-1", "series": "andor"},
        )
    assert "chunk-loki-1" in str(exc_info.value)


@pytest.mark.unit
def test_get_chunk_via_streamable_http(
    test_settings: Settings,
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """get_chunk is callable over Streamable HTTP POST /mcp."""
    memory_client = AsyncQdrantClient(":memory:")
    mcp = create_mcp_server(
        settings=test_settings,
        client=memory_client,
        registry=populated_registry,
        vector_index=populated_index,
    )
    app = create_http_app(mcp)

    with TestClient(app) as client:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "get_chunk",
                "arguments": {
                    "chunk_id": "chunk-loki-1",
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
        parsed_chunk_res = json.loads(data["result"]["content"][0]["text"])
        assert parsed_chunk_res["chunk_id"] == "chunk-loki-1"
        assert parsed_chunk_res["series_id"] == "loki"
        assert "glorious purpose" in parsed_chunk_res["text"]
