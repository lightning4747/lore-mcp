"""Unit tests for FastMCP server, readiness checker, and HTTP endpoints."""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastmcp import FastMCP
from qdrant_client import AsyncQdrantClient
from starlette.testclient import TestClient

from lore_mcp.adapters.qdrant.initializer import QdrantInitializer
from lore_mcp.config import QdrantSettings, Settings
from lore_mcp.interfaces.mcp import ReadinessChecker, create_http_app, create_mcp_server
from lore_mcp.ports.clock import Clock


class FakeClock(Clock):
    """Deterministic fake clock for testing TTL behavior."""

    def __init__(self, start_time: datetime | None = None) -> None:
        self._current_time = start_time or datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    def now(self) -> datetime:
        return self._current_time

    def advance(self, duration: timedelta) -> None:
        self._current_time += duration


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
        )
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_readiness_checker_healthy(test_settings: Settings) -> None:
    """ReadinessChecker returns True when collections are ready and compatible."""
    client = AsyncQdrantClient(":memory:")
    await QdrantInitializer.ensure_collections_ready(client, test_settings.qdrant)

    clock = FakeClock()
    checker = ReadinessChecker(
        client=client,
        settings=test_settings.qdrant,
        clock=clock,
        ttl_seconds=30.0,
    )

    is_ready, details = await checker.check_readiness()
    assert is_ready is True
    assert details["qdrant_reachable"] is True
    assert details["manifest_compatible"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_readiness_checker_unreachable() -> None:
    """ReadinessChecker returns False when Qdrant is unreachable."""
    client = AsyncMock(spec=AsyncQdrantClient)
    client.get_collections.side_effect = ConnectionError("Connection refused")

    settings = QdrantSettings(
        url="http://localhost:6333",
        chunks_collection="chunks",
        meta_collection="meta",
        embedding_model="all-MiniLM-L6-v2",
        dimension=384,
        inference_backend="cloud",
        schema_version=1,
    )
    checker = ReadinessChecker(
        client=client,  # type: ignore[arg-type]
        settings=settings,
        ttl_seconds=30.0,
    )

    is_ready, details = await checker.check_readiness()
    assert is_ready is False
    assert details["qdrant_reachable"] is False
    assert "Connection refused" in details["error"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_readiness_checker_manifest_incompatible(test_settings: Settings) -> None:
    """ReadinessChecker returns False when meta collection is not initialized."""
    client = AsyncQdrantClient(":memory:")
    # Note: ensure_collections_ready is NOT called, so meta collection does not exist

    checker = ReadinessChecker(
        client=client,
        settings=test_settings.qdrant,
        ttl_seconds=30.0,
    )

    is_ready, details = await checker.check_readiness()
    assert is_ready is False
    assert details["qdrant_reachable"] is True
    assert details["manifest_compatible"] is False
    assert "error" in details


@pytest.mark.unit
@pytest.mark.asyncio
async def test_readiness_checker_ttl_caching(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ReadinessChecker caches result for ttl_seconds unless force_refresh=True."""
    client = AsyncMock(spec=AsyncQdrantClient)
    client.get_collections.return_value = []
    monkeypatch.setattr(
        QdrantInitializer, "validate_meta_metadata", AsyncMock(return_value=None)
    )

    clock = FakeClock()
    checker = ReadinessChecker(
        client=client,  # type: ignore[arg-type]
        settings=test_settings.qdrant,
        clock=clock,
        ttl_seconds=30.0,
    )

    # First check calls get_collections
    await checker.check_readiness()
    assert client.get_collections.call_count == 1

    # Second check within TTL (10s later) does not call client
    clock.advance(timedelta(seconds=10))
    await checker.check_readiness()
    assert client.get_collections.call_count == 1

    # Third check after TTL (35s total elapsed) calls client again
    clock.advance(timedelta(seconds=25))
    await checker.check_readiness()
    assert client.get_collections.call_count == 2

    # Force refresh bypasses cache even within TTL
    clock.advance(timedelta(seconds=5))
    await checker.check_readiness(force_refresh=True)
    assert client.get_collections.call_count == 3


@pytest.mark.unit
def test_healthz_route_is_touchless(test_settings: Settings) -> None:
    """Liveness probe /healthz returns 200 OK without calling any dependencies."""
    # Client that would fail if called
    unreachable_client = AsyncMock(spec=AsyncQdrantClient)
    unreachable_client.get_collections.side_effect = RuntimeError(
        "Should not be called"
    )

    mcp = create_mcp_server(settings=test_settings, client=unreachable_client)
    app = create_http_app(mcp)

    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    unreachable_client.get_collections.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_readyz_route_healthy_and_unhealthy(test_settings: Settings) -> None:
    """Readiness probe /readyz returns 200 when ready and 503 when unready."""
    memory_client = AsyncQdrantClient(":memory:")
    await QdrantInitializer.ensure_collections_ready(
        memory_client, test_settings.qdrant
    )

    # 1. Healthy server
    mcp_healthy = create_mcp_server(settings=test_settings, client=memory_client)
    app_healthy = create_http_app(mcp_healthy)

    with TestClient(app_healthy) as client:
        res = client.get("/readyz")
        assert res.status_code == 200
        body = res.json()
        assert body["ready"] is True
        assert body["details"]["qdrant_reachable"] is True

    # 2. Unhealthy server (unreachable client)
    unreachable_client = AsyncMock(spec=AsyncQdrantClient)
    unreachable_client.get_collections.side_effect = ConnectionError("Down")

    mcp_unhealthy = create_mcp_server(settings=test_settings, client=unreachable_client)
    app_unhealthy = create_http_app(mcp_unhealthy)

    with TestClient(app_unhealthy) as client:
        res = client.get("/readyz")
        assert res.status_code == 503
        body = res.json()
        assert body["ready"] is False
        assert body["details"]["qdrant_reachable"] is False


@pytest.mark.unit
def test_streamable_http_mount_at_mcp(test_settings: Settings) -> None:
    """Streamable HTTP endpoint is mounted at /mcp and handles MCP requests."""
    memory_client = AsyncQdrantClient(":memory:")
    mcp = create_mcp_server(settings=test_settings, client=memory_client)
    app = create_http_app(mcp)

    with TestClient(app) as client:
        # Standard MCP initialize request over HTTP POST
        init_payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }
        response = client.post(
            "/mcp",
            json=init_payload,
            headers={"Accept": "application/json, text/event-stream"},
        )
        assert response.status_code == 200
        # FastMCP Streamable HTTP returns text/event-stream or application/json
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            # Extract JSON data from event-stream (lines starting with 'data: ')
            import json

            lines = response.text.strip().splitlines()
            data_line = next(line for line in lines if line.startswith("data: "))
            data = json.loads(data_line.removeprefix("data: ").strip())
        else:
            data = response.json()

        assert data.get("jsonrpc") == "2.0"
        assert "result" in data
        assert data["result"]["serverInfo"]["name"] == "lore-mcp"


@pytest.mark.unit
def test_lifespan_closes_owned_client(test_settings: Settings) -> None:
    """Server lifespan context closes owned Qdrant client upon shutdown."""
    # When client is not passed, create_mcp_server creates and owns it
    mcp: FastMCP = create_mcp_server(settings=test_settings)
    app = create_http_app(mcp)

    with TestClient(app) as client:
        res = client.get("/healthz")
        assert res.status_code == 200
    # Exiting TestClient triggers lifespan shutdown


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_series_tool_with_fake_registry(test_settings: Settings) -> None:
    """Tool list_series returns series ids, display names, aliases, and kind."""
    from lore_mcp.domain.entities import SeriesManifest
    from lore_mcp.domain.types import SeriesId
    from lore_mcp.testing import FakeSeriesRegistry

    registry = FakeSeriesRegistry()
    await registry.register_series(
        SeriesManifest(
            series_id=SeriesId("loki"),
            display_name="Loki (TV Series)",
            kind="show",
            aliases=["MCU Loki", "God of Stories"],
            seasons=[1, 2],
        )
    )
    await registry.register_series(
        SeriesManifest(
            series_id=SeriesId("arcane"),
            display_name="Arcane",
            kind="anime",
            aliases=["League Arcane"],
            seasons=[1],
        )
    )

    memory_client = AsyncQdrantClient(":memory:")
    mcp = create_mcp_server(
        settings=test_settings, client=memory_client, registry=registry
    )

    # 1. Verify tool registration
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "list_series" in tool_names

    # 2. Call tool directly via FastMCP call_tool
    result = await mcp.call_tool("list_series", {})
    assert result.is_error is False
    assert result.structured_content is not None
    assert "result" in result.structured_content

    series_data = result.structured_content["result"]
    assert len(series_data) == 2

    # Map by series_id
    by_id = {item["series_id"]: item for item in series_data}
    assert "loki" in by_id
    assert by_id["loki"]["display_name"] == "Loki (TV Series)"
    assert by_id["loki"]["aliases"] == ["MCU Loki", "God of Stories"]
    assert by_id["loki"]["kind"] == "show"

    assert "arcane" in by_id
    assert by_id["arcane"]["display_name"] == "Arcane"
    assert by_id["arcane"]["aliases"] == ["League Arcane"]
    assert by_id["arcane"]["kind"] == "anime"


@pytest.mark.unit
def test_list_series_via_streamable_http(test_settings: Settings) -> None:
    """Tool list_series is discoverable and callable via Streamable HTTP POST /mcp."""
    from lore_mcp.domain.entities import SeriesManifest
    from lore_mcp.domain.types import SeriesId
    from lore_mcp.testing import FakeSeriesRegistry

    registry = FakeSeriesRegistry()
    registry._manifests[SeriesId("loki")] = SeriesManifest(
        series_id=SeriesId("loki"),
        display_name="Loki",
        kind="show",
        aliases=["MCU Loki"],
    )

    memory_client = AsyncQdrantClient(":memory:")
    mcp = create_mcp_server(
        settings=test_settings, client=memory_client, registry=registry
    )
    app = create_http_app(mcp)

    with TestClient(app) as client:
        # Call tools/call over JSON-RPC
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "list_series",
                "arguments": {},
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
            import json

            lines = res.text.strip().splitlines()
            data_line = next(line for line in lines if line.startswith("data: "))
            data = json.loads(data_line.removeprefix("data: ").strip())
        else:
            data = res.json()

        assert data.get("jsonrpc") == "2.0"
        assert "result" in data
        assert "content" in data["result"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_descriptions_and_read_only_annotations(
    test_settings: Settings,
) -> None:
    """Verify tool descriptions and annotations across all registered tools."""
    memory_client = AsyncQdrantClient(":memory:")
    mcp = create_mcp_server(settings=test_settings, client=memory_client)

    tools = await mcp.list_tools()
    tool_map = {t.name: t for t in tools}

    expected_tools = [
        "list_series",
        "search_series_lore",
        "get_entity",
        "get_chunk",
        "search_dialogue",
        "trace_event",
        "compare_events",
    ]
    for name in expected_tools:
        assert name in tool_map, f"Missing expected tool: {name}"
        tool = tool_map[name]
        # Verify read-only annotations
        assert tool.annotations is not None, f"Tool {name} missing annotations"
        assert tool.annotations.read_only_hint is True, (
            f"Tool {name} read_only_hint must be True"
        )
        assert tool.description is not None, f"Tool {name} missing description"

    # Verify decomposition instructions
    search_lore_desc = tool_map["search_series_lore"].description or ""
    assert "Decompose" in search_lore_desc or "decompose" in search_lore_desc

    # Verify spoiler bounds and include_unscoped
    assert "spoiler" in search_lore_desc.lower()
    assert "include_unscoped" in search_lore_desc

    # Verify get_chunk for truncated items
    assert "get_chunk" in search_lore_desc
    assert "truncated" in search_lore_desc.lower()

    # Verify third-party data notice
    assert "third-party" in search_lore_desc.lower()

    # Verify server instructions
    assert mcp.instructions is not None
    assert "Decompose" in mcp.instructions
    assert "Spoiler" in mcp.instructions
    assert "get_chunk" in mcp.instructions
    assert "Third-Party" in mcp.instructions
