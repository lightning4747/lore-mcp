"""Unit tests for search_series_lore MCP tool and LoreRetrievalService."""

from datetime import UTC, datetime
from typing import Any

import pytest
from qdrant_client import AsyncQdrantClient
from starlette.testclient import TestClient

from lore_mcp.application.errors import InvalidScope
from lore_mcp.config import QdrantSettings, Settings
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.entities import SeriesManifest
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import EvidenceBundle, WebSearchResult
from lore_mcp.domain.types import ChunkId, EntityId, SeriesId
from lore_mcp.interfaces.mcp import create_http_app, create_mcp_server
from lore_mcp.testing import (
    FakeBudgetCounter,
    FakeCache,
    FakeSeriesRegistry,
    FakeVectorIndex,
    FakeWebSearch,
)


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


def make_chunk(
    chunk_id: str,
    series_id: str,
    text: str,
    season: int | None = 1,
    episode: int | None = 1,
    scene_index: int | None = 1,
    narrative_order: int | None = 1,
    entity_ids: list[str] | None = None,
    source_type: str = "transcript",
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
            source_type=source_type,
            source_url=f"https://example.com/{series_id}/{chunk_id}",
            retrieval_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )


@pytest.fixture
def populated_index() -> FakeVectorIndex:
    index = FakeVectorIndex()
    chunks = [
        make_chunk(
            "loki-1",
            "loki",
            "Loki: I am burdened with glorious purpose.",
            season=1,
            episode=1,
            scene_index=1,
            narrative_order=10,
            entity_ids=["loki"],
            source_type="transcript",
        ),
        make_chunk(
            "loki-2",
            "loki",
            "Mobius offers Loki a cup of coffee at the TVA interrogation table.",
            season=1,
            episode=1,
            scene_index=2,
            narrative_order=12,
            entity_ids=["loki", "mobius", "tva"],
            source_type="transcript",
        ),
        make_chunk(
            "loki-3",
            "loki",
            "Victor Timely invents the temporal loom throughput multiplier.",
            season=2,
            episode=4,
            scene_index=5,
            narrative_order=85,
            entity_ids=["victor-timely", "tva"],
            source_type="fandom",
        ),
    ]
    for c in chunks:
        index._chunks[c.chunk_id] = c
    return index


@pytest.fixture
def populated_registry() -> FakeSeriesRegistry:
    registry = FakeSeriesRegistry()
    registry._manifests[SeriesId("loki")] = SeriesManifest(
        series_id=SeriesId("loki"),
        display_name="Loki (TV Series)",
        kind="show",
        aliases=["MCU Loki", "God of Stories"],
        seasons=[1, 2],
    )
    return registry


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_series_lore_basic(
    test_settings: Settings,
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """search_series_lore performs scope resolution, hybrid retrieval, and
    returns EvidenceBundle.
    """
    memory_client = AsyncQdrantClient(":memory:")
    mcp = create_mcp_server(
        settings=test_settings,
        client=memory_client,
        registry=populated_registry,
        vector_index=populated_index,
    )

    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "search_series_lore" in tool_names

    # Call tool
    result = await mcp.call_tool(
        "search_series_lore",
        {"query": "glorious purpose", "series": "MCU Loki"},
    )
    assert result.is_error is False
    assert result.structured_content is not None

    bundle = EvidenceBundle.model_validate(result.structured_content)
    assert bundle.status == "ok"
    assert bundle.series_id == "loki"
    assert len(bundle.evidence) >= 1
    assert "glorious purpose" in bundle.evidence[0].content
    assert bundle.evidence[0].provenance.source_type == "transcript"
    assert len(bundle.chronology) >= 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_series_lore_caching(
    test_settings: Settings,
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """Subsequent identical searches return cached bundle with 'Cached result'
    notice.
    """
    cache = FakeCache()
    memory_client = AsyncQdrantClient(":memory:")
    mcp = create_mcp_server(
        settings=test_settings,
        client=memory_client,
        registry=populated_registry,
        vector_index=populated_index,
        cache=cache,
    )

    # First call - cache miss
    res1 = await mcp.call_tool(
        "search_series_lore",
        {"query": "Mobius coffee", "series": "loki"},
    )
    assert res1.structured_content is not None
    data1 = res1.structured_content
    assert "Cached result" not in data1.get("notices", [])

    # Second call - cache hit
    res2 = await mcp.call_tool(
        "search_series_lore",
        {"query": "Mobius coffee", "series": "loki"},
    )
    assert res2.structured_content is not None
    data2 = res2.structured_content
    assert "Cached result" in data2.get("notices", [])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_series_lore_spoiler_filtering(
    test_settings: Settings,
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """Spoiler bounds exclude Season 2 lore when max_season=1."""
    memory_client = AsyncQdrantClient(":memory:")
    mcp = create_mcp_server(
        settings=test_settings,
        client=memory_client,
        registry=populated_registry,
        vector_index=populated_index,
    )

    result = await mcp.call_tool(
        "search_series_lore",
        {
            "query": "multiplier",
            "series": "loki",
            "max_season": 1,
        },
    )
    assert result.structured_content is not None
    bundle = EvidenceBundle.model_validate(result.structured_content)
    # Season 2 chunk should NOT be in evidence
    assert all(ev.season is not None and ev.season <= 1 for ev in bundle.evidence)
    assert any("Spoiler filter active" in n for n in bundle.notices)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_series_lore_unknown_scope(
    test_settings: Settings,
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """Unknown scope raises ToolError (wrapping InvalidScope) with suggestions."""
    from fastmcp.exceptions import ToolError

    memory_client = AsyncQdrantClient(":memory:")
    mcp = create_mcp_server(
        settings=test_settings,
        client=memory_client,
        registry=populated_registry,
        vector_index=populated_index,
    )

    with pytest.raises((InvalidScope, ToolError)) as exc_info:
        await mcp.call_tool(
            "search_series_lore",
            {"query": "test", "series": "unknown-series"},
        )
    assert "unknown-series" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_series_lore_web_fallback(
    test_settings: Settings,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """When vector store yields no hits, web search fallback is executed."""
    empty_index = FakeVectorIndex()
    web_search = FakeWebSearch()
    web_search.set_default_results(
        [
            WebSearchResult(
                url="https://marvel.com/he-who-remains",
                title="He Who Remains Lore",
                snippet="He Who Remains lives at the Citadel at the End of Time.",
            )
        ]
    )
    budget = FakeBudgetCounter()

    memory_client = AsyncQdrantClient(":memory:")
    mcp = create_mcp_server(
        settings=test_settings,
        client=memory_client,
        registry=populated_registry,
        vector_index=empty_index,
        web_search=web_search,
        budget_counter=budget,
    )

    result = await mcp.call_tool(
        "search_series_lore",
        {"query": "Citadel at the End of Time", "series": "loki"},
    )
    assert result.structured_content is not None
    bundle = EvidenceBundle.model_validate(result.structured_content)
    assert bundle.status == "fallback_used"
    assert len(bundle.evidence) == 1
    assert "Citadel at the End of Time" in bundle.evidence[0].content
    assert bundle.evidence[0].provenance.source_type == "web"


@pytest.mark.unit
def test_search_series_lore_via_streamable_http(
    test_settings: Settings,
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """search_series_lore is callable over Streamable HTTP POST /mcp."""
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
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "search_series_lore",
                "arguments": {
                    "query": "glorious purpose",
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
            import json

            lines = res.text.strip().splitlines()
            data_line = next(line for line in lines if line.startswith("data: "))
            data = json.loads(data_line.removeprefix("data: ").strip())
        else:
            data = res.json()

        assert data.get("jsonrpc") == "2.0"
        assert "result" in data
        assert "content" in data["result"]
