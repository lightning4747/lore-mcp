"""Unit tests for trace_event MCP tool and EventService."""

from datetime import UTC, datetime
from typing import Any

import pytest
from fastmcp import FastMCP
from qdrant_client import AsyncQdrantClient
from starlette.testclient import TestClient

from lore_mcp.application.errors import DataUnavailable, InvalidArgument, InvalidScope
from lore_mcp.application.event_service import EventService
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.application.scope import ScopeResolver
from lore_mcp.config import QdrantSettings, RequestLimitsSettings, Settings
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.entities import Event, EventTrace, SeriesManifest
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.types import ChunkId, EntityId, EventId, SeriesId
from lore_mcp.interfaces.mcp import create_http_app, create_mcp_server
from lore_mcp.testing import (
    FakeEventRegistry,
    FakeSeriesRegistry,
    FakeVectorIndex,
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
def sample_events() -> list[Event]:
    return [
        Event(
            event_id=EventId("bombing-sacred-timeline"),
            series_id=SeriesId("loki"),
            name="Bombing of the Sacred Timeline",
            description="Sylvie detonates reset charges across branches.",
            season=1,
            episode=2,
            scene_index=15,
            narrative_order=20,
            participants=[EntityId("sylvie"), EntityId("loki")],
            location="Roxxcart Alabama 2050",
            consequences=["Multiverse branches begin redlining"],
        ),
        Event(
            event_id=EventId("nexus-lamentis"),
            series_id=SeriesId("loki"),
            name="Nexus Event on Lamentis-1",
            description="Loki and Sylvie hold hands facing apocalypse.",
            season=1,
            episode=4,
            scene_index=5,
            narrative_order=40,
            participants=[EntityId("loki"), EntityId("sylvie")],
            location="Lamentis-1",
            consequences=["TVA detects branch spike and extracts them"],
        ),
    ]


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        make_chunk(
            "chunk-1",
            "loki",
            "Loki arrives at the TVA after escaping with Tesseract.",
            season=1,
            episode=1,
            scene_index=1,
            narrative_order=10,
            entity_ids=["loki", "mobius"],
            source_url="https://wiki.org/ep1",
        ),
        make_chunk(
            "chunk-2",
            "loki",
            "At Roxxcart, Sylvie executes the Bombing of the Sacred Timeline.",
            season=1,
            episode=2,
            scene_index=15,
            narrative_order=20,
            entity_ids=["sylvie", "loki"],
            source_url="https://wiki.org/ep2",
        ),
        make_chunk(
            "chunk-3",
            "loki",
            "Sylvie and Loki trigger a massive Nexus Event on Lamentis-1.",
            season=1,
            episode=4,
            scene_index=5,
            narrative_order=40,
            entity_ids=["loki", "sylvie", "mobius"],
            source_url="https://wiki.org/ep4",
        ),
        make_chunk(
            "chunk-4",
            "loki",
            "Loki destroys the Temporal Loom and takes the throne as God of Stories.",
            season=2,
            episode=6,
            scene_index=20,
            narrative_order=100,
            entity_ids=["loki", "he-who-remains"],
            source_url="https://wiki.org/ep6",
        ),
    ]


@pytest.fixture
def populated_index(sample_chunks: list[Chunk]) -> FakeVectorIndex:
    index = FakeVectorIndex()
    for c in sample_chunks:
        index._chunks[c.chunk_id] = c
    return index


@pytest.fixture
def populated_event_registry(sample_events: list[Event]) -> FakeEventRegistry:
    return FakeEventRegistry(events=sample_events)


@pytest.fixture
def populated_series_registry() -> FakeSeriesRegistry:
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
async def test_trace_event_from_registry_by_id(
    populated_series_registry: FakeSeriesRegistry,
    populated_event_registry: FakeEventRegistry,
    populated_index: FakeVectorIndex,
) -> None:
    """Resolve an event by exact ID, retrieving participants and
    chronological evidence.
    """
    scope_resolver = ScopeResolver(populated_series_registry)
    retriever = HybridRetriever(populated_index)
    service = EventService(
        scope_resolver=scope_resolver,
        retriever=retriever,
        event_registry=populated_event_registry,
    )

    trace = await service.trace_event(
        event_id="bombing-sacred-timeline",
        series="loki",
    )

    assert isinstance(trace, EventTrace)
    assert trace.event.event_id == "bombing-sacred-timeline"
    assert trace.event.name == "Bombing of the Sacred Timeline"

    # Participants and location check
    assert EntityId("sylvie") in trace.participants
    assert EntityId("loki") in trace.participants
    assert "Roxxcart Alabama 2050" in trace.locations

    # Chronological evidence check
    assert len(trace.evidence) >= 1
    # Evidence is sorted chronologically
    for i in range(len(trace.evidence) - 1):
        curr_ev = trace.evidence[i]
        next_ev = trace.evidence[i + 1]
        curr_key = (
            curr_ev.season or 0,
            curr_ev.episode or 0,
            curr_ev.narrative_order or 0,
            curr_ev.scene_index or 0,
        )
        next_key = (
            next_ev.season or 0,
            next_ev.episode or 0,
            next_ev.narrative_order or 0,
            next_ev.scene_index or 0,
        )
        assert curr_key <= next_key

    # Chronology notes
    assert len(trace.chronology) >= 1
    assert any("S1" in note for note in trace.chronology)

    # Provenance
    assert len(trace.provenance) >= 1
    assert all(p.source_url.startswith("https://") for p in trace.provenance)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_trace_event_from_registry_by_name(
    populated_series_registry: FakeSeriesRegistry,
    populated_event_registry: FakeEventRegistry,
    populated_index: FakeVectorIndex,
) -> None:
    """Resolve an event by case-insensitive name match."""
    scope_resolver = ScopeResolver(populated_series_registry)
    retriever = HybridRetriever(populated_index)
    service = EventService(
        scope_resolver=scope_resolver,
        retriever=retriever,
        event_registry=populated_event_registry,
    )

    trace = await service.trace_event(
        event_id="nexus event on lamentis-1",
        series="mcu loki",  # Alias resolution
    )

    assert trace.event.event_id == "nexus-lamentis"
    assert "Lamentis-1" in trace.locations
    assert EntityId("mobius") in trace.entities


@pytest.mark.unit
@pytest.mark.asyncio
async def test_trace_event_fallback_from_chunks(
    populated_series_registry: FakeSeriesRegistry,
    populated_index: FakeVectorIndex,
) -> None:
    """Resolve an event directly from indexed lore chunks when not in registry."""
    scope_resolver = ScopeResolver(populated_series_registry)
    retriever = HybridRetriever(populated_index)
    # No event registry provided
    service = EventService(
        scope_resolver=scope_resolver,
        retriever=retriever,
        event_registry=None,
    )

    trace = await service.trace_event(
        event_id="Temporal Loom destruction",
        series="loki",
    )

    assert trace.event.name == "Temporal Loom Destruction"
    assert trace.event.season == 2
    assert trace.event.episode == 6
    assert EntityId("loki") in trace.participants
    assert len(trace.evidence) >= 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_trace_event_not_found_raises_data_unavailable(
    populated_series_registry: FakeSeriesRegistry,
    populated_index: FakeVectorIndex,
) -> None:
    """Unknown event raises DataUnavailable with event name."""
    scope_resolver = ScopeResolver(populated_series_registry)
    retriever = HybridRetriever(populated_index)
    service = EventService(
        scope_resolver=scope_resolver,
        retriever=retriever,
    )

    with pytest.raises(DataUnavailable) as exc:
        await service.trace_event(
            event_id="Nonexistent Alien Invasion",
            series="loki",
        )
    assert "Nonexistent Alien Invasion" in str(exc.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_trace_event_validation_errors(
    populated_series_registry: FakeSeriesRegistry,
    populated_index: FakeVectorIndex,
) -> None:
    """Validation errors for empty event_id, invalid limit, or unknown scope."""
    scope_resolver = ScopeResolver(populated_series_registry)
    retriever = HybridRetriever(populated_index)
    service = EventService(
        scope_resolver=scope_resolver,
        retriever=retriever,
    )

    # Empty event_id
    with pytest.raises(InvalidArgument):
        await service.trace_event(event_id="   ", series="loki")

    # Invalid max_results
    with pytest.raises(InvalidArgument):
        await service.trace_event(
            event_id="bombing-sacred-timeline",
            series="loki",
            max_results=0,
        )

    # Unknown series scope
    with pytest.raises(InvalidScope):
        await service.trace_event(
            event_id="bombing-sacred-timeline",
            series="unknown-series",
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_trace_event_fastmcp_call_tool(
    test_settings: Settings,
    populated_index: FakeVectorIndex,
    populated_series_registry: FakeSeriesRegistry,
    populated_event_registry: FakeEventRegistry,
) -> None:
    """trace_event is invocable via FastMCP call_tool returning
    structured event trace.
    """
    memory_client = AsyncQdrantClient(":memory:")
    mcp: FastMCP = create_mcp_server(
        settings=test_settings,
        client=memory_client,
        registry=populated_series_registry,
        event_registry=populated_event_registry,
        vector_index=populated_index,
    )

    result = await mcp.call_tool(
        "trace_event",
        {
            "event_id": "bombing-sacred-timeline",
            "series": "loki",
        },
    )

    assert result.is_error is False
    assert result.structured_content is not None
    data = result.structured_content
    assert data["event"]["event_id"] == "bombing-sacred-timeline"
    assert "sylvie" in data["participants"]
    assert "Roxxcart Alabama 2050" in data["locations"]
    assert len(data["evidence"]) >= 1
    assert len(data["chronology"]) >= 1


@pytest.mark.unit
def test_trace_event_streamable_http(
    test_settings: Settings,
    populated_index: FakeVectorIndex,
    populated_series_registry: FakeSeriesRegistry,
    populated_event_registry: FakeEventRegistry,
) -> None:
    """trace_event is invocable via Streamable HTTP JSON-RPC endpoint."""
    memory_client = AsyncQdrantClient(":memory:")
    mcp = create_mcp_server(
        settings=test_settings,
        client=memory_client,
        registry=populated_series_registry,
        event_registry=populated_event_registry,
        vector_index=populated_index,
    )
    app = create_http_app(mcp)

    with TestClient(app) as client:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "trace_event",
                "arguments": {
                    "event_id": "bombing-sacred-timeline",
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
