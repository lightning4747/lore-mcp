"""Unit tests for compare_events MCP tool and EventService comparison logic."""

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
from lore_mcp.domain.entities import (
    Entity,
    Event,
    EventComparison,
    SeriesManifest,
)
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.types import ChunkId, EntityId, EventId, SeriesId
from lore_mcp.interfaces.mcp import create_http_app, create_mcp_server
from lore_mcp.testing import (
    FakeEntityRegistry,
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
def sample_entities() -> list[Entity]:
    return [
        Entity(
            entity_id=EntityId("mobius"),
            series_id=SeriesId("loki"),
            name="Mobius M. Mobius",
            entity_type="character",
            aliases=["Mobius"],
            summary="TVA Analyst specializing in dangerous time variants.",
            related_entities=[EntityId("loki")],
        ),
        Entity(
            entity_id=EntityId("tva-headquarters"),
            series_id=SeriesId("loki"),
            name="TVA Headquarters",
            entity_type="location",
            aliases=["TVA"],
            summary="The headquarters of the Time Variance Authority outside space.",
        ),
    ]


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        make_chunk(
            "chunk-ep1",
            "loki",
            "Mobius interrogates Loki at TVA Headquarters regarding the variant.",
            season=1,
            episode=1,
            scene_index=4,
            narrative_order=10,
            entity_ids=["loki", "mobius", "tva-headquarters"],
            source_url="https://wiki.org/ep1-interrogation",
        ),
        make_chunk(
            "chunk-ep2",
            "loki",
            "Sylvie bombs the Sacred Timeline at Roxxcart Alabama 2050.",
            season=1,
            episode=2,
            scene_index=15,
            narrative_order=20,
            entity_ids=["sylvie", "loki"],
            source_url="https://wiki.org/ep2-bombing",
        ),
        make_chunk(
            "chunk-ep4",
            "loki",
            "A catastrophic Nexus Event on Lamentis-1 threatens to disrupt time.",
            season=1,
            episode=4,
            scene_index=5,
            narrative_order=40,
            entity_ids=["loki", "sylvie"],
            source_url="https://wiki.org/ep4-lamentis",
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
def populated_entity_registry(sample_entities: list[Entity]) -> FakeEntityRegistry:
    return FakeEntityRegistry(entities=sample_entities)


@pytest.fixture
def populated_series_registry() -> FakeSeriesRegistry:
    registry = FakeSeriesRegistry()
    registry._manifests[SeriesId("loki")] = SeriesManifest(
        series_id=SeriesId("loki"),
        display_name="Loki (TV Series)",
        kind="show",
        aliases=["MCU Loki"],
        seasons=[1, 2],
    )
    return registry


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_two_events(
    populated_series_registry: FakeSeriesRegistry,
    populated_event_registry: FakeEventRegistry,
    populated_entity_registry: FakeEntityRegistry,
    populated_index: FakeVectorIndex,
) -> None:
    """Compare two events, keeping evidence and provenance separate."""
    scope_resolver = ScopeResolver(populated_series_registry)
    retriever = HybridRetriever(populated_index)
    service = EventService(
        scope_resolver=scope_resolver,
        retriever=retriever,
        event_registry=populated_event_registry,
        entity_registry=populated_entity_registry,
    )

    comparison = await service.compare_events(
        event_id_a="bombing-sacred-timeline",
        event_id_b="nexus-lamentis",
        series="loki",
    )

    assert isinstance(comparison, EventComparison)
    assert comparison.series_id == "loki"

    # Side A checks
    assert comparison.side_a.reference == "bombing-sacred-timeline"
    assert comparison.side_a.target_type == "event"
    assert comparison.side_a.event is not None
    assert comparison.side_a.event.event_id == "bombing-sacred-timeline"
    assert "Roxxcart Alabama 2050" in comparison.side_a.locations

    # Side B checks
    assert comparison.side_b.reference == "nexus-lamentis"
    assert comparison.side_b.target_type == "event"
    assert comparison.side_b.event is not None
    assert comparison.side_b.event.event_id == "nexus-lamentis"
    assert "Lamentis-1" in comparison.side_b.locations

    # Verify provenance separation
    prov_a_urls = {p.source_url for p in comparison.side_a.provenance}
    prov_b_urls = {p.source_url for p in comparison.side_b.provenance}
    assert "https://wiki.org/ep2-bombing" in prov_a_urls
    assert "https://wiki.org/ep4-lamentis" in prov_b_urls
    # Provenance sets are distinct
    assert prov_a_urls != prov_b_urls

    # Verify evidence separation
    chunk_a_ids = {ev.chunk_id for ev in comparison.side_a.evidence}
    chunk_b_ids = {ev.chunk_id for ev in comparison.side_b.evidence}
    assert "chunk-ep2" in chunk_a_ids
    assert "chunk-ep4" in chunk_b_ids

    # Shared entities check: both events feature Loki and Sylvie
    assert EntityId("loki") in comparison.shared_entities
    assert EntityId("sylvie") in comparison.shared_entities


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_event_and_entity(
    populated_series_registry: FakeSeriesRegistry,
    populated_event_registry: FakeEventRegistry,
    populated_entity_registry: FakeEntityRegistry,
    populated_index: FakeVectorIndex,
) -> None:
    """Compare an event reference with an entity reference."""
    scope_resolver = ScopeResolver(populated_series_registry)
    retriever = HybridRetriever(populated_index)
    service = EventService(
        scope_resolver=scope_resolver,
        retriever=retriever,
        event_registry=populated_event_registry,
        entity_registry=populated_entity_registry,
    )

    comparison = await service.compare_events(
        event_id_a="bombing-sacred-timeline",
        event_id_b="mobius",
        series="loki",
    )

    # Side A is event, Side B is entity
    assert comparison.side_a.target_type == "event"
    assert comparison.side_b.target_type == "entity"
    assert comparison.side_b.entity is not None
    assert comparison.side_b.entity.entity_id == "mobius"

    # Provenance remains separate
    assert len(comparison.side_a.provenance) >= 1
    assert len(comparison.side_b.provenance) >= 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_events_validation(
    populated_series_registry: FakeSeriesRegistry,
    populated_index: FakeVectorIndex,
) -> None:
    """Validate empty references, invalid limits, and missing scopes."""
    scope_resolver = ScopeResolver(populated_series_registry)
    retriever = HybridRetriever(populated_index)
    service = EventService(
        scope_resolver=scope_resolver,
        retriever=retriever,
    )

    # Empty reference A
    with pytest.raises(InvalidArgument):
        await service.compare_events(
            event_id_a="   ",
            event_id_b="nexus-lamentis",
            series="loki",
        )

    # Empty reference B
    with pytest.raises(InvalidArgument):
        await service.compare_events(
            event_id_a="bombing-sacred-timeline",
            event_id_b="",
            series="loki",
        )

    # Invalid max_results
    with pytest.raises(InvalidArgument):
        await service.compare_events(
            event_id_a="bombing-sacred-timeline",
            event_id_b="nexus-lamentis",
            series="loki",
            max_results=-1,
        )

    # Unknown series scope
    with pytest.raises(InvalidScope):
        await service.compare_events(
            event_id_a="bombing-sacred-timeline",
            event_id_b="nexus-lamentis",
            series="unknown-series",
        )

    # Nonexistent reference
    with pytest.raises(DataUnavailable):
        await service.compare_events(
            event_id_a="Nonexistent Event Alpha",
            event_id_b="Nonexistent Event Beta",
            series="loki",
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compare_events_fastmcp_call_tool(
    test_settings: Settings,
    populated_index: FakeVectorIndex,
    populated_series_registry: FakeSeriesRegistry,
    populated_event_registry: FakeEventRegistry,
    populated_entity_registry: FakeEntityRegistry,
) -> None:
    """compare_events is invocable via FastMCP call_tool returning
    structured comparison.
    """
    memory_client = AsyncQdrantClient(":memory:")
    mcp: FastMCP = create_mcp_server(
        settings=test_settings,
        client=memory_client,
        registry=populated_series_registry,
        event_registry=populated_event_registry,
        entity_registry=populated_entity_registry,
        vector_index=populated_index,
    )

    result = await mcp.call_tool(
        "compare_events",
        {
            "event_id_a": "bombing-sacred-timeline",
            "event_id_b": "nexus-lamentis",
            "series": "loki",
        },
    )

    assert result.is_error is False
    assert result.structured_content is not None
    data = result.structured_content
    assert data["series_id"] == "loki"
    assert "side_a" in data
    assert "side_b" in data
    assert data["side_a"]["target_type"] == "event"
    assert data["side_b"]["target_type"] == "event"
    assert len(data["shared_entities"]) >= 1


@pytest.mark.unit
def test_compare_events_streamable_http(
    test_settings: Settings,
    populated_index: FakeVectorIndex,
    populated_series_registry: FakeSeriesRegistry,
    populated_event_registry: FakeEventRegistry,
    populated_entity_registry: FakeEntityRegistry,
) -> None:
    """compare_events is invocable via Streamable HTTP JSON-RPC endpoint."""
    memory_client = AsyncQdrantClient(":memory:")
    mcp = create_mcp_server(
        settings=test_settings,
        client=memory_client,
        registry=populated_series_registry,
        event_registry=populated_event_registry,
        entity_registry=populated_entity_registry,
        vector_index=populated_index,
    )
    app = create_http_app(mcp)

    with TestClient(app) as client:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "compare_events",
                "arguments": {
                    "event_id_a": "bombing-sacred-timeline",
                    "event_id_b": "nexus-lamentis",
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
