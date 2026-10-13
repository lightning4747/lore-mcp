"""Unit tests for search_dialogue MCP tool and DialogueService."""

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from fastmcp import FastMCP
from qdrant_client import AsyncQdrantClient
from starlette.testclient import TestClient

from lore_mcp.application.dialogue_service import DialogueService
from lore_mcp.application.errors import InvalidArgument, InvalidScope
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.application.scope import ScopeResolver
from lore_mcp.config import QdrantSettings, RequestLimitsSettings, Settings
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.entities import SeriesManifest
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import DialogueSnippet
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
    source_type: str = "transcript",
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
            source_type=source_type,
            source_url=source_url,
            retrieval_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )


@pytest.fixture
def populated_index() -> FakeVectorIndex:
    index = FakeVectorIndex()
    chunks = [
        # Transcript chunk with multiple lines, timestamps, and speaker labels
        make_chunk(
            "loki-trans-1",
            "loki",
            (
                "[00:01:10,000 --> 00:01:15,000] Hunter B-15: Reset the timeline.\n"
                "[00:01:16,000 --> 00:01:20,000] Loki: I am Loki of Asgard, and I am "
                "burdened with glorious purpose.\n"
                "[00:01:21,000 --> 00:01:25,000] Mobius: Yeah, about that."
            ),
            season=1,
            episode=1,
            scene_index=1,
            narrative_order=10,
            entity_ids=["loki", "mobius"],
            source_type="transcript",
            source_url="https://example.com/loki/s1e1.srt",
        ),
        # Transcript chunk in Episode 2 (Mobius talking about jet skis)
        make_chunk(
            "loki-trans-2",
            "loki",
            (
                "[00:15:20,000 --> 00:15:26,000] Mobius: You know what they say about "
                "jet skis? They are pure joy.\n"
                "[00:15:27,000 --> 00:15:30,000] Loki: Truly absurd."
            ),
            season=1,
            episode=2,
            scene_index=5,
            narrative_order=25,
            entity_ids=["mobius", "loki"],
            source_type="transcript",
            source_url="https://example.com/loki/s1e2.srt",
        ),
        # Transcript chunk without explicit speaker label (e.g. ambient/PA announcer)
        make_chunk(
            "loki-trans-3",
            "loki",
            (
                "[00:05:00,000 --> 00:05:04,000] Please take a ticket and "
                "wait for your variant trial."
            ),
            season=1,
            episode=1,
            scene_index=2,
            narrative_order=12,
            entity_ids=[],
            source_type="transcript",
            source_url="https://example.com/loki/s1e1.srt",
        ),
        # Non-transcript wiki chunk with matching text (MUST BE IGNORED)
        make_chunk(
            "loki-wiki-1",
            "loki",
            (
                "Loki repeated his famous line 'I am burdened with "
                "glorious purpose' in 2012."
            ),
            season=1,
            episode=1,
            scene_index=1,
            narrative_order=10,
            entity_ids=["loki"],
            source_type="fandom",
            source_url="https://marvel.fandom.com/wiki/Loki",
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
        entity_count=2,
        chunk_count=4,
    )
    reg._manifests[SeriesId("loki")] = manifest
    return reg


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_dialogue_transcript_only(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """search_dialogue queries transcript sources only, ignoring wiki/fandom hits."""
    scope_resolver = ScopeResolver(registry=populated_registry)
    retriever = HybridRetriever(index=populated_index)
    service = DialogueService(
        scope_resolver=scope_resolver,
        retriever=retriever,
    )

    results = await service.search_dialogue(
        query="glorious purpose",
        series="loki",
    )
    assert len(results) >= 1
    for snippet in results:
        assert snippet.provenance.source_type == "transcript"
        assert "wiki" not in snippet.provenance.source_url


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_dialogue_extracts_locators_and_snippets(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """Extract short dialogue quote with season, episode, timestamp, speaker."""
    scope_resolver = ScopeResolver(registry=populated_registry)

    retriever = HybridRetriever(index=populated_index)
    service = DialogueService(
        scope_resolver=scope_resolver,
        retriever=retriever,
    )

    results = await service.search_dialogue(
        query="glorious purpose",
        series="loki",
    )
    assert len(results) >= 1
    snippet = results[0]
    assert isinstance(snippet, DialogueSnippet)
    assert snippet.season == 1
    assert snippet.episode == 1
    assert snippet.timestamp == "00:01:16,000 --> 00:01:20,000"
    assert snippet.speaker == "Loki"
    assert "glorious purpose" in snippet.content
    # Ensure it's not the full 3-line chunk text
    assert "Hunter B-15" not in snippet.content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_dialogue_speaker_only_when_known(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """Speaker locator is populated when known, and None when untagged/unknown."""
    scope_resolver = ScopeResolver(registry=populated_registry)
    retriever = HybridRetriever(index=populated_index)
    service = DialogueService(
        scope_resolver=scope_resolver,
        retriever=retriever,
    )

    # Ambient PA announcement without speaker
    results = await service.search_dialogue(
        query="variant trial",
        series="loki",
    )
    assert len(results) >= 1
    snippet = results[0]
    assert snippet.speaker is None
    assert snippet.timestamp == "00:05:00,000 --> 00:05:04,000"
    assert "variant trial" in snippet.content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_dialogue_episode_filter(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """Filter dialogue search to a specific episode."""
    scope_resolver = ScopeResolver(registry=populated_registry)
    retriever = HybridRetriever(index=populated_index)
    service = DialogueService(
        scope_resolver=scope_resolver,
        retriever=retriever,
    )

    # Search in episode 2
    results_ep2 = await service.search_dialogue(
        query="joy",
        series="loki",
        season=1,
        episode=2,
    )
    assert len(results_ep2) == 1
    assert results_ep2[0].episode == 2
    assert "jet skis" in results_ep2[0].content

    # Searching for episode 2 line while restricting to episode 1 returns empty
    results_ep1 = await service.search_dialogue(
        query="jet skis",
        series="loki",
        season=1,
        episode=1,
    )
    assert len(results_ep1) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_dialogue_speaker_filter(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """Filter dialogue search to a specific character speaker."""
    scope_resolver = ScopeResolver(registry=populated_registry)
    retriever = HybridRetriever(index=populated_index)
    service = DialogueService(
        scope_resolver=scope_resolver,
        retriever=retriever,
    )

    results = await service.search_dialogue(
        query="tick",
        series="loki",
        speaker="Mobius",
    )
    # Speaker filter matches Mobius
    for res in results:
        if res.speaker:
            assert "mobius" in res.speaker.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_dialogue_validation_and_errors(
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """Empty query raises InvalidArgument, unknown series raises InvalidScope."""
    scope_resolver = ScopeResolver(registry=populated_registry)
    retriever = HybridRetriever(index=populated_index)
    service = DialogueService(
        scope_resolver=scope_resolver,
        retriever=retriever,
    )

    with pytest.raises(InvalidArgument) as exc1:
        await service.search_dialogue(query="   ", series="loki")
    assert exc1.value.argument_name == "query"

    with pytest.raises(InvalidScope) as exc2:
        await service.search_dialogue(query="hello", series="unknown-series")
    assert "unknown-series" in str(exc2.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_dialogue_via_fastmcp_call_tool(
    test_settings: Settings,
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """search_dialogue tool is invocable via FastMCP call_tool
    returning structured snippets.
    """
    memory_client = AsyncQdrantClient(":memory:")
    mcp: FastMCP = create_mcp_server(
        settings=test_settings,
        client=memory_client,
        registry=populated_registry,
        vector_index=populated_index,
    )

    result = await mcp.call_tool(
        "search_dialogue",
        {"query": "glorious purpose", "series": "loki"},
    )
    assert result.is_error is False
    assert result.structured_content is not None

    data = result.structured_content
    raw_snippets = (
        data["result"] if isinstance(data, dict) and "result" in data else data
    )
    assert isinstance(raw_snippets, list)
    assert len(raw_snippets) >= 1
    snippet = DialogueSnippet.model_validate(raw_snippets[0])

    assert snippet.speaker == "Loki"
    assert "glorious purpose" in snippet.content
    assert snippet.season == 1
    assert snippet.episode == 1


@pytest.mark.unit
def test_search_dialogue_via_streamable_http(
    test_settings: Settings,
    populated_index: FakeVectorIndex,
    populated_registry: FakeSeriesRegistry,
) -> None:
    """search_dialogue is callable over Streamable HTTP POST /mcp."""
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
            "id": 13,
            "method": "tools/call",
            "params": {
                "name": "search_dialogue",
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
            lines = res.text.strip().splitlines()
            data_line = next(line for line in lines if line.startswith("data: "))
            data = json.loads(data_line.removeprefix("data: ").strip())
        else:
            data = res.json()

        assert data.get("jsonrpc") == "2.0"
        assert "result" in data
        assert "content" in data["result"]
        parsed_res = json.loads(data["result"]["content"][0]["text"])
        assert isinstance(parsed_res, list)
        assert len(parsed_res) >= 1
        assert parsed_res[0]["speaker"] == "Loki"
        assert "glorious purpose" in parsed_res[0]["content"]
