"""Unit tests for complex query client decomposition contract (Task 8.3 & 8.4)."""

from datetime import UTC, datetime

import pytest
from qdrant_client import AsyncQdrantClient

from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.application.retrieval_service import LoreRetrievalService
from lore_mcp.application.scope import ScopeResolver
from lore_mcp.config import QdrantSettings, Settings
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.entities import SeriesManifest
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.types import ChunkId, EntityId, SeriesId
from lore_mcp.interfaces.mcp.server import create_mcp_server
from lore_mcp.testing.fakes import FakeSeriesRegistry, FakeVectorIndex


@pytest.mark.unit
class TestComplexQueryContract:
    """Verifies that the server does not decompose queries, preserves independent
    retrieval, issues decomposition notices, and advertises the contract.
    """

    @pytest.fixture
    def manifest(self) -> SeriesManifest:
        return SeriesManifest(
            series_id=SeriesId("loki"),
            display_name="Loki",
            kind="show",
            entity_aliases={
                "god of mischief": "Loki",
                "sylvie": "Sylvie Laufeydottir",
                "mobius": "Mobius M. Mobius",
                "thor": "Thor Odinson",
            },
        )

    @pytest.fixture
    def chunk(self) -> Chunk:
        return Chunk(
            chunk_id=ChunkId("chunk-1"),
            series_id=SeriesId("loki"),
            text="Loki and Sylvie fought in the Citadel at the End of Time.",
            content_hash="mock-hash",
            entity_ids=[EntityId("loki"), EntityId("sylvie")],
            entity_type="event",
            primary_concept="Citadel Battle",
            provenance=Provenance(
                source_type="fandom",
                source_url="https://marvel.fandom.com/wiki/Citadel",
                retrieval_timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            ),
        )

    @pytest.mark.asyncio
    async def test_complex_multi_entity_query_emits_guidance_notice(
        self, manifest: SeriesManifest, chunk: Chunk
    ) -> None:
        registry = FakeSeriesRegistry()
        await registry.register_series(manifest)
        index = FakeVectorIndex()
        await index.upsert([chunk])

        scope_resolver = ScopeResolver(registry)
        retriever = HybridRetriever(index)
        service = LoreRetrievalService(
            scope_resolver=scope_resolver,
            retriever=retriever,
        )

        # Complex query with two entities
        bundle = await service.search(
            query="How did Loki and Sylvie defeat Alioth?",
            series="loki",
        )

        # Notice must advise client to decompose
        assert any(
            "Query detected as multi-entity/complex" in notice
            for notice in bundle.notices
        )

    @pytest.mark.asyncio
    async def test_simple_factual_query_does_not_emit_complex_notice(
        self, manifest: SeriesManifest, chunk: Chunk
    ) -> None:
        registry = FakeSeriesRegistry()
        await registry.register_series(manifest)
        index = FakeVectorIndex()
        await index.upsert([chunk])

        scope_resolver = ScopeResolver(registry)
        retriever = HybridRetriever(index)
        service = LoreRetrievalService(
            scope_resolver=scope_resolver,
            retriever=retriever,
        )

        bundle = await service.search(
            query="When was Loki arrested?",
            series="loki",
        )

        assert not any(
            "Query detected as multi-entity/complex" in notice
            for notice in bundle.notices
        )

    @pytest.mark.asyncio
    async def test_mcp_tool_descriptions_contain_complex_query_contract(
        self,
    ) -> None:
        settings = Settings(
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
        memory_client = AsyncQdrantClient(":memory:")
        mcp = create_mcp_server(settings=settings, client=memory_client)

        tools = await mcp.list_tools()
        tool_map = {t.name: t for t in tools}

        # Check search_series_lore docstring
        tool = tool_map.get("search_series_lore")
        assert tool is not None
        assert "COMPLEX QUERY CONTRACT" in (tool.description or "")
        assert "does NOT decompose" in (tool.description or "")
        assert "Clients MUST split complex" in (tool.description or "")

        # Check search_dialogue docstring
        diag_tool = tool_map.get("search_dialogue")
        assert diag_tool is not None
        assert "COMPLEX QUERY CONTRACT" in (diag_tool.description or "")

        # Check compare_events docstring
        comp_tool = tool_map.get("compare_events")
        assert comp_tool is not None
        assert "COMPLEX QUERY CONTRACT" in (comp_tool.description or "")
