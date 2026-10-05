"""Reusable contract test suites for port implementations.

Every suite defines the behavioral invariants required of any adapter implementing
the port (both in-memory fakes and production network/database adapters).
"""

from datetime import UTC, datetime

import pytest

from lore_mcp.domain.documents import (
    Chunk,
    DocumentSection,
    NormalizedDocument,
    RawDocument,
)
from lore_mcp.domain.entities import SeriesManifest
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import SearchQuery
from lore_mcp.domain.types import ChunkId, SeriesId
from lore_mcp.ports.budget import BudgetCounter
from lore_mcp.ports.cache import Cache
from lore_mcp.ports.clock import Clock
from lore_mcp.ports.documents import DocumentSource, DocumentStore
from lore_mcp.ports.series_registry import SeriesRegistry
from lore_mcp.ports.vector_index import VectorIndex
from lore_mcp.ports.web_search import WebSearch


def make_test_provenance() -> Provenance:
    return Provenance(
        source_type="fandom",
        source_url="https://marvel.fandom.com/wiki/Loki",
        retrieval_timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
    )


def make_test_chunk(
    chunk_id: str = "c1",
    series_id: str = "loki",
    text: str = "Test chunk text",
    season: int | None = 1,
    episode: int | None = 1,
) -> Chunk:
    return Chunk(
        chunk_id=ChunkId(chunk_id),
        series_id=SeriesId(series_id),
        text=text,
        season=season,
        episode=episode,
        content_hash="hash1",
        provenance=make_test_provenance(),
    )


class VectorIndexContractSuite:
    """Contract test suite for VectorIndex port implementations."""

    @pytest.mark.asyncio
    async def test_upsert_and_get_by_id(self, index: VectorIndex) -> None:
        chunk = make_test_chunk("c1", "loki", "Sample content")
        await index.upsert([chunk])

        fetched = await index.get_by_id(ChunkId("c1"))
        assert fetched is not None
        assert fetched.chunk_id == "c1"
        assert fetched.text == "Sample content"

        missing = await index.get_by_id(ChunkId("nonexistent"))
        assert missing is None

    @pytest.mark.asyncio
    async def test_idempotent_upsert(self, index: VectorIndex) -> None:
        chunk1 = make_test_chunk("c1", "loki", "Initial content")
        await index.upsert([chunk1])
        assert await index.count(SeriesId("loki")) == 1

        # Second upsert with updated content must overwrite without creating duplicates
        chunk1_updated = make_test_chunk("c1", "loki", "Updated content")
        await index.upsert([chunk1_updated])
        assert await index.count(SeriesId("loki")) == 1

        fetched = await index.get_by_id(ChunkId("c1"))
        assert fetched is not None
        assert fetched.text == "Updated content"

    @pytest.mark.asyncio
    async def test_search_filters_series(self, index: VectorIndex) -> None:
        c1 = make_test_chunk("c1", "loki", "Loki in TVA")
        c2 = make_test_chunk("c2", "star-wars", "Loki in a galaxy far away")
        await index.upsert([c1, c2])

        hits = await index.search(SearchQuery(series_id=SeriesId("loki"), query="Loki"))
        assert len(hits) == 1
        assert hits[0].chunk.chunk_id == "c1"
        assert hits[0].chunk.series_id == "loki"

    @pytest.mark.asyncio
    async def test_delete_series(self, index: VectorIndex) -> None:
        c1 = make_test_chunk("c1", "loki", "Loki text")
        c2 = make_test_chunk("c2", "star-wars", "Star wars text")
        await index.upsert([c1, c2])

        await index.delete_series(SeriesId("loki"))
        assert await index.get_by_id(ChunkId("c1")) is None
        assert await index.get_by_id(ChunkId("c2")) is not None
        assert await index.count(SeriesId("loki")) == 0
        assert await index.count(SeriesId("star-wars")) == 1


class SeriesRegistryContractSuite:
    """Contract test suite for SeriesRegistry port implementations."""

    @pytest.mark.asyncio
    async def test_register_and_get(self, registry: SeriesRegistry) -> None:
        manifest = SeriesManifest(
            series_id=SeriesId("loki"),
            display_name="Loki (TV Series)",
            kind="show",
            aliases=["Loki Series", "MCU Loki"],
            seasons=[1, 2],
        )
        await registry.register_series(manifest)

        fetched = await registry.get_series(SeriesId("loki"))
        assert fetched is not None
        assert fetched.display_name == "Loki (TV Series)"
        assert fetched.aliases == ["Loki Series", "MCU Loki"]

        missing = await registry.get_series(SeriesId("unknown"))
        assert missing is None

    @pytest.mark.asyncio
    async def test_list_and_delete(self, registry: SeriesRegistry) -> None:
        m1 = SeriesManifest(series_id=SeriesId("loki"), display_name="Loki")
        m2 = SeriesManifest(series_id=SeriesId("andalor"), display_name="Andor")
        await registry.register_series(m1)
        await registry.register_series(m2)

        series_list = await registry.list_series()
        assert "loki" in series_list
        assert "andalor" in series_list

        deleted = await registry.delete_series(SeriesId("loki"))
        assert deleted is True
        assert await registry.get_series(SeriesId("loki")) is None


class CacheContractSuite:
    """Contract test suite for Cache port implementations."""

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache: Cache) -> None:
        await cache.set("key1", "val1")
        assert await cache.get("key1") == "val1"
        assert await cache.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_exists_and_delete(self, cache: Cache) -> None:
        await cache.set("key2", "val2")
        assert await cache.exists("key2") is True

        deleted = await cache.delete("key2")
        assert deleted is True
        assert await cache.exists("key2") is False
        assert await cache.get("key2") is None


class BudgetCounterContractSuite:
    """Contract test suite for BudgetCounter port implementations."""

    @pytest.mark.asyncio
    async def test_increment_and_get_count(self, counter: BudgetCounter) -> None:
        val1 = await counter.increment("budget:web", 60)
        assert val1 == 1
        val2 = await counter.increment("budget:web", 60)
        assert val2 == 2

        count = await counter.get_count("budget:web")
        assert count == 2

    @pytest.mark.asyncio
    async def test_check_budget_and_reset(self, counter: BudgetCounter) -> None:
        await counter.reset("test:key")
        assert await counter.check_budget("test:key", limit=2) is True
        await counter.increment("test:key", 60)
        assert await counter.check_budget("test:key", limit=2) is True
        await counter.increment("test:key", 60)
        # count is 2, limit is 2 -> not strictly less
        assert await counter.check_budget("test:key", limit=2) is False

        await counter.reset("test:key")
        assert await counter.get_count("test:key") == 0


class WebSearchContractSuite:
    """Contract test suite for WebSearch port implementations."""

    @pytest.mark.asyncio
    async def test_search_results(self, search: WebSearch) -> None:
        results = await search.search("TVA Loki", limit=3)
        assert isinstance(results, list)
        assert len(results) <= 3


class DocumentSourceContractSuite:
    """Contract test suite for DocumentSource port implementations."""

    @pytest.mark.asyncio
    async def test_fetch_document(self, source: DocumentSource) -> None:
        locators = await source.list_locators(SeriesId("loki"))
        assert len(locators) > 0
        doc = await source.fetch_document(SeriesId("loki"), locators[0])
        assert doc.series_id == "loki"
        assert len(doc.content) > 0


class DocumentStoreContractSuite:
    """Contract test suite for DocumentStore port implementations."""

    @pytest.mark.asyncio
    async def test_raw_and_normalized_store(self, store: DocumentStore) -> None:
        raw = RawDocument(
            series_id=SeriesId("loki"),
            source_url="https://wiki.org/1",
            source_type="fandom",
            content="Raw data",
            retrieval_timestamp=datetime.now(UTC),
        )
        await store.put_raw(raw)
        fetched_raw = await store.get_raw(SeriesId("loki"), "https://wiki.org/1")
        assert fetched_raw is not None
        assert fetched_raw.content == "Raw data"

        norm = NormalizedDocument(
            series_id=SeriesId("loki"),
            source_url="https://wiki.org/1",
            source_type="fandom",
            title="Page 1",
            text="Clean data",
            sections=[DocumentSection(heading="H", content="C")],
            content_hash="h1",
            retrieval_timestamp=datetime.now(UTC),
        )
        await store.put_normalized(norm)
        fetched_norm = await store.get_normalized(
            SeriesId("loki"), "https://wiki.org/1"
        )
        assert fetched_norm is not None
        assert fetched_norm.title == "Page 1"

        urls = await store.list_urls(SeriesId("loki"))
        assert "https://wiki.org/1" in urls


class ClockContractSuite:
    """Contract test suite for Clock port implementations."""

    def test_now_returns_datetime(self, clock: Clock) -> None:
        now = clock.now()
        assert isinstance(now, datetime)
        assert now.tzinfo is not None
