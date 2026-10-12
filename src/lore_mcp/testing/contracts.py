"""Reusable contract test suites for port implementations.

Every suite defines the behavioral invariants required of any adapter implementing
the port (both in-memory fakes and production network/database adapters).
"""

import asyncio
from datetime import UTC, datetime

import pytest

from lore_mcp.domain.documents import (
    Chunk,
    DocumentSection,
    NormalizedDocument,
    RawDocument,
)
from lore_mcp.domain.entities import Entity, SeriesManifest
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import SearchQuery
from lore_mcp.domain.types import ChunkId, EntityId, SeriesId
from lore_mcp.ports.budget import BudgetCounter
from lore_mcp.ports.cache import Cache
from lore_mcp.ports.clock import Clock
from lore_mcp.ports.documents import DocumentSource, DocumentStore
from lore_mcp.ports.entity_registry import EntityRegistry
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
    entity_ids: list[str] | None = None,
    source_type: str = "fandom",
) -> Chunk:
    return Chunk(
        chunk_id=ChunkId(chunk_id),
        series_id=SeriesId(series_id),
        text=text,
        season=season,
        episode=episode,
        content_hash="hash1",
        entity_ids=[EntityId(e) for e in (entity_ids or [])],
        provenance=Provenance(
            source_type=source_type,
            source_url="https://marvel.fandom.com/wiki/Loki",
            retrieval_timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        ),
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
    async def test_search_dense_ranks_candidates_with_scores(
        self, index: VectorIndex
    ) -> None:
        c1 = make_test_chunk("c1", "loki", "Loki and Sylvie at the Citadel")
        c2 = make_test_chunk("c2", "loki", "Loki in TVA archives")
        c3 = make_test_chunk("c3", "star-wars", "Loki cross series leak")
        await index.upsert([c1, c2, c3])

        query = SearchQuery(series_id=SeriesId("loki"), query="Sylvie Citadel")
        hits = await index.search_dense(query)

        assert len(hits) >= 1
        assert all(h.chunk.series_id == "loki" for h in hits)
        for h in hits:
            assert h.dense_score is not None
            assert h.score == h.dense_score
            assert h.chunk.text != ""
        # Descending order verification
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_search_sparse_ranks_candidates_with_scores(
        self, index: VectorIndex
    ) -> None:
        c1 = make_test_chunk("c1", "loki", "Loki and Sylvie at the Citadel")
        c2 = make_test_chunk("c2", "loki", "Loki in TVA archives")
        c3 = make_test_chunk("c3", "star-wars", "Sylvie in wrong galaxy")
        await index.upsert([c1, c2, c3])

        query = SearchQuery(series_id=SeriesId("loki"), query="Sylvie Citadel")
        hits = await index.search_sparse(query)

        assert len(hits) >= 1
        assert all(h.chunk.series_id == "loki" for h in hits)
        for h in hits:
            assert h.sparse_score is not None
            assert h.score == h.sparse_score
            assert h.chunk.text != ""
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_parallel_dense_and_sparse_search(self, index: VectorIndex) -> None:
        c1 = make_test_chunk("c1", "loki", "Loki and Sylvie at the Citadel")
        await index.upsert([c1])

        query = SearchQuery(series_id=SeriesId("loki"), query="Sylvie Citadel")
        dense_hits, sparse_hits = await asyncio.gather(
            index.search_dense(query),
            index.search_sparse(query),
        )

        assert len(dense_hits) >= 1
        assert len(sparse_hits) >= 1
        assert dense_hits[0].dense_score is not None
        assert sparse_hits[0].sparse_score is not None

    @pytest.mark.asyncio
    async def test_metadata_filters_equality_and_entities(
        self, index: VectorIndex
    ) -> None:
        c1 = make_test_chunk(
            "c1",
            "loki",
            "Loki and Mobius at TVA",
            season=1,
            episode=1,
            entity_ids=["loki", "mobius"],
            source_type="transcript",
        )
        c2 = make_test_chunk(
            "c2",
            "loki",
            "Sylvie at Citadel",
            season=1,
            episode=6,
            entity_ids=["sylvie"],
            source_type="fandom",
        )
        c3 = make_test_chunk(
            "c3",
            "loki",
            "Loki and Sylvie in void",
            season=1,
            episode=5,
            entity_ids=["loki", "sylvie"],
            source_type="transcript",
        )
        await index.upsert([c1, c2, c3])

        # Filter by season and episode
        hits = await index.search_dense(
            SearchQuery(series_id=SeriesId("loki"), query="Loki", season=1, episode=1)
        )
        assert len(hits) == 1
        assert hits[0].chunk.chunk_id == "c1"

        # Filter by entity
        hits_entity = await index.search_dense(
            SearchQuery(series_id=SeriesId("loki"), query="Loki", entity="mobius")
        )
        assert len(hits_entity) == 1
        assert hits_entity[0].chunk.chunk_id == "c1"

        # Filter by source_type
        hits_src = await index.search_dense(
            SearchQuery(
                series_id=SeriesId("loki"), query="Citadel", source_type="fandom"
            )
        )
        assert len(hits_src) == 1
        assert hits_src[0].chunk.chunk_id == "c2"

    @pytest.mark.asyncio
    async def test_metadata_filters_spoiler_bounds_and_unscoped(
        self, index: VectorIndex
    ) -> None:
        c_s1e6 = make_test_chunk("c1", "loki", "End of season 1", season=1, episode=6)
        c_s2e2 = make_test_chunk(
            "c2", "loki", "Early season 2 Brad Wolfe", season=2, episode=2
        )
        c_s2e5 = make_test_chunk(
            "c3", "loki", "Late season 2 timeline collapse", season=2, episode=5
        )
        c_s3e1 = make_test_chunk(
            "c4", "loki", "Season 3 hypothetical", season=3, episode=1
        )
        c_unscoped = make_test_chunk(
            "c5", "loki", "General background lore", season=None, episode=None
        )
        await index.upsert([c_s1e6, c_s2e2, c_s2e5, c_s3e1, c_unscoped])

        # Spoiler bound: max_season=2, max_episode=2, include_unscoped=False
        # Should include S1E6 and S2E2, but NOT S2E5, S3E1, or unscoped
        hits_bound = await index.search_dense(
            SearchQuery(
                series_id=SeriesId("loki"),
                query="Loki",
                max_season=2,
                max_episode=2,
                include_unscoped=False,
            )
        )
        hit_ids = {h.chunk.chunk_id for h in hits_bound}
        assert hit_ids == {ChunkId("c1"), ChunkId("c2")}

        # Spoiler bound: max_season=2, max_episode=2, include_unscoped=True
        # Should include S1E6, S2E2, and unscoped
        hits_unscoped = await index.search_dense(
            SearchQuery(
                series_id=SeriesId("loki"),
                query="Loki",
                max_season=2,
                max_episode=2,
                include_unscoped=True,
            )
        )
        hit_ids_unscoped = {h.chunk.chunk_id for h in hits_unscoped}
        assert hit_ids_unscoped == {ChunkId("c1"), ChunkId("c2"), ChunkId("c5")}

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

    @pytest.mark.asyncio
    async def test_get_manifests_and_aliases(self, registry: SeriesRegistry) -> None:
        m1 = SeriesManifest(
            series_id=SeriesId("loki"),
            display_name="Loki",
            aliases=["MCU Loki", "God of Stories"],
        )
        m2 = SeriesManifest(
            series_id=SeriesId("andor"),
            display_name="Andor",
            aliases=["Cassian"],
        )
        await registry.register_series(m1)
        await registry.register_series(m2)

        manifests = await registry.get_manifests()
        series_ids = {m.series_id for m in manifests}
        assert SeriesId("loki") in series_ids
        assert SeriesId("andor") in series_ids

        alias_map = await registry.get_alias_map()
        assert alias_map["loki"] == SeriesId("loki")
        assert alias_map["mcu loki"] == SeriesId("loki")
        assert alias_map["god of stories"] == SeriesId("loki")
        assert alias_map["andor"] == SeriesId("andor")
        assert alias_map["cassian"] == SeriesId("andor")

        assert await registry.resolve_alias("LOKI") == SeriesId("loki")
        assert await registry.resolve_alias("mcu loki") == SeriesId("loki")
        assert await registry.resolve_alias("Cassian") == SeriesId("andor")
        assert await registry.resolve_alias("nonexistent") is None


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


class EntityRegistryContractSuite:
    """Contract test suite for EntityRegistry port implementations."""

    @pytest.mark.asyncio
    async def test_register_and_get_entity(self, registry: EntityRegistry) -> None:
        entity = Entity(
            entity_id=EntityId("loki"),
            series_id=SeriesId("loki"),
            name="Loki Laufeyson",
            entity_type="character",
            aliases=["God of Mischief", "Variant L1130"],
            summary="God of Mischief.",
            related_entities=[EntityId("mobius"), EntityId("sylvie")],
            metadata={"status": "alive"},
        )
        await registry.register_entity(entity)

        retrieved = await registry.get_entity(SeriesId("loki"), EntityId("loki"))
        assert retrieved is not None
        assert retrieved.entity_id == EntityId("loki")
        assert retrieved.name == "Loki Laufeyson"
        assert retrieved.aliases == ["God of Mischief", "Variant L1130"]
        assert retrieved.summary == "God of Mischief."
        assert retrieved.related_entities == [EntityId("mobius"), EntityId("sylvie")]
        assert retrieved.metadata == {"status": "alive"}

        # Non-existent
        assert (
            await registry.get_entity(SeriesId("loki"), EntityId("nonexistent")) is None
        )

    @pytest.mark.asyncio
    async def test_list_entities(self, registry: EntityRegistry) -> None:
        e1 = Entity(
            entity_id=EntityId("loki"),
            series_id=SeriesId("loki"),
            name="Loki",
            entity_type="character",
        )
        e2 = Entity(
            entity_id=EntityId("mobius"),
            series_id=SeriesId("loki"),
            name="Mobius M. Mobius",
            entity_type="character",
        )
        await registry.register_entities([e1, e2])

        entities = await registry.list_entities(SeriesId("loki"))
        entity_ids = {e.entity_id for e in entities}
        assert EntityId("loki") in entity_ids
        assert EntityId("mobius") in entity_ids

    @pytest.mark.asyncio
    async def test_resolve_entity(self, registry: EntityRegistry) -> None:
        entity = Entity(
            entity_id=EntityId("he-who-remains"),
            series_id=SeriesId("loki"),
            name="He Who Remains",
            entity_type="character",
            aliases=["Kang Variant", "Creator of TVA"],
        )
        await registry.register_entity(entity)

        # By canonical slug ID
        r1 = await registry.resolve_entity(SeriesId("loki"), "he-who-remains")
        assert r1 is not None and r1.entity_id == "he-who-remains"

        # By name (case-insensitive)
        r2 = await registry.resolve_entity(SeriesId("loki"), "he who remains")
        assert r2 is not None and r2.entity_id == "he-who-remains"

        # By alias (case-insensitive)
        r3 = await registry.resolve_entity(SeriesId("loki"), "kang variant")
        assert r3 is not None and r3.entity_id == "he-who-remains"

        # Non-existent
        assert await registry.resolve_entity(SeriesId("loki"), "Thor") is None
