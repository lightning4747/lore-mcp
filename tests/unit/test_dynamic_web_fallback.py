"""Unit tests for dynamic web fallback logic, filtering,
and resilience (Tasks 13.2-13.4).
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from lore_mcp.adapters.redis import CachedWebSearch
from lore_mcp.adapters.tavily import WebSearchError
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.application.retrieval_service import (
    LoreRetrievalService,
    canonicalize_url,
    is_domain_allowed,
    is_series_relevant,
)
from lore_mcp.application.scope import ScopeResolver
from lore_mcp.config import WebSearchSettings
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.entities import SeriesManifest
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import WebSearchResult
from lore_mcp.domain.types import ChunkId, SeriesId
from lore_mcp.testing.fakes import (
    FakeBudgetCounter,
    FakeCache,
    FakeClock,
    FakeSeriesRegistry,
    FakeVectorIndex,
    FakeWebSearch,
)


@pytest.fixture
def manifest() -> SeriesManifest:
    return SeriesManifest(
        series_id=SeriesId("loki"),
        display_name="Loki (TV Series)",
        kind="show",
        aliases=["MCU Loki", "God of Stories"],
        seasons=[1, 2],
        web_fallback_domain_allowlist=[
            "marvel.fandom.com",
            "en.wikipedia.org",
            "marvelcinematicuniverse.fandom.com",
        ],
    )


@pytest.fixture
def fake_registry(manifest: SeriesManifest) -> FakeSeriesRegistry:
    registry = FakeSeriesRegistry()
    registry._manifests[SeriesId("loki")] = manifest
    return registry


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock(start_time=datetime(2026, 10, 28, 12, 0, 0, tzinfo=UTC))


class TestFallbackHelpers:
    """Tests for URL canonicalization, domain allowlisting, and series relevance."""

    def test_canonicalize_url(self) -> None:
        assert (
            canonicalize_url("https://marvel.fandom.com/wiki/Loki/")
            == "https://marvel.fandom.com/wiki/Loki"
        )
        assert (
            canonicalize_url("HTTPS://Marvel.Fandom.COM:443/wiki/Loki")
            == "https://marvel.fandom.com/wiki/Loki"
        )
        assert (
            canonicalize_url("http://example.com:80/path?query=val")
            == "http://example.com/path?query=val"
        )

    def test_is_domain_allowed_empty_allowlist(self) -> None:
        assert is_domain_allowed("https://anything.com/page", []) is True

    def test_is_domain_allowed_filtering(self) -> None:
        allowlist = ["marvel.fandom.com", "en.wikipedia.org"]
        assert (
            is_domain_allowed("https://marvel.fandom.com/wiki/Loki", allowlist) is True
        )
        assert (
            is_domain_allowed("https://sub.marvel.fandom.com/wiki/Loki", allowlist)
            is True
        )
        assert (
            is_domain_allowed("https://en.wikipedia.org/wiki/Loki", allowlist) is True
        )
        # Disallowed domains
        assert is_domain_allowed("https://random.com/wiki/Loki", allowlist) is False
        assert (
            is_domain_allowed("https://evil-marvel.fandom.com.fake/test", allowlist)
            is False
        )

    def test_is_series_relevant(self, manifest: SeriesManifest) -> None:
        # Matches display name token "loki"
        assert (
            is_series_relevant("Loki in the Multiverse", "Story summary", manifest)
            is True
        )
        # Matches alias "god of stories"
        assert (
            is_series_relevant(
                "Season Finale Review",
                "The God of Stories crowns the timeline",
                manifest,
            )
            is True
        )
        # Matches series id "loki"
        assert (
            is_series_relevant("He Who Remains", "Citadel seen in loki s1", manifest)
            is True
        )
        # Irrelevant result
        assert (
            is_series_relevant(
                "Baking Chocolate Chip Cookies",
                "Delicious recipe for easy cookies at home",
                manifest,
            )
            is False
        )


class TestLoreRetrievalServiceFallback:
    """Tests for fallback decision, provider resilience, and budget enforcement."""

    @pytest.mark.asyncio
    async def test_sufficient_evidence_skips_web_search(
        self,
        manifest: SeriesManifest,
        fake_registry: FakeSeriesRegistry,
        fake_clock: FakeClock,
    ) -> None:
        index = FakeVectorIndex()
        chunk = Chunk(
            chunk_id=ChunkId("chunk-1"),
            series_id=SeriesId("loki"),
            text="Loki Laufeyson was taken to the TVA.",
            season=1,
            episode=1,
            scene_index=1,
            content_hash="abc123hash",
            provenance=Provenance(
                source_type="transcript",
                source_url="https://transcripts.org/1",
                retrieval_timestamp=fake_clock.now(),
            ),
        )
        await index.upsert([chunk])

        web_search = FakeWebSearch()
        web_search.set_default_results(
            [
                WebSearchResult(
                    url="https://marvel.fandom.com/wiki/Loki",
                    title="Loki Info",
                    snippet="TVA Loki",
                )
            ]
        )
        scope = ScopeResolver(registry=fake_registry, clock=fake_clock)
        retriever = HybridRetriever(index=index)
        service = LoreRetrievalService(
            scope_resolver=scope,
            retriever=retriever,
            web_search=web_search,
            clock=fake_clock,
            sufficiency_threshold=0.01,
        )

        bundle = await service.search(query="TVA", series="loki")
        assert bundle.status == "ok"
        assert len(bundle.evidence) == 1
        assert bundle.evidence[0].chunk_id == "chunk-1"
        assert bundle.status != "fallback_used"

    @pytest.mark.asyncio
    async def test_insufficient_evidence_triggers_fallback_and_filters(
        self,
        manifest: SeriesManifest,
        fake_registry: FakeSeriesRegistry,
        fake_clock: FakeClock,
    ) -> None:
        empty_index = FakeVectorIndex()
        web_search = FakeWebSearch()
        web_search.set_default_results(
            [
                # Allowed and relevant
                WebSearchResult(
                    url="https://marvel.fandom.com/wiki/Citadel_at_the_End_of_Time",
                    title="Citadel at the End of Time (Loki)",
                    snippet="He Who Remains lives at the Citadel in the Loki series.",
                ),
                # Duplicate URL (trailing slash)
                WebSearchResult(
                    url="https://marvel.fandom.com/wiki/Citadel_at_the_End_of_Time/",
                    title="Citadel at the End of Time (Loki) Duplicate",
                    snippet="Duplicate result text.",
                ),
                # Disallowed domain
                WebSearchResult(
                    url="https://unauthorized-blog.com/citadel",
                    title="Loki Citadel Review",
                    snippet="Review of Loki episode 6.",
                ),
                # Allowed domain but irrelevant to Loki
                WebSearchResult(
                    url="https://en.wikipedia.org/wiki/Baking_powder",
                    title="Baking powder",
                    snippet="Baking powder is a dry chemical leavening agent.",
                ),
            ]
        )

        scope = ScopeResolver(registry=fake_registry, clock=fake_clock)
        retriever = HybridRetriever(index=empty_index)
        service = LoreRetrievalService(
            scope_resolver=scope,
            retriever=retriever,
            web_search=web_search,
            clock=fake_clock,
        )

        bundle = await service.search(query="Citadel at the End of Time", series="loki")
        assert bundle.status == "fallback_used"
        assert len(bundle.evidence) == 1
        assert bundle.evidence[0].provenance.source_url == (
            "https://marvel.fandom.com/wiki/Citadel_at_the_End_of_Time"
        )
        assert bundle.evidence[0].provenance.source_type == "web"
        assert "Retrieved from external web search fallback" in bundle.notices

    @pytest.mark.asyncio
    async def test_budget_exhaustion_returns_insufficient_status(
        self,
        manifest: SeriesManifest,
        fake_registry: FakeSeriesRegistry,
        fake_clock: FakeClock,
    ) -> None:
        empty_index = FakeVectorIndex()
        web_search = FakeWebSearch()
        web_search.set_default_results(
            [
                WebSearchResult(
                    url="https://marvel.fandom.com/wiki/Loki",
                    title="Loki Lore",
                    snippet="Loki series details",
                )
            ]
        )
        budget = FakeBudgetCounter()
        budget_key = "lore:v1:budget:web_search_daily:2026-10-28"
        for _ in range(5):
            await budget.increment(budget_key, 86400)

        scope = ScopeResolver(registry=fake_registry, clock=fake_clock)
        retriever = HybridRetriever(index=empty_index)
        settings = WebSearchSettings(daily_budget=5)
        service = LoreRetrievalService(
            scope_resolver=scope,
            retriever=retriever,
            web_search=web_search,
            budget_counter=budget,
            web_search_settings=settings,
            clock=fake_clock,
        )

        bundle = await service.search(query="He Who Remains", series="loki")
        assert bundle.status == "insufficient"
        assert len(bundle.evidence) == 0
        assert "Web search budget exhausted" in bundle.notices

    @pytest.mark.asyncio
    async def test_budget_counter_failure_fails_closed(
        self,
        manifest: SeriesManifest,
        fake_registry: FakeSeriesRegistry,
        fake_clock: FakeClock,
    ) -> None:
        empty_index = FakeVectorIndex()
        web_search = FakeWebSearch()
        web_search.set_default_results(
            [
                WebSearchResult(
                    url="https://marvel.fandom.com/wiki/Loki",
                    title="Loki Lore",
                    snippet="Loki series details",
                )
            ]
        )
        mock_budget = AsyncMock()
        mock_budget.check_budget.side_effect = RuntimeError("Redis connection lost")

        scope = ScopeResolver(registry=fake_registry, clock=fake_clock)
        retriever = HybridRetriever(index=empty_index)
        service = LoreRetrievalService(
            scope_resolver=scope,
            retriever=retriever,
            web_search=web_search,
            budget_counter=mock_budget,
            clock=fake_clock,
        )

        bundle = await service.search(query="He Who Remains", series="loki")
        assert bundle.status == "insufficient"
        assert len(bundle.evidence) == 0
        assert "Web search budget counter unavailable" in bundle.notices

    @pytest.mark.asyncio
    async def test_provider_error_graceful_degradation(
        self,
        manifest: SeriesManifest,
        fake_registry: FakeSeriesRegistry,
        fake_clock: FakeClock,
    ) -> None:
        empty_index = FakeVectorIndex()
        mock_web = AsyncMock()
        mock_web.search.side_effect = WebSearchError("Tavily 503 Service Unavailable")

        scope = ScopeResolver(registry=fake_registry, clock=fake_clock)
        retriever = HybridRetriever(index=empty_index)
        service = LoreRetrievalService(
            scope_resolver=scope,
            retriever=retriever,
            web_search=mock_web,
            clock=fake_clock,
        )

        bundle = await service.search(query="He Who Remains", series="loki")
        assert bundle.status == "insufficient"
        assert len(bundle.evidence) == 0
        assert any("Web search provider error" in notice for notice in bundle.notices)

    @pytest.mark.asyncio
    async def test_fallback_with_web_cache_avoids_repeated_provider_call(
        self,
        manifest: SeriesManifest,
        fake_registry: FakeSeriesRegistry,
        fake_clock: FakeClock,
    ) -> None:
        empty_index = FakeVectorIndex()
        mock_web = AsyncMock()
        mock_web.search.return_value = [
            WebSearchResult(
                url="https://marvel.fandom.com/wiki/Citadel_at_the_End_of_Time",
                title="Citadel at the End of Time (Loki)",
                snippet="He Who Remains lives at the Citadel in the Loki series.",
            )
        ]
        fake_cache = FakeCache()
        cached_web = CachedWebSearch(inner=mock_web, cache=fake_cache)

        scope = ScopeResolver(registry=fake_registry, clock=fake_clock)
        retriever = HybridRetriever(index=empty_index)
        service = LoreRetrievalService(
            scope_resolver=scope,
            retriever=retriever,
            web_search=cached_web,
            clock=fake_clock,
        )

        bundle1 = await service.search(query="Citadel", series="loki")
        assert bundle1.status == "fallback_used"
        assert mock_web.search.await_count == 1

        # Second identical fallback call should hit web cache and NOT invoke mock_web
        bundle2 = await service.search(query="Citadel", series="loki")
        assert bundle2.status == "fallback_used"
        assert mock_web.search.await_count == 1
        assert len(bundle2.evidence) == 1
        assert (
            bundle2.evidence[0].provenance.source_url
            == "https://marvel.fandom.com/wiki/Citadel_at_the_End_of_Time"
        )
