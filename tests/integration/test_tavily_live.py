"""Live manual and integration tests against real Tavily search API (Task 13.5).

Requires a valid TAVILY_API_KEY in the environment or .env file.
Marked with @pytest.mark.live.
"""

from datetime import UTC, datetime

import pytest

from lore_mcp.adapters.tavily import TavilyWebSearch
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.application.retrieval_service import LoreRetrievalService
from lore_mcp.application.scope import ScopeResolver
from lore_mcp.config import get_settings
from lore_mcp.domain.entities import SeriesManifest
from lore_mcp.domain.search import WebSearchResult
from lore_mcp.domain.types import SeriesId
from lore_mcp.testing.fakes import (
    FakeBudgetCounter,
    FakeClock,
    FakeSeriesRegistry,
    FakeVectorIndex,
)


@pytest.fixture
def tavily_api_key() -> str:
    settings = get_settings()
    if (
        settings.web_search.api_key is None
        or not settings.web_search.api_key.get_secret_value().strip()
    ):
        pytest.skip("TAVILY_API_KEY is not configured in environment or .env")
    return settings.web_search.api_key.get_secret_value().strip()


@pytest.mark.live
@pytest.mark.integration
@pytest.mark.asyncio
async def test_tavily_live_search(tavily_api_key: str) -> None:
    """Verify live search query to Tavily API using basic search depth."""
    search_adapter = TavilyWebSearch(
        api_key=tavily_api_key,
        timeout=15.0,
    )
    try:
        results = await search_adapter.search(
            query="Loki TVA Time Variance Authority",
            limit=2,
        )
        assert len(results) > 0
        assert len(results) <= 2

        first = results[0]
        assert isinstance(first, WebSearchResult)
        assert first.url.startswith("http")
        assert len(first.title) > 0
        assert len(first.snippet) > 0
        assert first.retrieval_timestamp is not None
    finally:
        await search_adapter.close()


@pytest.mark.live
@pytest.mark.integration
@pytest.mark.asyncio
async def test_tavily_live_retrieval_fallback(tavily_api_key: str) -> None:
    """Verify end-to-end fallback search with live Tavily adapter."""
    manifest = SeriesManifest(
        series_id=SeriesId("loki"),
        display_name="Loki (TV Series)",
        kind="show",
        aliases=["MCU Loki", "Marvel Loki"],
        seasons=[1, 2],
        web_fallback_domain_allowlist=[
            "marvel.fandom.com",
            "en.wikipedia.org",
            "marvelcinematicuniverse.fandom.com",
            "marvel.com",
        ],
    )
    registry = FakeSeriesRegistry()
    registry._manifests[SeriesId("loki")] = manifest

    clock = FakeClock(start_time=datetime(2026, 10, 28, 12, 0, 0, tzinfo=UTC))
    scope = ScopeResolver(registry=registry, clock=clock)
    empty_index = FakeVectorIndex()
    retriever = HybridRetriever(index=empty_index)

    search_adapter = TavilyWebSearch(
        api_key=tavily_api_key,
        timeout=15.0,
    )
    budget = FakeBudgetCounter()

    service = LoreRetrievalService(
        scope_resolver=scope,
        retriever=retriever,
        web_search=search_adapter,
        budget_counter=budget,
        clock=clock,
    )

    try:
        bundle = await service.search(
            query="Citadel at the End of Time He Who Remains",
            series="loki",
        )
        assert bundle.status == "fallback_used"
        assert len(bundle.evidence) > 0
        first_ev = bundle.evidence[0]
        assert first_ev.provenance.source_type == "web"
        assert first_ev.provenance.source_url.startswith("http")
        assert "Retrieved from external web search fallback" in bundle.notices
    finally:
        await search_adapter.close()
