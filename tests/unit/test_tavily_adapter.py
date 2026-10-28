"""Unit and contract tests for TavilyWebSearch adapter (Task 13.1)."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from lore_mcp.adapters.tavily import TavilyWebSearch, WebSearchError
from lore_mcp.domain.search import WebSearchResult
from lore_mcp.ports.web_search import WebSearch
from lore_mcp.testing.contracts import WebSearchContractSuite


class MockAsyncTavilyClient:
    """Mock Tavily async client for unit testing."""

    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.search = AsyncMock()
        if response is not None:
            self.search.return_value = response
        else:
            self.search.return_value = {
                "results": [
                    {
                        "url": "https://marvel.fandom.com/wiki/Loki",
                        "title": "Loki (Earth-616) | Marvel Database",
                        "content": "Loki Laufeyson is the God of Mischief.",
                        "score": 0.98,
                    },
                    {
                        "url": "https://marvelcinematicuniverse.fandom.com/wiki/Loki",
                        "title": "Loki | Marvel Cinematic Universe Wiki",
                        "content": "Loki is the Prince of Asgard and God of Mischief.",
                        "score": 0.95,
                    },
                ]
            }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tavily_search_success() -> None:
    mock_client = MockAsyncTavilyClient()
    adapter = TavilyWebSearch(
        api_key="tvly-mock-key",
        timeout=5.0,
        client=mock_client,  # type: ignore[arg-type]
    )

    results = await adapter.search("Loki variants", limit=2)

    assert len(results) == 2
    mock_client.search.assert_awaited_once_with(
        query="Loki variants",
        search_depth="basic",
        max_results=2,
        timeout=5.0,
    )
    first = results[0]
    assert isinstance(first, WebSearchResult)
    assert first.url == "https://marvel.fandom.com/wiki/Loki"
    assert first.title == "Loki (Earth-616) | Marvel Database"
    assert "God of Mischief" in first.snippet
    assert first.retrieval_timestamp is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tavily_search_empty_query() -> None:
    mock_client = MockAsyncTavilyClient()
    adapter = TavilyWebSearch(
        api_key="tvly-mock-key",
        client=mock_client,  # type: ignore[arg-type]
    )

    results = await adapter.search("   ", limit=5)
    assert results == []
    mock_client.search.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tavily_search_error_handling() -> None:
    mock_client = MockAsyncTavilyClient()
    mock_client.search.side_effect = RuntimeError("Network timeout or connection reset")
    adapter = TavilyWebSearch(
        api_key="tvly-mock-key",
        client=mock_client,  # type: ignore[arg-type]
    )

    with pytest.raises(WebSearchError) as exc_info:
        await adapter.search("Loki", limit=3)

    assert "Tavily search request failed" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.contract
class TestTavilyWebSearchContract(WebSearchContractSuite):
    @pytest.fixture
    def search(self) -> WebSearch:
        mock_client = MockAsyncTavilyClient()
        return TavilyWebSearch(
            api_key="tvly-mock-key",
            client=mock_client,  # type: ignore[arg-type]
        )
