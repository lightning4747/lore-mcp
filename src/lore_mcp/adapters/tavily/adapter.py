"""Tavily search provider adapter implementing WebSearch port (Task 13.1)."""

import logging
from datetime import UTC, datetime
from typing import Any

from tavily import AsyncTavilyClient

from lore_mcp.domain.search import WebSearchResult
from lore_mcp.ports.web_search import WebSearch

logger = logging.getLogger(__name__)


class WebSearchError(Exception):
    """Base exception for web search provider failures."""


class TavilyWebSearch(WebSearch):
    """Tavily search API implementation of WebSearch port."""

    def __init__(
        self,
        api_key: str,
        timeout: float = 10.0,
        client: AsyncTavilyClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._client = client or AsyncTavilyClient(api_key=api_key)

    async def search(self, query: str, limit: int = 5) -> list[WebSearchResult]:
        """Perform external web search query using Tavily basic search depth (1 credit).

        Returns normalized WebSearchResult models with URL, title, snippet,
        and UTC retrieval timestamp.
        """
        stripped_query = query.strip()
        if not stripped_query:
            return []

        try:
            response: dict[str, Any] = await self._client.search(
                query=stripped_query,
                search_depth="basic",
                max_results=limit,
                timeout=self._timeout,
            )
        except Exception as err:
            logger.warning(
                "Tavily search failed for query '%s': %s",
                stripped_query,
                err,
            )
            raise WebSearchError(f"Tavily search request failed: {err}") from err

        raw_results = response.get("results", [])
        if not isinstance(raw_results, list):
            return []

        now = datetime.now(UTC)
        results: list[WebSearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            if not url:
                continue
            title = str(item.get("title", "")).strip()
            snippet = str(item.get("content", "")).strip()
            results.append(
                WebSearchResult(
                    url=url,
                    title=title,
                    snippet=snippet,
                    retrieval_timestamp=now,
                )
            )
            if len(results) >= limit:
                break

        return results

    async def close(self) -> None:
        """Close underlying Tavily client resources."""
        if hasattr(self._client, "close"):
            await self._client.close()
