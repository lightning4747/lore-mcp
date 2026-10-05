"""WebSearch port interface."""

from typing import Protocol, runtime_checkable

from lore_mcp.domain.search import WebSearchResult


@runtime_checkable
class WebSearch(Protocol):
    """Port for external web search fallback queries."""

    async def search(self, query: str, limit: int = 5) -> list[WebSearchResult]:
        """Perform a web search query and return normalized results."""
        ...
