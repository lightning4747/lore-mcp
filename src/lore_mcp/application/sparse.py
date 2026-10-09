"""Sparse BM25 vector retrieval service for lexical lore search."""

from lore_mcp.application.errors import InvalidArgument
from lore_mcp.domain.search import SearchHit, SearchQuery
from lore_mcp.ports.vector_index import VectorIndex


class SparseRetriever:
    """Service that executes BM25 sparse vector lexical retrieval locked to a series."""

    def __init__(self, index: VectorIndex) -> None:
        self._index = index

    async def retrieve(self, query: SearchQuery) -> list[SearchHit]:
        """Execute sparse BM25 retrieval returning lexical candidates with scores.

        Raises:
            InvalidArgument: If query text is empty or whitespace.
        """
        if not query.query.strip():
            raise InvalidArgument(
                "Query text must not be empty or whitespace.",
                argument_name="query",
            )

        hits = await self._index.search_sparse(query)

        # Ensure candidates are ranked in descending order by score
        return sorted(hits, key=lambda h: h.score, reverse=True)
