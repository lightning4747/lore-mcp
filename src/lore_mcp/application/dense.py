"""Dense vector retrieval service for semantic lore search."""

from lore_mcp.application.errors import InvalidArgument
from lore_mcp.domain.search import SearchHit, SearchQuery
from lore_mcp.ports.vector_index import VectorIndex


class DenseRetriever:
    """Service that executes dense vector semantic retrieval locked to a series."""

    def __init__(self, index: VectorIndex) -> None:
        self._index = index

    async def retrieve(self, query: SearchQuery) -> list[SearchHit]:
        """Execute dense retrieval returning ranked candidates with cosine scores.

        Raises:
            InvalidArgument: If query text is empty or whitespace.
        """
        if not query.query.strip():
            raise InvalidArgument(
                "Query text must not be empty or whitespace.",
                argument_name="query",
            )

        hits = await self._index.search_dense(query)

        # Ensure candidates are ranked in descending order by score
        return sorted(hits, key=lambda h: h.score, reverse=True)
