"""Hybrid retrieval coordinator for parallel dense and sparse lore search."""

import asyncio

from lore_mcp.application.dense import DenseRetriever
from lore_mcp.application.errors import InvalidArgument
from lore_mcp.application.sparse import SparseRetriever
from lore_mcp.domain.search import SearchHit, SearchQuery
from lore_mcp.ports.vector_index import VectorIndex


class HybridRetriever:
    """Coordinates parallel dense semantic and sparse lexical vector retrieval."""

    def __init__(
        self,
        index: VectorIndex,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
    ) -> None:
        self._index = index
        self._dense = dense_retriever or DenseRetriever(index)
        self._sparse = sparse_retriever or SparseRetriever(index)

    async def retrieve_parallel(
        self, query: SearchQuery
    ) -> tuple[list[SearchHit], list[SearchHit]]:
        """Run dense and sparse queries in parallel via asyncio.gather.

        Raises:
            InvalidArgument: If query text is empty or whitespace.
        """
        if not query.query.strip():
            raise InvalidArgument(
                "Query text must not be empty or whitespace.",
                argument_name="query",
            )

        dense_hits, sparse_hits = await asyncio.gather(
            self._dense.retrieve(query),
            self._sparse.retrieve(query),
        )
        return dense_hits, sparse_hits
