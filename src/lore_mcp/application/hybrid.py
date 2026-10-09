"""Hybrid retrieval coordinator for parallel dense and sparse lore search."""

import asyncio

from lore_mcp.application.dense import DenseRetriever
from lore_mcp.application.errors import InvalidArgument
from lore_mcp.application.sparse import SparseRetriever
from lore_mcp.domain.search import (
    SearchHit,
    SearchQuery,
    reciprocal_rank_fusion,
    rerank_candidates,
)
from lore_mcp.ports.vector_index import VectorIndex


class HybridRetriever:
    """Coordinates parallel dense semantic and sparse lexical vector retrieval."""

    def __init__(
        self,
        index: VectorIndex,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        *,
        rrf_k: int = 60,
        enable_rerank: bool = True,
    ) -> None:
        self._index = index
        self._dense = dense_retriever or DenseRetriever(index)
        self._sparse = sparse_retriever or SparseRetriever(index)
        self._rrf_k = rrf_k
        self._enable_rerank = enable_rerank

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

    async def retrieve(self, query: SearchQuery) -> list[SearchHit]:
        """Execute parallel hybrid retrieval, fuse candidates with
        Reciprocal Rank Fusion [S3], and apply heuristic reranking.

        Removes duplicate chunks, preserves per-signal scores, and returns
        ranked candidates sorted descending by relevance and reranking score.
        """
        dense_hits, sparse_hits = await self.retrieve_parallel(query)
        fused = reciprocal_rank_fusion(
            [dense_hits, sparse_hits],
            k=self._rrf_k,
            max_results=None if self._enable_rerank else query.max_results,
        )
        if self._enable_rerank:
            return rerank_candidates(fused, query, max_results=query.max_results)
        return fused
