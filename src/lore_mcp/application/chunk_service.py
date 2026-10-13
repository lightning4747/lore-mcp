"""Application service for retrieving individual chunks with scope validation [S11]."""

import logging

from lore_mcp.application.errors import DataUnavailable, InvalidArgument
from lore_mcp.application.scope import ScopeResolver
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.types import ChunkId
from lore_mcp.ports.vector_index import VectorIndex

logger = logging.getLogger(__name__)


class ChunkService:
    """Orchestrates single-chunk retrieval by ID with post-fetch series validation."""

    def __init__(
        self,
        scope_resolver: ScopeResolver,
        vector_index: VectorIndex,
    ) -> None:
        self._scope_resolver = scope_resolver
        self._vector_index = vector_index

    async def get_chunk(
        self,
        chunk_id: str,
        series: str,
    ) -> Chunk:
        """Fetch a full chunk by ID to expand truncated evidence.

        Re-checks the chunk's series after fetching because direct ID retrieval
        in vector stores cannot apply tenant payload filters [S11].
        """
        cleaned_chunk_id = chunk_id.strip()
        if not cleaned_chunk_id:
            raise InvalidArgument(
                "chunk_id must not be empty or whitespace.",
                argument_name="chunk_id",
            )

        cleaned_series = series.strip()
        if not cleaned_series:
            raise InvalidArgument(
                "series must not be empty or whitespace.",
                argument_name="series",
            )

        # 1. Resolve series scope
        resolved_scope = await self._scope_resolver.resolve(cleaned_series)
        expected_series_id = resolved_scope.series_id

        # 2. Fetch chunk by ID
        chunk = await self._vector_index.get_by_id(ChunkId(cleaned_chunk_id))
        if chunk is None:
            raise DataUnavailable(
                f"Chunk '{cleaned_chunk_id}' not found.",
                resource_id=cleaned_chunk_id,
            )

        # 3. Re-check the chunk's series after fetching [S11]
        if chunk.series_id != expected_series_id:
            logger.warning(
                "Cross-series chunk access blocked: chunk '%s' belongs to series "
                "'%s', not requested series '%s'",
                cleaned_chunk_id,
                chunk.series_id,
                expected_series_id,
            )
            raise DataUnavailable(
                f"Chunk '{cleaned_chunk_id}' not found in "
                f"series '{expected_series_id}'.",
                resource_id=cleaned_chunk_id,
            )

        return chunk
