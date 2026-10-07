"""VectorIndex port interface."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.search import SearchHit, SearchQuery
from lore_mcp.domain.types import ChunkId, SeriesId


@runtime_checkable
class VectorIndex(Protocol):
    """Port for indexing and querying chunks in a vector store."""

    async def upsert(self, chunks: Sequence[Chunk]) -> None:
        """Insert or update chunks idempotently."""
        ...

    async def search(self, query: SearchQuery) -> list[SearchHit]:
        """Search for candidate chunks matching constraints."""
        ...

    async def get_by_id(self, chunk_id: ChunkId) -> Chunk | None:
        """Retrieve a specific chunk by its stable identifier."""
        ...

    async def delete_series(self, series_id: SeriesId) -> None:
        """Delete all indexed chunks for a series."""
        ...

    async def count(self, series_id: SeriesId | None = None) -> int:
        """Return total chunks count, optionally scoped to a series."""
        ...

    async def prune_older_versions(
        self, series_id: SeriesId, active_version: int
    ) -> None:
        """Prune chunks belonging to series_id with ingest_version < active_version."""
        ...
