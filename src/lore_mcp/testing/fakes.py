"""In-memory fakes implementing port protocols for testing and development."""

import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from lore_mcp.domain.documents import Chunk, NormalizedDocument, RawDocument
from lore_mcp.domain.entities import SeriesManifest
from lore_mcp.domain.search import SearchHit, SearchQuery, WebSearchResult
from lore_mcp.domain.types import ChunkId, SeriesId
from lore_mcp.ports.budget import BudgetCounter
from lore_mcp.ports.cache import Cache
from lore_mcp.ports.clock import Clock
from lore_mcp.ports.documents import DocumentSource, DocumentStore
from lore_mcp.ports.series_registry import SeriesRegistry
from lore_mcp.ports.vector_index import VectorIndex
from lore_mcp.ports.web_search import WebSearch


class FakeVectorIndex(VectorIndex):
    """In-memory fake implementation of VectorIndex."""

    def __init__(self) -> None:
        self._chunks: dict[ChunkId, Chunk] = {}

    async def upsert(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    async def search(self, query: SearchQuery) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for chunk in self._chunks.values():
            if chunk.series_id != query.series_id:
                continue
            if query.season is not None and chunk.season != query.season:
                continue
            if query.episode is not None and chunk.episode != query.episode:
                continue
            if query.max_season is not None and chunk.season is not None:
                if chunk.season > query.max_season:
                    continue
            if query.max_episode is not None and chunk.episode is not None:
                if (
                    query.season is not None
                    and chunk.season == query.season
                    and chunk.episode > query.max_episode
                ):
                    continue

            # Basic relevance calculation for in-memory testing
            text_lower = chunk.text.lower()
            query_lower = query.query.lower()
            score = 1.0 if query_lower in text_lower else 0.5

            hits.append(
                SearchHit(
                    chunk=chunk,
                    score=score,
                    dense_score=score,
                    sparse_score=score,
                    rerank_score=score,
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: query.max_results]

    async def get_by_id(self, chunk_id: ChunkId) -> Chunk | None:
        return self._chunks.get(chunk_id)

    async def delete_series(self, series_id: SeriesId) -> None:
        to_delete = [cid for cid, c in self._chunks.items() if c.series_id == series_id]
        for cid in to_delete:
            del self._chunks[cid]

    async def count(self, series_id: SeriesId | None = None) -> int:
        if series_id is None:
            return len(self._chunks)
        return sum(1 for c in self._chunks.values() if c.series_id == series_id)

    async def prune_older_versions(
        self, series_id: SeriesId, active_version: int
    ) -> None:
        to_delete = [
            cid
            for cid, c in self._chunks.items()
            if c.series_id == series_id and c.ingest_version < active_version
        ]
        for cid in to_delete:
            del self._chunks[cid]


class FakeSeriesRegistry(SeriesRegistry):
    """In-memory fake implementation of SeriesRegistry."""

    def __init__(self) -> None:
        self._manifests: dict[SeriesId, SeriesManifest] = {}

    async def register_series(self, manifest: SeriesManifest) -> None:
        self._manifests[manifest.series_id] = manifest

    async def get_series(self, series_id: SeriesId) -> SeriesManifest | None:
        return self._manifests.get(series_id)

    async def list_series(self) -> list[SeriesId]:
        return list(self._manifests.keys())

    async def delete_series(self, series_id: SeriesId) -> bool:
        if series_id in self._manifests:
            del self._manifests[series_id]
            return True
        return False

    async def get_alias_map(self) -> dict[str, SeriesId]:
        alias_map: dict[str, SeriesId] = {}
        for s_id, manifest in self._manifests.items():
            alias_map[str(s_id).lower()] = s_id
            alias_map[manifest.display_name.lower()] = s_id
            for a in manifest.aliases:
                alias_map[a.strip().lower()] = s_id
        return alias_map

    async def resolve_alias(self, alias_or_name: str) -> SeriesId | None:
        alias_map = await self.get_alias_map()
        return alias_map.get(alias_or_name.strip().lower())


class FakeCache(Cache):
    """In-memory fake implementation of Cache with TTL support."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._expires_at: dict[str, float] = {}

    def _is_expired(self, key: str) -> bool:
        if key in self._expires_at:
            if time.time() >= self._expires_at[key]:
                del self._store[key]
                del self._expires_at[key]
                return True
        return False

    async def get(self, key: str) -> str | None:
        if self._is_expired(key):
            return None
        return self._store.get(key)

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        self._store[key] = value
        if ttl_seconds is not None:
            self._expires_at[key] = time.time() + ttl_seconds
        elif key in self._expires_at:
            del self._expires_at[key]

    async def delete(self, key: str) -> bool:
        existed = key in self._store
        self._store.pop(key, None)
        self._expires_at.pop(key, None)
        return existed

    async def exists(self, key: str) -> bool:
        if self._is_expired(key):
            return False
        return key in self._store


class FakeBudgetCounter(BudgetCounter):
    """In-memory fake implementation of BudgetCounter with window expiration."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._expires_at: dict[str, float] = {}

    def _check_expiry(self, key: str) -> None:
        if key in self._expires_at and time.time() >= self._expires_at[key]:
            self._counts.pop(key, None)
            self._expires_at.pop(key, None)

    async def increment(self, key: str, window_seconds: int) -> int:
        self._check_expiry(key)
        new_count = self._counts.get(key, 0) + 1
        self._counts[key] = new_count
        if key not in self._expires_at:
            self._expires_at[key] = time.time() + window_seconds
        return new_count

    async def get_count(self, key: str) -> int:
        self._check_expiry(key)
        return self._counts.get(key, 0)

    async def check_budget(self, key: str, limit: int) -> bool:
        count = await self.get_count(key)
        return count < limit

    async def reset(self, key: str) -> None:
        self._counts.pop(key, None)
        self._expires_at.pop(key, None)


class FakeWebSearch(WebSearch):
    """In-memory fake implementation of WebSearch fallback."""

    def __init__(self) -> None:
        self._canned_results: dict[str, list[WebSearchResult]] = {}
        self._default_results: list[WebSearchResult] = []

    def set_results(self, query: str, results: list[WebSearchResult]) -> None:
        self._canned_results[query] = results

    def set_default_results(self, results: list[WebSearchResult]) -> None:
        self._default_results = results

    async def search(self, query: str, limit: int = 5) -> list[WebSearchResult]:
        results = self._canned_results.get(query, self._default_results)
        return results[:limit]


class FakeDocumentSource(DocumentSource):
    """In-memory fake implementation of DocumentSource."""

    def __init__(self) -> None:
        self._documents: dict[tuple[SeriesId, str], RawDocument] = {}

    def add_document(self, doc: RawDocument) -> None:
        self._documents[(doc.series_id, doc.source_url)] = doc

    async def fetch_document(self, series_id: SeriesId, locator: str) -> RawDocument:
        key = (series_id, locator)
        if key not in self._documents:
            raise KeyError(f"Document not found for series {series_id} at {locator}")
        return self._documents[key]

    async def list_locators(self, series_id: SeriesId) -> list[str]:
        return [loc for s_id, loc in self._documents if s_id == series_id]


class FakeDocumentStore(DocumentStore):
    """In-memory fake implementation of DocumentStore."""

    def __init__(self) -> None:
        self._raw_docs: dict[tuple[SeriesId, str], RawDocument] = {}
        self._norm_docs: dict[tuple[SeriesId, str], NormalizedDocument] = {}

    async def put_raw(self, doc: RawDocument) -> None:
        self._raw_docs[(doc.series_id, doc.source_url)] = doc

    async def get_raw(self, series_id: SeriesId, source_url: str) -> RawDocument | None:
        return self._raw_docs.get((series_id, source_url))

    async def put_normalized(self, doc: NormalizedDocument) -> None:
        self._norm_docs[(doc.series_id, doc.source_url)] = doc

    async def get_normalized(
        self, series_id: SeriesId, source_url: str
    ) -> NormalizedDocument | None:
        return self._norm_docs.get((series_id, source_url))

    async def list_urls(self, series_id: SeriesId) -> list[str]:
        raw_keys = {url for s_id, url in self._raw_docs if s_id == series_id}
        norm_keys = {url for s_id, url in self._norm_docs if s_id == series_id}
        return list(raw_keys | norm_keys)


class FakeClock(Clock):
    """Controllable fake implementation of Clock."""

    def __init__(self, start_time: datetime | None = None) -> None:
        self._current_time: datetime = start_time or datetime(
            2026, 1, 1, 0, 0, 0, tzinfo=UTC
        )

    def now(self) -> datetime:
        return self._current_time

    def set_time(self, new_time: datetime) -> None:
        self._current_time = new_time

    def advance(self, delta: timedelta) -> None:
        self._current_time += delta
