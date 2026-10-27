"""Unit tests for CachedRetrievalService and CachedWebSearch (Tasks 12.2-12.4)."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from lore_mcp.adapters.redis.decorators import (
    CachedRetrievalService,
    CachedWebSearch,
    GlobalRateGuard,
)
from lore_mcp.adapters.redis.keys import build_retrieval_cache_key
from lore_mcp.application.errors import RateLimited
from lore_mcp.config import RedisSettings
from lore_mcp.domain.entities import SeriesManifest
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import Evidence, EvidenceBundle, WebSearchResult
from lore_mcp.domain.types import ChunkId, SeriesId
from lore_mcp.testing import FakeBudgetCounter, FakeCache, FakeClock


@pytest.fixture
def sample_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        query="loki coffee",
        series_id=SeriesId("loki"),
        status="ok",
        evidence=[
            Evidence(
                chunk_id=ChunkId("chunk-1"),
                series_id=SeriesId("loki"),
                content="Mobius offers Loki coffee",
                score=0.9,
                provenance=Provenance(
                    source_type="transcript",
                    source_url="https://example.com/loki",
                    retrieval_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                ),
            )
        ],
        notices=["Original notice"],
    )


@pytest.mark.unit
class TestCachedRetrievalService:
    """Tests transparent caching, versioning, invalidation, and payload caps."""

    @pytest.mark.asyncio
    async def test_cache_miss_then_hit(self, sample_bundle: EvidenceBundle) -> None:
        mock_inner = AsyncMock()
        mock_scope_resolver = AsyncMock()
        mock_manifest = SeriesManifest(
            series_id=SeriesId("loki"),
            display_name="Loki",
            kind="show",
            ingest_version=1,
        )
        resolved = AsyncMock()
        resolved.series_id = SeriesId("loki")
        resolved.manifest = mock_manifest
        mock_scope_resolver.resolve.return_value = resolved
        mock_inner._scope_resolver = mock_scope_resolver
        mock_inner._limits_settings.default_max_results = 8
        mock_inner.search.return_value = sample_bundle

        cache = FakeCache()
        cached_svc = CachedRetrievalService(inner=mock_inner, cache=cache)

        # 1. First call: Cache miss -> calls inner.search()
        res1 = await cached_svc.search(query="loki coffee", series="loki")
        assert mock_inner.search.call_count == 1
        assert res1.query == "loki coffee"
        assert "Cached result" not in res1.notices

        # Verify entry in cache
        expected_key = build_retrieval_cache_key(
            tool_name="search_series_lore",
            series="loki",
            ingest_version=1,
            query="loki coffee",
        )
        assert await cache.exists(expected_key) is True

        # 2. Second call: Cache hit -> does NOT call inner.search() again
        res2 = await cached_svc.search(query="loki coffee", series="loki")
        assert mock_inner.search.call_count == 1
        assert res2.query == "loki coffee"
        assert "Cached result" in res2.notices
        assert len(res2.evidence) == 1
        assert res2.evidence[0].chunk_id == "chunk-1"

    @pytest.mark.asyncio
    async def test_ingest_version_bump_invalidates_cache(
        self, sample_bundle: EvidenceBundle
    ) -> None:
        mock_inner = AsyncMock()
        mock_scope_resolver = AsyncMock()

        # Ingest version 1
        manifest_v1 = SeriesManifest(
            series_id=SeriesId("loki"),
            display_name="Loki",
            kind="show",
            ingest_version=1,
        )
        resolved_v1 = AsyncMock()
        resolved_v1.series_id = SeriesId("loki")
        resolved_v1.manifest = manifest_v1
        mock_scope_resolver.resolve.return_value = resolved_v1
        mock_inner._scope_resolver = mock_scope_resolver
        mock_inner._limits_settings.default_max_results = 8
        mock_inner.search.return_value = sample_bundle

        cache = FakeCache()
        cached_svc = CachedRetrievalService(inner=mock_inner, cache=cache)

        await cached_svc.search(query="loki coffee", series="loki")
        assert mock_inner.search.call_count == 1

        # Bump ingest_version to 2 on manifest
        manifest_v2 = SeriesManifest(
            series_id=SeriesId("loki"),
            display_name="Loki",
            kind="show",
            ingest_version=2,
        )
        resolved_v2 = AsyncMock()
        resolved_v2.series_id = SeriesId("loki")
        resolved_v2.manifest = manifest_v2
        mock_scope_resolver.resolve.return_value = resolved_v2

        # Second call with new ingest version is a cache miss
        await cached_svc.search(query="loki coffee", series="loki")
        assert mock_inner.search.call_count == 2

    @pytest.mark.asyncio
    async def test_mismatched_schema_version_is_ignored(
        self, sample_bundle: EvidenceBundle
    ) -> None:
        mock_inner = AsyncMock()
        mock_scope_resolver = AsyncMock()
        mock_manifest = SeriesManifest(
            series_id=SeriesId("loki"),
            display_name="Loki",
            kind="show",
            ingest_version=1,
        )
        resolved = AsyncMock()
        resolved.series_id = SeriesId("loki")
        resolved.manifest = mock_manifest
        mock_scope_resolver.resolve.return_value = resolved
        mock_inner._scope_resolver = mock_scope_resolver
        mock_inner._limits_settings.default_max_results = 8
        mock_inner.search.return_value = sample_bundle

        cache = FakeCache()
        cache_key = build_retrieval_cache_key(
            tool_name="search_series_lore",
            series="loki",
            ingest_version=1,
            query="loki coffee",
        )
        # Store an envelope with an old/incompatible schema_version = 99
        stale_envelope = {
            "schema_version": 99,
            "payload": sample_bundle.model_dump(mode="json"),
        }
        await cache.set(cache_key, json.dumps(stale_envelope))

        cached_svc = CachedRetrievalService(inner=mock_inner, cache=cache)

        # Mismatched schema version should be ignored, invoking inner service
        res = await cached_svc.search(query="loki coffee", series="loki")
        assert mock_inner.search.call_count == 1
        assert "Cached result" not in res.notices

    @pytest.mark.asyncio
    async def test_oversized_payload_skipped_by_cache(
        self, sample_bundle: EvidenceBundle
    ) -> None:
        mock_inner = AsyncMock()
        mock_scope_resolver = AsyncMock()
        mock_manifest = SeriesManifest(
            series_id=SeriesId("loki"),
            display_name="Loki",
            kind="show",
            ingest_version=1,
        )
        resolved = AsyncMock()
        resolved.series_id = SeriesId("loki")
        resolved.manifest = mock_manifest
        mock_scope_resolver.resolve.return_value = resolved
        mock_inner._scope_resolver = mock_scope_resolver
        mock_inner._limits_settings.default_max_results = 8
        mock_inner.search.return_value = sample_bundle

        # Configure small max_payload_bytes (e.g. 50 bytes)
        small_settings = RedisSettings(max_payload_bytes=50)
        cache = FakeCache()
        cached_svc = CachedRetrievalService(
            inner=mock_inner, cache=cache, settings=small_settings
        )

        res = await cached_svc.search(query="loki coffee", series="loki")
        assert res.query == "loki coffee"
        assert mock_inner.search.call_count == 1


@pytest.mark.unit
class TestCachedWebSearch:
    """Verifies web search caching and 24h metadata preservation."""

    @pytest.mark.asyncio
    async def test_web_search_miss_then_hit(self) -> None:
        mock_web = AsyncMock()
        mock_results = [
            WebSearchResult(
                url="https://marvel.com/loki",
                title="Loki Episode Guide",
                snippet="Loki interrogates Mobius at TVA.",
            )
        ]
        mock_web.search.return_value = mock_results

        cache = FakeCache()
        cached_web = CachedWebSearch(inner=mock_web, cache=cache)

        # 1. Miss
        res1 = await cached_web.search("Loki TVA episode", limit=3)
        assert mock_web.search.call_count == 1
        assert len(res1) == 1
        assert res1[0].url == "https://marvel.com/loki"

        # 2. Hit
        res2 = await cached_web.search("Loki TVA episode", limit=3)
        assert mock_web.search.call_count == 1
        assert len(res2) == 1
        assert res2[0].url == "https://marvel.com/loki"
        assert res2[0].snippet == "Loki interrogates Mobius at TVA."


@pytest.mark.unit
class TestGlobalRateGuard:
    """Verifies atomic global connector rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_guard_under_and_over_limit(self) -> None:
        counter = FakeBudgetCounter()
        clock = FakeClock(start_time=datetime(2026, 10, 26, 12, 0, tzinfo=UTC))
        guard = GlobalRateGuard(
            budget_counter=counter, rate_limit_per_minute=2, clock=clock
        )

        # First request: 1 <= 2 -> allowed
        await guard.check_rate_limit()

        # Second request: 2 <= 2 -> allowed
        await guard.check_rate_limit()

        # Third request: 3 > 2 -> raises RateLimited
        with pytest.raises(RateLimited, match="Global request rate limit exceeded"):
            await guard.check_rate_limit()
