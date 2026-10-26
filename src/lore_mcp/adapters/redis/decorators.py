"""Caching decorators and rate guards for service decoration (Tasks 12.2-12.4)."""

import json
import logging
from typing import Any

from lore_mcp.adapters.redis.keys import (
    build_budget_key,
    build_retrieval_cache_key,
    build_web_cache_key,
)
from lore_mcp.application.errors import RateLimited
from lore_mcp.application.retrieval_service import LoreRetrievalService
from lore_mcp.config import RedisSettings
from lore_mcp.domain.search import EvidenceBundle, WebSearchResult
from lore_mcp.ports.budget import BudgetCounter
from lore_mcp.ports.cache import Cache
from lore_mcp.ports.clock import Clock, SystemClock
from lore_mcp.ports.web_search import WebSearch

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 1


class CachedRetrievalService(LoreRetrievalService):
    """Decorator wrapping LoreRetrievalService to provide transparent caching.

    Use cases remain completely unaware of caching (Task 12.2).
    Cache entries are versioned JSON envelopes keyed with ingest_version.
    """

    def __init__(
        self,
        inner: LoreRetrievalService,
        cache: Cache,
        settings: RedisSettings | None = None,
    ) -> None:
        super().__init__(
            scope_resolver=inner._scope_resolver,
            retriever=inner._retriever,
            cache=None,
            web_search=inner._web_search,
            budget_counter=inner._budget_counter,
            classifier=inner._classifier,
            chronology_builder=inner._chronology_builder,
            clock=inner._clock,
            web_search_settings=inner._web_search_settings,
            limits_settings=inner._limits_settings,
            sufficiency_threshold=inner._sufficiency_threshold,
        )
        self._inner = inner
        self._cache = cache
        self._settings = settings or RedisSettings()
        self._ttl_seconds = self._settings.retrieval_cache_ttl

    @property
    def inner(self) -> LoreRetrievalService:
        return self._inner

    async def search(
        self,
        query: str,
        series: str,
        season: int | None = None,
        episode: int | None = None,
        max_season: int | None = None,
        max_episode: int | None = None,
        include_unscoped: bool = False,
        entity: str | None = None,
        max_results: int | None = None,
    ) -> EvidenceBundle:
        """Execute retrieval with transparent versioned caching."""
        # 1. Resolve scope to obtain series_id and active ingest_version
        resolved_scope = await self._inner._scope_resolver.resolve(series)
        series_id = str(resolved_scope.series_id)
        ingest_version = resolved_scope.manifest.ingest_version

        limit = (
            max_results
            if max_results is not None
            else self._inner._limits_settings.default_max_results
        )

        # 2. Build canonical cache key
        cache_key = build_retrieval_cache_key(
            tool_name="search_series_lore",
            series=series_id,
            ingest_version=ingest_version,
            query=query,
            season=season,
            episode=episode,
            max_season=max_season,
            max_episode=max_episode,
            include_unscoped=include_unscoped,
            entity=entity,
            source_type=None,
            max_results=limit,
            schema_version=CURRENT_SCHEMA_VERSION,
        )

        # 3. Check Cache
        try:
            cached_raw = await self._cache.get(cache_key)
            if cached_raw is not None:
                envelope = json.loads(cached_raw)
                cached_version = envelope.get("schema_version")
                if cached_version != CURRENT_SCHEMA_VERSION:
                    logger.info(
                        "Ignoring cached entry with mismatched schema "
                        "version (%s != %d)",
                        cached_version,
                        CURRENT_SCHEMA_VERSION,
                    )
                else:
                    payload = envelope.get("payload")
                    if isinstance(payload, dict):
                        bundle = EvidenceBundle.model_validate(payload)
                        notices = list(bundle.notices)
                        if "Cached result" not in notices:
                            notices.append("Cached result")
                        return bundle.model_copy(update={"notices": notices})
        except Exception as err:
            logger.warning("Cache lookup failed for key '%s': %s", cache_key, err)

        # 4. Cache miss: execute underlying retrieval service
        bundle = await self._inner.search(
            query=query,
            series=series,
            season=season,
            episode=episode,
            max_season=max_season,
            max_episode=max_episode,
            include_unscoped=include_unscoped,
            entity=entity,
            max_results=max_results,
        )

        # 5. Store in Cache as versioned envelope
        try:
            envelope_data: dict[str, Any] = {
                "schema_version": CURRENT_SCHEMA_VERSION,
                "payload": bundle.model_dump(mode="json"),
            }
            serialized = json.dumps(envelope_data)
            await self._cache.set(cache_key, serialized, ttl_seconds=self._ttl_seconds)
        except Exception as err:
            logger.warning("Cache write failed for key '%s': %s", cache_key, err)

        return bundle


class CachedWebSearch(WebSearch):
    """Decorator wrapping WebSearch port to cache web search results for 24h."""

    def __init__(
        self,
        inner: WebSearch,
        cache: Cache,
        settings: RedisSettings | None = None,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._settings = settings or RedisSettings()
        self._ttl_seconds = self._settings.web_cache_ttl

    async def search(
        self, query: str, limit: int = 3
    ) -> list[WebSearchResult]:
        """Execute web search with 24-hour caching preserving metadata."""
        cache_key = build_web_cache_key(series="global", query=query)

        try:
            cached_raw = await self._cache.get(cache_key)
            if cached_raw is not None:
                data = json.loads(cached_raw)
                if isinstance(data, list):
                    return [WebSearchResult.model_validate(item) for item in data]
        except Exception as err:
            logger.warning("Web cache lookup failed for key '%s': %s", cache_key, err)

        results = await self._inner.search(query, limit=limit)

        if results:
            try:
                serialized = json.dumps([r.model_dump(mode="json") for r in results])
                await self._cache.set(
                    cache_key, serialized, ttl_seconds=self._ttl_seconds
                )
            except Exception as err:
                logger.warning(
                    "Web cache write failed for key '%s': %s", cache_key, err
                )

        return results


class GlobalRateGuard:
    """Atomic global request-rate guard across all connector traffic (Task 12.4)."""

    def __init__(
        self,
        budget_counter: BudgetCounter,
        rate_limit_per_minute: int = 120,
        clock: Clock | None = None,
    ) -> None:
        self._budget_counter = budget_counter
        self._rate_limit = rate_limit_per_minute
        self._clock: Clock = clock or SystemClock()

    async def check_rate_limit(self) -> None:
        """Verify global connector request rate has not exceeded minute threshold."""
        now = self._clock.now()
        window = now.strftime("%Y-%m-%d-%H-%M")
        key = build_budget_key("global_rate", window)

        try:
            count = await self._budget_counter.increment(key, window_seconds=60)
            if count > self._rate_limit:
                raise RateLimited(
                    f"Global request rate limit exceeded "
                    f"({count} > {self._rate_limit} req/min)."
                )
        except RateLimited:
            raise
        except Exception as err:
            logger.warning("Global rate guard check failed: %s", err)
