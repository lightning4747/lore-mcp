"""Redis implementation of the Cache port with fail-open degradation."""

import logging

import redis.asyncio as aioredis

from lore_mcp.adapters.redis.circuit_breaker import CircuitBreaker
from lore_mcp.adapters.redis.command_tracker import CommandTracker
from lore_mcp.config import RedisSettings
from lore_mcp.ports.cache import Cache

logger = logging.getLogger(__name__)


class RedisCache(Cache):
    """Cache port implementation backed by Redis with fail-open semantics.

    If Redis is unavailable, timing out, circuit-breaking, or shedding commands,
    all cache read/write operations fail open without impacting retrieval correctness.
    """

    def __init__(
        self,
        client: aioredis.Redis | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        command_tracker: CommandTracker | None = None,
        settings: RedisSettings | None = None,
    ) -> None:
        self._client = client
        self._settings = settings or RedisSettings()
        self._circuit_breaker = circuit_breaker or CircuitBreaker(
            failure_threshold=self._settings.circuit_breaker_failure_threshold,
            reset_timeout=self._settings.circuit_breaker_reset_seconds,
        )
        self._command_tracker = command_tracker or CommandTracker(
            monthly_budget=self._settings.monthly_command_budget,
            shedding_ratio=self._settings.command_budget_shedding_ratio,
        )
        self._max_payload_bytes = self._settings.max_payload_bytes

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._circuit_breaker

    @property
    def command_tracker(self) -> CommandTracker:
        return self._command_tracker

    async def get(self, key: str) -> str | None:
        """Retrieve a value by key. Fails open returning None on any failure."""
        if self._client is None:
            return None

        if self._command_tracker.is_shedding_active():
            logger.debug("Cache read skipped due to command budget shedding: %s", key)
            return None

        if not self._circuit_breaker.can_execute():
            logger.debug("Cache read skipped due to open circuit breaker: %s", key)
            return None

        try:
            val = await self._client.get(key)
            self._command_tracker.record_command()
            self._circuit_breaker.record_success()
            if isinstance(val, bytes):
                return val.decode("utf-8")
            if isinstance(val, str):
                return val
            return None
        except Exception as err:
            self._circuit_breaker.record_failure(err)
            logger.warning("Redis cache get failed for key '%s': %s", key, err)
            return None

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        """Store key-value pair. Fails open without raising on any failure."""
        if self._client is None:
            return

        if self._command_tracker.is_shedding_active():
            logger.debug("Cache write skipped due to command budget shedding: %s", key)
            return

        if not self._circuit_breaker.can_execute():
            logger.debug("Cache write skipped due to open circuit breaker: %s", key)
            return

        # Check payload size cap
        encoded = value.encode("utf-8")
        if len(encoded) > self._max_payload_bytes:
            logger.info(
                "Skipping caching oversized payload for key '%s' (%d > %d bytes)",
                key,
                len(encoded),
                self._max_payload_bytes,
            )
            return

        try:
            if ttl_seconds is not None and ttl_seconds > 0:
                await self._client.set(key, value, ex=ttl_seconds)
            else:
                await self._client.set(key, value)
            self._command_tracker.record_command()
            self._circuit_breaker.record_success()
        except Exception as err:
            self._circuit_breaker.record_failure(err)
            logger.warning("Redis cache set failed for key '%s': %s", key, err)

    async def delete(self, key: str) -> bool:
        """Delete key from cache. Returns False on failure."""
        if self._client is None:
            return False

        if not self._circuit_breaker.can_execute():
            return False

        try:
            result = await self._client.delete(key)
            self._command_tracker.record_command()
            self._circuit_breaker.record_success()
            return bool(result > 0)
        except Exception as err:
            self._circuit_breaker.record_failure(err)
            logger.warning("Redis cache delete failed for key '%s': %s", key, err)
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache. Returns False on failure."""
        if self._client is None:
            return False

        if not self._circuit_breaker.can_execute():
            return False

        try:
            result = await self._client.exists(key)
            self._command_tracker.record_command()
            self._circuit_breaker.record_success()
            return bool(result > 0)
        except Exception as err:
            self._circuit_breaker.record_failure(err)
            logger.warning("Redis cache exists check failed for key '%s': %s", key, err)
            return False
