"""Unit tests for Redis cache degradation and shedding (Tasks 12.5, 12.6)."""

from unittest.mock import AsyncMock

import pytest

from lore_mcp.adapters.redis.budget import RedisBudgetCounter
from lore_mcp.adapters.redis.cache import RedisCache
from lore_mcp.adapters.redis.circuit_breaker import CircuitBreaker
from lore_mcp.adapters.redis.command_tracker import CommandTracker
from lore_mcp.adapters.redis.errors import RedisCircuitOpenError, RedisConnectionError
from lore_mcp.config import RedisSettings


@pytest.mark.unit
class TestRedisCacheDegradation:
    """Verifies fail-open semantics: cache errors never interrupt retrieval."""

    @pytest.mark.asyncio
    async def test_cache_fails_open_when_client_is_none(self) -> None:
        cache = RedisCache(client=None)
        assert await cache.get("any_key") is None
        # set, delete, exists should not raise
        await cache.set("any_key", "val", ttl_seconds=60)
        assert await cache.delete("any_key") is False
        assert await cache.exists("any_key") is False

    @pytest.mark.asyncio
    async def test_cache_fails_open_on_redis_exceptions(self) -> None:
        mock_client = AsyncMock()
        mock_client.get.side_effect = ConnectionError("Connection refused")
        mock_client.set.side_effect = TimeoutError("Redis socket timeout")

        cache = RedisCache(client=mock_client)

        # get fails open (returns None)
        assert await cache.get("broken_key") is None
        # set fails open (does not raise)
        await cache.set("broken_key", "val")

    @pytest.mark.asyncio
    async def test_cache_drops_oversized_payload(self) -> None:
        mock_client = AsyncMock()
        settings = RedisSettings(max_payload_bytes=50)
        cache = RedisCache(client=mock_client, settings=settings)

        # 100 character payload exceeds 50 bytes cap
        large_payload = "x" * 100
        await cache.set("oversized_key", large_payload)

        # mock_client.set should not be called
        mock_client.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_skipped_when_command_shedding_is_active(self) -> None:
        mock_client = AsyncMock()
        tracker = CommandTracker(monthly_budget=100, shedding_ratio=0.80)
        tracker.record_command(85)  # Trips 80% shedding threshold

        cache = RedisCache(client=mock_client, command_tracker=tracker)

        assert await cache.get("some_key") is None
        await cache.set("some_key", "val")
        mock_client.get.assert_not_called()
        mock_client.set.assert_not_called()


@pytest.mark.unit
class TestRedisBudgetDegradation:
    """Verifies fail-closed semantics: budget errors block external spending."""

    @pytest.mark.asyncio
    async def test_budget_fails_closed_when_client_is_none(self) -> None:
        budget = RedisBudgetCounter(client=None)

        # check_budget fails closed returning False
        assert await budget.check_budget("web_search_daily", limit=10) is False

        # increment raises to prevent unmetered operation
        with pytest.raises(RedisConnectionError, match="client is not configured"):
            await budget.increment("web_search_daily", 86400)

    @pytest.mark.asyncio
    async def test_budget_fails_closed_on_redis_error(self) -> None:
        mock_client = AsyncMock()
        mock_client.get.side_effect = ConnectionError("Redis down")
        mock_client.eval.side_effect = ConnectionError("Redis down")

        budget = RedisBudgetCounter(client=mock_client)

        # check_budget fails closed returning False
        assert await budget.check_budget("web_search_daily", limit=10) is False

        # increment fails closed raising exception
        with pytest.raises(RedisConnectionError):
            await budget.increment("web_search_daily", 86400)

    @pytest.mark.asyncio
    async def test_budget_fails_closed_when_circuit_breaker_open(self) -> None:
        mock_client = AsyncMock()
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=60.0)
        cb.record_failure(RuntimeError("Outage"))
        assert cb.state == "open"

        budget = RedisBudgetCounter(client=mock_client, circuit_breaker=cb)

        # check_budget fails closed returning False
        assert await budget.check_budget("key", limit=5) is False

        # increment raises RedisCircuitOpenError
        with pytest.raises(RedisCircuitOpenError, match="circuit breaker is OPEN"):
            await budget.increment("key", 60)
