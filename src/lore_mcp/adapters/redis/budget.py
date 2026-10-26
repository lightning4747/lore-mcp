"""Redis implementation of BudgetCounter port with atomic Lua scripts (Task 12.4)."""

import logging

import redis.asyncio as aioredis

from lore_mcp.adapters.redis.circuit_breaker import CircuitBreaker
from lore_mcp.adapters.redis.command_tracker import CommandTracker
from lore_mcp.adapters.redis.errors import RedisCircuitOpenError, RedisConnectionError
from lore_mcp.config import RedisSettings
from lore_mcp.ports.budget import BudgetCounter

logger = logging.getLogger(__name__)

# Atomic INCR and EXPIRE on initial key creation
LUA_INCR_EXPIRE = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return current
"""


class RedisBudgetCounter(BudgetCounter):
    """Atomic rate guard and quota budget tracker with fail-closed semantics.

    If Redis is down, unreachable, or circuit-breaking:
    budget checks fail closed (returning False) to prevent unmetered external spend.
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

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._circuit_breaker

    @property
    def command_tracker(self) -> CommandTracker:
        return self._command_tracker

    async def increment(self, key: str, window_seconds: int) -> int:
        """Atomically increment counter with TTL window. Fails closed on error."""
        if self._client is None:
            raise RedisConnectionError(
                f"Cannot increment budget key '{key}': Redis client is not configured."
            )

        if not self._circuit_breaker.can_execute():
            raise RedisCircuitOpenError(
                f"Cannot increment budget key '{key}': Redis circuit breaker is OPEN."
            )

        try:
            val = await self._client.eval(
                LUA_INCR_EXPIRE,
                1,
                key,
                str(window_seconds),
            )
            self._command_tracker.record_command()
            self._circuit_breaker.record_success()
            return int(val)
        except Exception as err:
            self._circuit_breaker.record_failure(err)
            logger.error("Redis budget increment failed for key '%s': %s", key, err)
            raise RedisConnectionError(
                f"Budget counter increment failed for key '{key}': {err}"
            ) from err

    async def get_count(self, key: str) -> int:
        """Return the current count for key. Returns 0 if missing or unconfigured."""
        if self._client is None or not self._circuit_breaker.can_execute():
            return 0

        try:
            val = await self._client.get(key)
            self._command_tracker.record_command()
            self._circuit_breaker.record_success()
            if val is None:
                return 0
            return int(val)
        except Exception as err:
            self._circuit_breaker.record_failure(err)
            logger.warning("Redis budget get_count failed for key '%s': %s", key, err)
            return 0

    async def check_budget(self, key: str, limit: int) -> bool:
        """Check if usage is under limit. Fails closed (False) on error."""
        if self._client is None:
            logger.warning(
                "Budget check failed closed for '%s': Redis client not configured.",
                key,
            )
            return False

        if not self._circuit_breaker.can_execute():
            logger.warning(
                "Budget check failed closed for '%s': circuit breaker is OPEN.",
                key,
            )
            return False

        try:
            val = await self._client.get(key)
            self._command_tracker.record_command()
            self._circuit_breaker.record_success()
            count = int(val) if val is not None else 0
            return count < limit
        except Exception as err:
            self._circuit_breaker.record_failure(err)
            logger.warning(
                "Budget check failed closed for key '%s' due to Redis error: %s",
                key,
                err,
            )
            return False

    async def reset(self, key: str) -> None:
        """Reset or clear the counter for key."""
        if self._client is None or not self._circuit_breaker.can_execute():
            return

        try:
            await self._client.delete(key)
            self._command_tracker.record_command()
            self._circuit_breaker.record_success()
        except Exception as err:
            self._circuit_breaker.record_failure(err)
            logger.warning("Redis budget reset failed for key '%s': %s", key, err)
