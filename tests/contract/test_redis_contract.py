"""Contract tests verifying RedisCache and RedisBudgetCounter (Task 12.7)."""

from typing import Any

import pytest

from lore_mcp.adapters.redis.budget import RedisBudgetCounter
from lore_mcp.adapters.redis.cache import RedisCache
from lore_mcp.ports.budget import BudgetCounter
from lore_mcp.ports.cache import Cache
from lore_mcp.testing.contracts import BudgetCounterContractSuite, CacheContractSuite


class InMemoryAsyncRedis:
    """In-memory async Redis double simulating key-value and Lua script commands."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._data.get(key)

    async def set(
        self, key: str, value: str, ex: int | None = None
    ) -> bool:
        self._data[key] = str(value)
        return True

    async def delete(self, *keys: str) -> int:
        count = 0
        for k in keys:
            if k in self._data:
                del self._data[k]
                count += 1
        return count

    async def exists(self, *keys: str) -> int:
        return sum(1 for k in keys if k in self._data)

    async def eval(
        self, script: str, numkeys: int, *args: Any
    ) -> int:
        # Simulates atomic LUA_INCR_EXPIRE script
        key = str(args[0])
        current = int(self._data.get(key, "0")) + 1
        self._data[key] = str(current)
        return current


@pytest.mark.contract
class TestRedisCacheContract(CacheContractSuite):
    """Verifies RedisCache satisfies Cache port contract."""

    @pytest.fixture
    def cache(self) -> Cache:
        fake_redis = InMemoryAsyncRedis()
        return RedisCache(client=fake_redis)  # type: ignore[arg-type]


@pytest.mark.contract
class TestRedisBudgetCounterContract(BudgetCounterContractSuite):
    """Verifies RedisBudgetCounter satisfies BudgetCounter port contract."""

    @pytest.fixture
    def counter(self) -> BudgetCounter:
        fake_redis = InMemoryAsyncRedis()
        return RedisBudgetCounter(client=fake_redis)  # type: ignore[arg-type]
