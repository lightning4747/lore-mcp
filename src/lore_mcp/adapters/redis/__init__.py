"""Redis cache and budget counter adapters."""

from lore_mcp.adapters.redis.budget import RedisBudgetCounter
from lore_mcp.adapters.redis.cache import RedisCache
from lore_mcp.adapters.redis.circuit_breaker import CircuitBreaker, CircuitState
from lore_mcp.adapters.redis.client import create_redis_client
from lore_mcp.adapters.redis.command_tracker import CommandTracker
from lore_mcp.adapters.redis.decorators import (
    CURRENT_SCHEMA_VERSION,
    CachedRetrievalService,
    CachedWebSearch,
    GlobalRateGuard,
)
from lore_mcp.adapters.redis.errors import (
    RedisAdapterError,
    RedisCircuitOpenError,
    RedisConnectionError,
)
from lore_mcp.adapters.redis.keys import (
    build_budget_key,
    build_retrieval_cache_key,
    build_web_cache_key,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "CachedRetrievalService",
    "CachedWebSearch",
    "CircuitBreaker",
    "CircuitState",
    "CommandTracker",
    "GlobalRateGuard",
    "RedisAdapterError",
    "RedisBudgetCounter",
    "RedisCache",
    "RedisCircuitOpenError",
    "RedisConnectionError",
    "build_budget_key",
    "build_retrieval_cache_key",
    "build_web_cache_key",
    "create_redis_client",
]
