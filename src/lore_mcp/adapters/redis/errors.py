"""Redis adapter error definitions."""

from lore_mcp.application.errors import ApplicationError


class RedisAdapterError(ApplicationError):
    """Base exception for Redis adapter errors."""


class RedisConnectionError(RedisAdapterError):
    """Raised when Redis server is unreachable or connection fails."""


class RedisCircuitOpenError(RedisAdapterError):
    """Raised when Redis operations are blocked because circuit breaker is open."""
