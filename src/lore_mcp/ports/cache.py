"""Cache port interface."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Cache(Protocol):
    """Port for short-lived key-value caching."""

    async def get(self, key: str) -> str | None:
        """Retrieve a value by key, returning None if missing or expired."""
        ...

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        """Store a key-value pair with an optional TTL in seconds."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete a key from the cache, returning True if the key existed."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        ...
