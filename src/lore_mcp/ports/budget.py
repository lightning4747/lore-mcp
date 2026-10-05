"""BudgetCounter port interface."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class BudgetCounter(Protocol):
    """Port for atomic rate and daily quota budget tracking."""

    async def increment(self, key: str, window_seconds: int) -> int:
        """Atomically increment counter for key with TTL window, returning new value."""
        ...

    async def get_count(self, key: str) -> int:
        """Return the current count for key."""
        ...

    async def check_budget(self, key: str, limit: int) -> bool:
        """Check if current usage is strictly under the limit."""
        ...

    async def reset(self, key: str) -> None:
        """Reset or clear the counter for key."""
        ...
