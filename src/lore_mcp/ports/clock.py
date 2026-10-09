"""Clock port interface."""

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Port for time inspection to enable deterministic testing."""

    def now(self) -> datetime:
        """Return the current datetime."""
        ...


class SystemClock(Clock):
    """Default system clock providing UTC current time."""

    def now(self) -> datetime:
        return datetime.now(UTC)
