"""Clock port interface."""

from datetime import datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Port for time inspection to enable deterministic testing."""

    def now(self) -> datetime:
        """Return the current datetime."""
        ...
