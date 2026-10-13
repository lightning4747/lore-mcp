"""EventRegistry port interface."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from lore_mcp.domain.entities import Event
from lore_mcp.domain.types import EventId, SeriesId


@runtime_checkable
class EventRegistry(Protocol):
    """Port for managing, storing, and resolving narrative events."""

    async def register_event(self, event: Event) -> None:
        """Register or update a narrative event."""
        ...

    async def register_events(self, events: Sequence[Event]) -> None:
        """Register or update a batch of narrative events."""
        ...

    async def get_event(self, series_id: SeriesId, event_id: EventId) -> Event | None:
        """Retrieve an event by its deterministic ID within a series."""
        ...

    async def resolve_event(self, series_id: SeriesId, name_or_id: str) -> Event | None:
        """Resolve an event by deterministic ID or name/title."""
        ...

    async def list_events(self, series_id: SeriesId) -> list[Event]:
        """List all registered events for a series."""
        ...
