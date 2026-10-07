"""SeriesRegistry port interface."""

from typing import Protocol, runtime_checkable

from lore_mcp.domain.entities import SeriesManifest
from lore_mcp.domain.types import SeriesId


@runtime_checkable
class SeriesRegistry(Protocol):
    """Port for publishing and resolving series manifests and boundaries."""

    async def register_series(self, manifest: SeriesManifest) -> None:
        """Register or update a series manifest."""
        ...

    async def get_series(self, series_id: SeriesId) -> SeriesManifest | None:
        """Retrieve the manifest for a given series."""
        ...

    async def list_series(self) -> list[SeriesId]:
        """List all registered series identifiers."""
        ...

    async def delete_series(self, series_id: SeriesId) -> bool:
        """Delete a series manifest from the registry."""
        ...

    async def resolve_alias(self, alias_or_name: str) -> SeriesId | None:
        """Resolve a display name or alias to canonical SeriesId."""
        ...

    async def get_alias_map(self) -> dict[str, SeriesId]:
        """Return a mapping of all lowercase aliases and names to SeriesId."""
        ...
