"""EntityRegistry port interface."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from lore_mcp.domain.entities import Entity
from lore_mcp.domain.types import EntityId, SeriesId


@runtime_checkable
class EntityRegistry(Protocol):
    """Port for managing, storing, and resolving canonical lore entities."""

    async def register_entity(self, entity: Entity) -> None:
        """Register or update a canonical lore entity."""
        ...

    async def register_entities(self, entities: Sequence[Entity]) -> None:
        """Register or update a batch of canonical lore entities."""
        ...

    async def get_entity(
        self, series_id: SeriesId, entity_id: EntityId
    ) -> Entity | None:
        """Retrieve an entity by its canonical slug ID within a series."""
        ...

    async def resolve_entity(
        self, series_id: SeriesId, name_or_alias: str
    ) -> Entity | None:
        """Resolve a canonical entity by canonical ID, display name, or alias."""
        ...

    async def list_entities(self, series_id: SeriesId) -> list[Entity]:
        """List all registered entities for a series."""
        ...
