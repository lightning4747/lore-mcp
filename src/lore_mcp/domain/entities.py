"""Domain models for entities and narrative events."""

from pydantic import BaseModel, ConfigDict, Field

from lore_mcp.domain.types import EntityId, EventId, ValidatedSeriesId


class Entity(BaseModel):
    """Canonical lore entity (character, location, device, organization, concept)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_id: EntityId = Field(
        description="Canonical slug identifier within the series"
    )
    series_id: ValidatedSeriesId = Field(description="Owning series identifier")
    name: str = Field(description="Canonical display name of the entity")
    entity_type: str = Field(
        description="Category (e.g. character, location, device, organization, concept)"
    )
    aliases: list[str] = Field(
        default_factory=list[str],
        description="Alternative names, titles, and shorthand references",
    )
    summary: str = Field(default="", description="Concise synopsis of the entity")
    related_entities: list[EntityId] = Field(
        default_factory=list[EntityId],
        description="Identifiers of directly associated entities",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict[str, str],
        description="Extensible entity properties (e.g. status, affiliations)",
    )


class Event(BaseModel):
    """Narrative event or occurrence within a series timeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: EventId = Field(
        description="Deterministic event identifier within the series"
    )
    series_id: ValidatedSeriesId = Field(description="Owning series identifier")
    name: str = Field(description="Name or title of the event")
    description: str = Field(description="Narrative account of what occurred")
    season: int | None = Field(
        default=None, description="Season where the event is depicted"
    )
    episode: int | None = Field(
        default=None, description="Episode where the event is depicted"
    )
    scene_index: int | None = Field(
        default=None, description="Scene index within the depicting episode"
    )
    narrative_order: int | None = Field(
        default=None,
        description="Position in fictional in-universe chronological order",
    )
    participants: list[EntityId] = Field(
        default_factory=list[EntityId],
        description="Entities participating in or affected by the event",
    )
    location: str | None = Field(
        default=None, description="Location where the event occurs"
    )
    consequences: list[str] = Field(
        default_factory=list[str],
        description="Resulting consequences or follow-up events",
    )
