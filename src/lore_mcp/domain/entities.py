"""Domain models for entities and narrative events."""

import unicodedata

from pydantic import BaseModel, ConfigDict, Field

from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import Evidence
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


class SeriesManifest(BaseModel):
    """Metadata manifest describing an indexed series and its boundaries."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: ValidatedSeriesId = Field(description="Unique series slug identifier")
    display_name: str = Field(description="Human-readable title of the series")
    kind: str = Field(
        default="show",
        description="Type of series (e.g. 'show', 'anime')",
    )
    aliases: list[str] = Field(
        default_factory=list[str],
        description="Alternative names and search aliases for the series",
    )
    seasons: list[int] = Field(
        default_factory=list[int],
        description="List of available season numbers",
    )
    entity_count: int = Field(
        default=0, description="Total number of extracted entities"
    )
    chunk_count: int = Field(
        default=0, description="Total number of indexed vector chunks"
    )
    ingest_version: int = Field(
        default=1, description="Current ingestion batch version"
    )
    schema_version: int = Field(
        default=1, description="Data schema version for manifest compatibility"
    )


class EntityDetails(BaseModel):
    """Canonical entity metadata with associated evidence and provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity: Entity = Field(description="Canonical entity metadata and attributes")
    evidence: list[Evidence] = Field(
        default_factory=list[Evidence],
        description="Associated evidence chunks retrieved for this entity",
    )
    provenance: list[Provenance] = Field(
        default_factory=list[Provenance],
        description="Deduplicated source provenance records for the entity",
    )

    notices: list[str] = Field(
        default_factory=list[str],
        description="Operational notices or warnings",
    )


class EventTrace(BaseModel):
    """Resolved event trace containing participants, locations, entities,
    and chronological evidence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event: Event = Field(description="Resolved canonical event metadata")
    participants: list[EntityId] = Field(
        default_factory=list[EntityId],
        description="Entities participating in or affected by the event",
    )
    locations: list[str] = Field(
        default_factory=list[str],
        description="Relevant locations where the event took place",
    )
    entities: list[EntityId] = Field(
        default_factory=list[EntityId],
        description="Relevant entities associated with the event and evidence",
    )
    evidence: list[Evidence] = Field(
        default_factory=list[Evidence],
        description="Related evidence chunks organized in chronological order",
    )
    chronology: list[str] = Field(
        default_factory=list[str],
        description="Sequential chronological notes or timeline markers",
    )
    provenance: list[Provenance] = Field(
        default_factory=list[Provenance],
        description="Deduplicated source provenance records for retrieved evidence",
    )
    notices: list[str] = Field(
        default_factory=list[str],
        description="Operational notices or warnings",
    )


EventDetails = EventTrace


class EventComparisonSide(BaseModel):
    """Structured evidence and provenance for one side of a narrative comparison."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reference: str = Field(description="Original event or entity reference queried")
    target_type: str = Field(
        default="event",
        description="Type of resolved reference ('event' or 'entity')",
    )
    event: Event | None = Field(
        default=None, description="Resolved canonical event metadata if an event"
    )
    entity: Entity | None = Field(
        default=None, description="Resolved canonical entity metadata if an entity"
    )
    participants: list[EntityId] = Field(
        default_factory=list[EntityId],
        description="Key participating or involved entities",
    )
    locations: list[str] = Field(
        default_factory=list[str],
        description="Locations associated with this occurrence or entity",
    )
    entities: list[EntityId] = Field(
        default_factory=list[EntityId],
        description="All relevant entities extracted from evidence",
    )
    evidence: list[Evidence] = Field(
        default_factory=list[Evidence],
        description="Chronologically ordered evidence chunks for this side",
    )
    chronology: list[str] = Field(
        default_factory=list[str],
        description="Sequential timeline notes for this side",
    )
    provenance: list[Provenance] = Field(
        default_factory=list[Provenance],
        description="Dedicated, separate source provenance records for this side",
    )


class EventComparison(BaseModel):
    """Comparable structured evidence between two narrative events or entities."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: ValidatedSeriesId = Field(description="Owning series identifier")
    side_a: EventComparisonSide = Field(
        description="Resolved evidence and provenance for reference A"
    )
    side_b: EventComparisonSide = Field(
        description="Resolved evidence and provenance for reference B"
    )
    shared_entities: list[EntityId] = Field(
        default_factory=list[EntityId],
        description="Entities present in or shared across both occurrences",
    )
    shared_locations: list[str] = Field(
        default_factory=list[str],
        description="Locations present in or shared across both occurrences",
    )
    notices: list[str] = Field(
        default_factory=list[str],
        description="Operational notices or warnings",
    )


class AliasEntry(BaseModel):
    """Single alias entry mapping a normalized lookup key to a canonical entity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lookup_key: str = Field(
        description="Lowercase and NFC-normalized alias or canonical name"
    )
    entity_id: EntityId = Field(description="Target canonical entity identifier")
    canonical_name: str = Field(description="Display name for the entity")
    entity_type: str = Field(
        description="Entity type category (e.g. character, location)"
    )


class AliasTable(BaseModel):
    """Per-series alias lookup index for case/diacritic-insensitive resolution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: ValidatedSeriesId = Field(description="Owning series identifier")
    entries: list[AliasEntry] = Field(
        default_factory=list[AliasEntry], description="List of alias mapping entries"
    )

    def lookup(self, name: str) -> EntityId | None:
        """Case-insensitive, NFC-normalized lookup. Returns None on miss."""
        key = unicodedata.normalize("NFC", name.strip().casefold())
        for entry in self.entries:
            if entry.lookup_key == key:
                return entry.entity_id
        return None


class ExtractionReport(BaseModel):
    """Summary of entity and event extraction results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: ValidatedSeriesId = Field(description="Owning series identifier")
    documents_processed: int = Field(
        default=0, description="Total source documents processed"
    )
    entities_extracted: int = Field(
        default=0, description="Total valid entities extracted"
    )
    events_extracted: int = Field(
        default=0, description="Total valid events extracted"
    )
    disambiguation_pages_skipped: int = Field(
        default=0, description="Total disambiguation pages skipped"
    )
    unmapped_categories: list[str] = Field(
        default_factory=list[str],
        description="Categories encountered that had no explicit entity type mapping",
    )
    unmapped_category_documents: dict[str, list[str]] = Field(
        default_factory=dict[str, list[str]],
        description="Mapping of unmapped categories to affected document titles",
    )
