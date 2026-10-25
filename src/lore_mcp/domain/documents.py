"""Domain models for raw documents, normalized documents, and chunks."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.types import ChunkId, EntityId, ValidatedSeriesId


class DocumentSection(BaseModel):
    """Section or heading unit within a normalized document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    heading: str = Field(description="Section heading or path title")
    content: str = Field(description="Body content of the section")


class DialogueCue(BaseModel):
    """A single clean dialogue cue with timestamps, speaker, and speech text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start_ms: int = Field(description="Start time in milliseconds from episode start")
    end_ms: int = Field(description="End time in milliseconds from episode start")
    start_time: str = Field(description="Formatted start timestamp (HH:MM:SS,mmm)")
    end_time: str = Field(description="Formatted end timestamp (HH:MM:SS,mmm)")
    speaker: str | None = Field(default=None, description="Identified speaker name")
    text: str = Field(description="Normalized spoken dialogue text")


class DialogueSegment(BaseModel):
    """A scene-like temporal cluster of dialogue cues within an episode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scene_index: int = Field(description="1-indexed sequential scene order")
    start_ms: int = Field(description="Earliest cue start time in milliseconds")
    end_ms: int = Field(description="Latest cue end time in milliseconds")
    start_time: str = Field(description="Formatted start timestamp")
    end_time: str = Field(description="Formatted end timestamp")
    cues: list[DialogueCue] = Field(
        default_factory=list[DialogueCue],
        description="Chronologically ordered dialogue cues in this scene",
    )
    text: str = Field(description="Formatted scene dialogue transcript")


class RawDocument(BaseModel):
    """Raw unmodified source document fetched from an external source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: ValidatedSeriesId = Field(description="Owning series identifier")
    source_url: str = Field(description="Canonical URL or locator")
    source_type: str = Field(
        description="Source type (e.g. 'fandom', 'transcript', 'wikipedia')"
    )
    content: str = Field(description="Raw fetched payload text/markup")
    retrieval_timestamp: datetime = Field(
        description="Timestamp when document was fetched"
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Source-specific key-value metadata",
    )


class NormalizedDocument(BaseModel):
    """Cleaned and structured document ready for entity extraction and chunking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: ValidatedSeriesId = Field(description="Owning series identifier")
    source_url: str = Field(description="Canonical URL or locator")
    source_type: str = Field(description="Source type")
    title: str = Field(description="Document or page title")
    text: str = Field(description="Cleaned, normalized plaintext")
    sections: list[DocumentSection] = Field(
        default_factory=list[DocumentSection],
        description="Ordered list of structural sections",
    )
    content_hash: str = Field(
        description="Cryptographic hash of the normalized content"
    )
    retrieval_timestamp: datetime = Field(
        description="Timestamp when document was fetched"
    )
    metadata: dict[str, str] = Field(
        default_factory=dict[str, str],
        description="Cleaned metadata attributes",
    )


class Chunk(BaseModel):
    """Discrete entity-aware or narrative chunk indexed for retrieval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: ChunkId = Field(description="Deterministic stable chunk identifier")
    series_id: ValidatedSeriesId = Field(description="Owning series identifier")
    text: str = Field(description="Textual chunk content")
    season: int | None = Field(default=None, description="Season number if applicable")
    episode: int | None = Field(
        default=None, description="Episode number if applicable"
    )
    scene_index: int | None = Field(
        default=None,
        description="Displayed scene index within the episode",
    )
    narrative_order: int | None = Field(
        default=None,
        description="Chronological position in the universe timeline",
    )
    schema_version: int = Field(
        default=1, description="Data schema version for vector payload"
    )
    ingest_version: int = Field(
        default=1,
        description="Ingestion batch version to support cache invalidation",
    )
    content_hash: str = Field(
        description="Cryptographic hash of chunk content for change detection"
    )
    entity_ids: list[EntityId] = Field(
        default_factory=list[EntityId],
        description="Associated canonical entity identifiers",
    )
    provenance: Provenance = Field(
        description="Source provenance for attribution and traceability"
    )
