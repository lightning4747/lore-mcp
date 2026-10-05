"""Domain layer: pure business models and domain logic."""

from lore_mcp.domain.documents import (
    Chunk,
    DocumentSection,
    NormalizedDocument,
    RawDocument,
)
from lore_mcp.domain.entities import Entity, Event
from lore_mcp.domain.identifiers import (
    LORE_CHUNK_NAMESPACE,
    generate_chunk_id,
    generate_entity_id,
    generate_event_id,
    slugify,
)
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import (
    Evidence,
    EvidenceBundle,
    SearchHit,
    SearchQuery,
)
from lore_mcp.domain.types import (
    ChunkId,
    EntityId,
    EventId,
    SeriesId,
    ValidatedSeriesId,
    validate_series_id,
)

__all__ = [
    "LORE_CHUNK_NAMESPACE",
    "Chunk",
    "ChunkId",
    "DocumentSection",
    "Entity",
    "EntityId",
    "Event",
    "EventId",
    "Evidence",
    "EvidenceBundle",
    "NormalizedDocument",
    "Provenance",
    "RawDocument",
    "SearchHit",
    "SearchQuery",
    "SeriesId",
    "ValidatedSeriesId",
    "generate_chunk_id",
    "generate_entity_id",
    "generate_event_id",
    "slugify",
    "validate_series_id",
]
