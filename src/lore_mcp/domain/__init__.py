"""Domain layer: pure business models and domain logic."""

from lore_mcp.domain.documents import (
    Chunk,
    DocumentSection,
    NormalizedDocument,
    RawDocument,
)
from lore_mcp.domain.entities import Entity, Event
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
    "validate_series_id",
]
