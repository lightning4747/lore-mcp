"""Domain layer: pure business models and domain logic."""

from lore_mcp.domain.documents import (
    Chunk,
    DocumentSection,
    NormalizedDocument,
    RawDocument,
)
from lore_mcp.domain.entities import (
    Entity,
    EntityDetails,
    Event,
    EventDetails,
    EventTrace,
    SeriesManifest,
)
from lore_mcp.domain.identifiers import (
    LORE_CHUNK_NAMESPACE,
    generate_chunk_id,
    generate_entity_id,
    generate_event_id,
    slugify,
)
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import (
    DialogueSnippet,
    Evidence,
    EvidenceBundle,
    SearchHit,
    SearchQuery,
    WebSearchResult,
    reciprocal_rank_fusion,
    rerank_candidates,
)
from lore_mcp.domain.series_config import SeriesConfig, WikiEndpointConfig
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
    "DialogueSnippet",
    "DocumentSection",
    "Entity",
    "EntityDetails",
    "EntityId",
    "Event",
    "EventDetails",
    "EventId",
    "EventTrace",
    "Evidence",
    "EvidenceBundle",
    "NormalizedDocument",
    "Provenance",
    "RawDocument",
    "SearchHit",
    "SearchQuery",
    "SeriesConfig",
    "SeriesId",
    "SeriesManifest",
    "ValidatedSeriesId",
    "WebSearchResult",
    "WikiEndpointConfig",
    "generate_chunk_id",
    "generate_entity_id",
    "generate_event_id",
    "reciprocal_rank_fusion",
    "rerank_candidates",
    "slugify",
    "validate_series_id",
]
