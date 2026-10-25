"""Application layer: orchestrations, use cases, and ingestion stages."""

from lore_mcp.application.chunk_service import ChunkService
from lore_mcp.application.classifier import ClassifiedQuery, QueryClassifier
from lore_mcp.application.dense import DenseRetriever
from lore_mcp.application.dialogue_service import DialogueService
from lore_mcp.application.entity_service import EntityService
from lore_mcp.application.errors import (
    ApplicationError,
    DataUnavailable,
    DependencyUnavailable,
    IncompatibleData,
    InvalidArgument,
    InvalidScope,
    RateLimited,
)
from lore_mcp.application.event_service import EventService
from lore_mcp.application.extraction import (
    AliasTableBuilder,
    CuratedTimelineLoader,
    DefaultEntityExtractor,
    DefaultEventExtractor,
    ManifestBuilder,
)
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.application.normalization import (
    ContentDeduplicator,
    DefaultDialogueNormalizer,
    DefaultDocumentNormalizer,
    DefaultNormalizationPipeline,
)
from lore_mcp.application.retrieval_service import (
    LoreRetrievalService,
    compute_cache_key,
)
from lore_mcp.application.scope import ResolvedScope, ScopeResolver
from lore_mcp.application.series_loader import (
    load_all_series_configs,
    load_series_config,
)
from lore_mcp.application.sparse import SparseRetriever

__all__ = [
    "AliasTableBuilder",
    "ApplicationError",
    "ChunkService",
    "ClassifiedQuery",
    "ContentDeduplicator",
    "CuratedTimelineLoader",
    "DataUnavailable",
    "DefaultDialogueNormalizer",
    "DefaultDocumentNormalizer",
    "DefaultEntityExtractor",
    "DefaultEventExtractor",
    "DefaultNormalizationPipeline",
    "DenseRetriever",
    "DependencyUnavailable",
    "DialogueService",
    "EntityService",
    "EventService",
    "HybridRetriever",
    "IncompatibleData",
    "InvalidArgument",
    "InvalidScope",
    "LoreRetrievalService",
    "ManifestBuilder",
    "QueryClassifier",
    "RateLimited",
    "ResolvedScope",
    "ScopeResolver",
    "SparseRetriever",
    "compute_cache_key",
    "load_all_series_configs",
    "load_series_config",
]
