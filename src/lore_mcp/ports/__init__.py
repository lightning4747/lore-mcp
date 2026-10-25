"""Ports layer: typing.Protocol interfaces defining system boundaries."""

from lore_mcp.ports.budget import BudgetCounter
from lore_mcp.ports.cache import Cache
from lore_mcp.ports.clock import Clock, SystemClock
from lore_mcp.ports.documents import DocumentSource, DocumentStore
from lore_mcp.ports.entity_registry import EntityRegistry
from lore_mcp.ports.event_registry import EventRegistry
from lore_mcp.ports.extraction import EntityExtractor, EventExtractor
from lore_mcp.ports.normalization import (
    Deduplicator,
    DialogueNormalizer,
    DocumentNormalizer,
    NormalizationPipeline,
)
from lore_mcp.ports.series_registry import SeriesRegistry
from lore_mcp.ports.vector_index import VectorIndex
from lore_mcp.ports.web_search import WebSearch

__all__ = [
    "BudgetCounter",
    "Cache",
    "Clock",
    "Deduplicator",
    "DialogueNormalizer",
    "DocumentNormalizer",
    "DocumentSource",
    "DocumentStore",
    "EntityExtractor",
    "EntityRegistry",
    "EventExtractor",
    "EventRegistry",
    "NormalizationPipeline",
    "SeriesRegistry",
    "SystemClock",
    "VectorIndex",
    "WebSearch",
]
