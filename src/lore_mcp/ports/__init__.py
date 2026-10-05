"""Ports layer: typing.Protocol interfaces defining system boundaries."""

from lore_mcp.ports.budget import BudgetCounter
from lore_mcp.ports.cache import Cache
from lore_mcp.ports.clock import Clock
from lore_mcp.ports.documents import DocumentSource, DocumentStore
from lore_mcp.ports.series_registry import SeriesRegistry
from lore_mcp.ports.vector_index import VectorIndex
from lore_mcp.ports.web_search import WebSearch

__all__ = [
    "BudgetCounter",
    "Cache",
    "Clock",
    "DocumentSource",
    "DocumentStore",
    "SeriesRegistry",
    "VectorIndex",
    "WebSearch",
]
