"""Extraction port interfaces for entity and event discovery."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from lore_mcp.domain.documents import NormalizedDocument
from lore_mcp.domain.entities import (
    AliasTable,
    Entity,
    Event,
    ExtractionReport,
)
from lore_mcp.domain.series_config import SeriesConfig


@runtime_checkable
class EntityExtractor(Protocol):
    """Port for extracting typed Entity records from normalized wiki documents."""

    def extract_entity(
        self,
        doc: NormalizedDocument,
        config: SeriesConfig,
    ) -> Entity | None:
        """Extract a single Entity or None (e.g. disambiguation page)."""
        ...

    def extract_entities(
        self,
        docs: Sequence[NormalizedDocument],
        config: SeriesConfig,
    ) -> tuple[list[Entity], ExtractionReport]:
        """Extract entities from a batch of documents and compile report."""
        ...


@runtime_checkable
class EventExtractor(Protocol):
    """Port for extracting Event records from plot sections of normalized documents."""

    def extract_events(
        self,
        doc: NormalizedDocument,
        config: SeriesConfig,
        entity_alias_table: AliasTable | None = None,
        curated_narrative_orders: dict[str, int] | None = None,
    ) -> list[Event]:
        """Extract zero or more Event records from a single normalized document."""
        ...
