"""Chunking port interfaces for entity, event, and dialogue chunking."""

from typing import Protocol, runtime_checkable

from lore_mcp.domain.documents import Chunk, NormalizedDocument
from lore_mcp.domain.entities import Entity, Event
from lore_mcp.domain.provenance import Provenance


@runtime_checkable
class EntityChunker(Protocol):
    """Port for chunking entity documents around lore units."""

    def chunk_entity(
        self,
        document: NormalizedDocument,
        entity: Entity,
    ) -> list[Chunk]:
        """Split a normalized entity document into discrete chunks."""
        ...


@runtime_checkable
class EventChunker(Protocol):
    """Port for chunking events with participants and consequences."""

    def chunk_event(
        self,
        event: Event,
        provenance: Provenance,
    ) -> list[Chunk]:
        """Convert a narrative event into one or more chunks."""
        ...


@runtime_checkable
class DialogueChunker(Protocol):
    """Port for chunking dialogue transcripts by scene-like windows."""

    def chunk_dialogue(
        self,
        document: NormalizedDocument,
    ) -> list[Chunk]:
        """Split a normalized dialogue transcript document into scene chunks."""
        ...
