"""Default event chunker implementation (Task 5.3)."""

import hashlib

from lore_mcp.application.chunking.splitter import ChunkSplitter
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.entities import Event
from lore_mcp.domain.identifiers import generate_chunk_id
from lore_mcp.domain.provenance import Provenance
from lore_mcp.ports.chunking import EventChunker


class DefaultEventChunker(EventChunker):
    """Chunks narrative events keeping participants, location, and consequences."""

    def __init__(self, splitter: ChunkSplitter | None = None) -> None:
        self._splitter = splitter or ChunkSplitter()

    def chunk_event(
        self,
        event: Event,
        provenance: Provenance,
    ) -> list[Chunk]:
        """Convert an Event into one or more discrete chunks."""
        elements: list[str] = [event.description.strip()]

        if event.participants:
            elements.append(f"Participants: {', '.join(event.participants)}")
        if event.location:
            elements.append(f"Location: {event.location}")
        if event.consequences:
            elements.append(f"Consequences: {'; '.join(event.consequences)}")

        body = "\n\n".join(elem for elem in elements if elem)
        if not body:
            return []

        section_path = str(event.event_id)
        splits = self._splitter.split(
            content=body,
            heading=event.name,
            split_index_offset=0,
        )

        chunks: list[Chunk] = []
        for chunk_text, split_idx in splits:
            chunk_id = generate_chunk_id(
                series_id=event.series_id,
                source_locator=str(event.event_id),
                section_path=section_path,
                split_index=split_idx,
            )
            content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    series_id=event.series_id,
                    text=chunk_text,
                    season=event.season,
                    episode=event.episode,
                    scene_index=event.scene_index,
                    narrative_order=event.narrative_order,
                    schema_version=1,
                    ingest_version=1,
                    content_hash=content_hash,
                    entity_ids=list(event.participants),
                    entity_type="event",
                    primary_concept=event.name,
                    related_entities=[],
                    provenance=provenance,
                )
            )

        return chunks
