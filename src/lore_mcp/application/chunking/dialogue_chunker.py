"""Default dialogue chunker implementation (Task 5.3)."""

import hashlib

from pydantic import TypeAdapter

from lore_mcp.application.chunking.splitter import ChunkSplitter
from lore_mcp.domain.documents import Chunk, DialogueSegment, NormalizedDocument
from lore_mcp.domain.identifiers import generate_chunk_id
from lore_mcp.domain.provenance import Provenance
from lore_mcp.ports.chunking import DialogueChunker

_SEGMENTS_ADAPTER: TypeAdapter[list[DialogueSegment]] = TypeAdapter(
    list[DialogueSegment]
)


class DefaultDialogueChunker(DialogueChunker):
    """Chunks dialogue transcripts by scene windows with episode metadata."""

    def __init__(self, splitter: ChunkSplitter | None = None) -> None:
        self._splitter = splitter or ChunkSplitter()

    def chunk_dialogue(
        self,
        document: NormalizedDocument,
    ) -> list[Chunk]:
        """Split a normalized dialogue transcript document into scene chunks."""
        if "dialogue_segments" not in document.metadata:
            raise ValueError(
                f"Document '{document.source_url}' is missing "
                "'dialogue_segments' metadata"
            )

        raw_segments = document.metadata["dialogue_segments"]
        segments = _SEGMENTS_ADAPTER.validate_json(raw_segments)
        if not segments:
            return []

        season_val: int | None = None
        if "season" in document.metadata:
            try:
                season_val = int(document.metadata["season"])
            except ValueError:
                pass

        episode_val: int | None = None
        if "episode" in document.metadata:
            try:
                episode_val = int(document.metadata["episode"])
            except ValueError:
                pass

        provenance = Provenance(
            source_type=document.source_type,
            source_url=document.source_url,
            retrieval_timestamp=document.retrieval_timestamp,
            license=document.metadata.get("license"),
            attribution_text=document.metadata.get("attribution"),
            revision_id=document.metadata.get("revision_id"),
        )

        chunks: list[Chunk] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue

            section_path = f"scene-{segment.scene_index}"
            splits = self._splitter.split(
                content=text,
                heading="",
                split_index_offset=0,
            )

            for chunk_text, split_idx in splits:
                chunk_id = generate_chunk_id(
                    series_id=document.series_id,
                    source_locator=document.source_url,
                    section_path=section_path,
                    split_index=split_idx,
                )
                content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        series_id=document.series_id,
                        text=chunk_text,
                        season=season_val,
                        episode=episode_val,
                        scene_index=segment.scene_index,
                        narrative_order=None,
                        schema_version=1,
                        ingest_version=1,
                        content_hash=content_hash,
                        entity_ids=[],
                        entity_type="dialogue",
                        primary_concept=document.title,
                        related_entities=[],
                        provenance=provenance,
                    )
                )

        return chunks
