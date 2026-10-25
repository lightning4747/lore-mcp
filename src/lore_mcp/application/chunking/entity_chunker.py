"""Default entity chunker implementation (Task 5.2)."""

import hashlib
from collections import defaultdict

from lore_mcp.application.chunking.splitter import ChunkSplitter
from lore_mcp.domain.documents import Chunk, DocumentSection, NormalizedDocument
from lore_mcp.domain.entities import Entity
from lore_mcp.domain.identifiers import generate_chunk_id
from lore_mcp.domain.provenance import Provenance
from lore_mcp.ports.chunking import EntityChunker


class DefaultEntityChunker(EntityChunker):
    """Chunks normalized entity wiki documents around meaningful lore units."""

    def __init__(
        self,
        splitter: ChunkSplitter | None = None,
        min_section_tokens: int = 20,
    ) -> None:
        self._splitter = splitter or ChunkSplitter()
        self._min_section_tokens = min_section_tokens

    def _coalesce_sections(
        self, sections: list[DocumentSection]
    ) -> list[DocumentSection]:
        """Merge short sections under min_section_tokens into adjacent sections."""
        if not sections:
            return []

        merged: list[DocumentSection] = []
        accumulator_heading: str | None = None
        accumulator_content: list[str] = []

        for sec in sections:
            content = sec.content.strip()
            heading = sec.heading.strip()
            if not content:
                continue

            tokens = self._splitter.estimate_tokens(content)
            if tokens < self._min_section_tokens:
                if accumulator_heading is None:
                    accumulator_heading = heading
                elif heading:
                    accumulator_heading = f"{accumulator_heading} - {heading}"
                accumulator_content.append(content)
            else:
                if accumulator_content:
                    # Flush accumulated short content before this normal section
                    comb_heading = accumulator_heading or heading
                    comb_content = "\n\n".join(accumulator_content)
                    merged.append(
                        DocumentSection(
                            heading=comb_heading,
                            content=comb_content,
                        )
                    )
                    accumulator_heading = None
                    accumulator_content = []

                merged.append(DocumentSection(heading=heading, content=content))

        if accumulator_content:
            comb_content = "\n\n".join(accumulator_content)
            comb_heading = accumulator_heading or ""
            if merged:
                # Merge trailing short section with previous section
                prev = merged[-1]
                trailing_heading = (
                    f"{prev.heading} - {comb_heading}"
                    if prev.heading and comb_heading
                    else (prev.heading or comb_heading)
                )
                merged[-1] = DocumentSection(
                    heading=trailing_heading,
                    content=f"{prev.content}\n\n{comb_content}",
                )
            else:
                merged.append(
                    DocumentSection(heading=comb_heading, content=comb_content)
                )

        return merged

    def chunk_entity(
        self,
        document: NormalizedDocument,
        entity: Entity,
    ) -> list[Chunk]:
        """Split a normalized entity document into discrete chunks."""
        sections = document.sections
        if not sections:
            if not document.text.strip():
                return []
            sections = [DocumentSection(heading="", content=document.text.strip())]

        coalesced = self._coalesce_sections(sections)
        chunks: list[Chunk] = []
        path_counters: dict[str, int] = defaultdict(int)

        provenance = Provenance(
            source_type=document.source_type,
            source_url=document.source_url,
            retrieval_timestamp=document.retrieval_timestamp,
            license=document.metadata.get("license"),
            attribution_text=document.metadata.get("attribution"),
            revision_id=document.metadata.get("revision_id"),
        )

        for sec in coalesced:
            section_path = sec.heading or document.title or "main"
            offset = path_counters[section_path]
            splits = self._splitter.split(
                content=sec.content,
                heading=sec.heading,
                split_index_offset=offset,
            )
            for chunk_text, split_idx in splits:
                path_counters[section_path] = split_idx + 1
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
                        season=None,
                        episode=None,
                        scene_index=None,
                        narrative_order=None,
                        schema_version=1,
                        ingest_version=1,
                        content_hash=content_hash,
                        entity_ids=[entity.entity_id],
                        entity_type=entity.entity_type,
                        primary_concept=entity.name,
                        related_entities=list(entity.related_entities),
                        provenance=provenance,
                    )
                )

        return chunks
