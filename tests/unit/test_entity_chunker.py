"""Unit tests for DefaultEntityChunker (Task 5.2)."""

from datetime import UTC, datetime

import pytest

from lore_mcp.application.chunking.entity_chunker import DefaultEntityChunker
from lore_mcp.application.chunking.splitter import ChunkSplitter
from lore_mcp.domain.documents import DocumentSection, NormalizedDocument
from lore_mcp.domain.entities import Entity
from lore_mcp.domain.types import EntityId, SeriesId


@pytest.mark.unit
class TestEntityChunker:
    """Verifies entity chunk boundaries, section merging, and metadata preservation."""

    @pytest.fixture
    def sample_entity(self) -> Entity:
        return Entity(
            entity_id=EntityId("loki-laufeyson"),
            series_id=SeriesId("loki"),
            name="Loki",
            entity_type="character",
            aliases=["God of Mischief"],
            summary="Prince of Asgard.",
            related_entities=[EntityId("thor"), EntityId("mobius")],
        )

    @pytest.fixture
    def entity_document(self) -> NormalizedDocument:
        return NormalizedDocument(
            series_id=SeriesId("loki"),
            source_url="https://marvel.fandom.com/wiki/Loki",
            source_type="fandom",
            title="Loki Laufeyson",
            text="Loki full body text.",
            sections=[
                DocumentSection(
                    heading="Biography",
                    content="Loki was born in Jotunheim as the son of Laufey.",
                ),
                DocumentSection(
                    heading="Powers and Abilities",
                    content=(
                        "Loki possesses superhuman strength, illusion projection, "
                        "and master manipulation skills honed over centuries."
                    ),
                ),
            ],
            content_hash="mock-hash",
            retrieval_timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            metadata={"license": "CC-BY-SA-3.0", "attribution": "Fandom Contributors"},
        )

    def test_single_section_document(self, sample_entity: Entity) -> None:
        doc = NormalizedDocument(
            series_id=SeriesId("loki"),
            source_url="https://marvel.fandom.com/wiki/Loki",
            source_type="fandom",
            title="Loki",
            text="Short synopsis.",
            sections=[
                DocumentSection(
                    heading="Overview",
                    content=(
                        "Loki is a complex character who alternates between "
                        "villain and antihero throughout the timeline."
                    ),
                )
            ],
            content_hash="single-hash",
            retrieval_timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        )
        chunker = DefaultEntityChunker(min_section_tokens=5)
        chunks = chunker.chunk_entity(doc, sample_entity)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.series_id == "loki"
        assert chunk.entity_ids == [EntityId("loki-laufeyson")]
        assert chunk.entity_type == "character"
        assert chunk.primary_concept == "Loki"
        assert chunk.related_entities == [EntityId("thor"), EntityId("mobius")]
        assert chunk.provenance.license is None
        assert "Overview\n\nLoki is a complex character" in chunk.text

    def test_multi_section_document_preserves_sections(
        self, sample_entity: Entity, entity_document: NormalizedDocument
    ) -> None:
        chunker = DefaultEntityChunker(min_section_tokens=5)
        chunks = chunker.chunk_entity(entity_document, sample_entity)

        assert len(chunks) == 2
        assert "Biography" in chunks[0].text
        assert "Powers and Abilities" in chunks[1].text
        assert chunks[0].provenance.license == "CC-BY-SA-3.0"
        assert chunks[1].provenance.attribution_text == "Fandom Contributors"

    def test_coalesces_short_sections(self, sample_entity: Entity) -> None:
        doc = NormalizedDocument(
            series_id=SeriesId("loki"),
            source_url="https://marvel.fandom.com/wiki/Loki",
            source_type="fandom",
            title="Loki",
            text="Combined",
            sections=[
                DocumentSection(heading="Aliases", content="God of Mischief."),
                DocumentSection(heading="Status", content="Alive in multiverse."),
                DocumentSection(
                    heading="Full Biography",
                    content=(
                        "Loki Laufeyson was the biological son of King Laufey, "
                        "the ruler of the Frost Giants of Jotunheim, who was abandoned "
                        "and left to die shortly after his birth."
                    ),
                ),
            ],
            content_hash="short-hash",
            retrieval_timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        )
        # min_section_tokens is 20, so the first two sections (3 and 4 tokens) merge
        chunker = DefaultEntityChunker(min_section_tokens=20)
        chunks = chunker.chunk_entity(doc, sample_entity)

        # The short sections coalesce into one, followed by Full Biography
        assert len(chunks) == 2
        assert "Aliases - Status" in chunks[0].text
        assert "God of Mischief." in chunks[0].text
        assert "Alive in multiverse." in chunks[0].text
        assert "Full Biography" in chunks[1].text

    def test_oversized_section_split_deterministically(
        self, sample_entity: Entity
    ) -> None:
        small_splitter = ChunkSplitter(max_tokens=40)
        long_sentences = [
            f"Sentence {i} gives deep detailed context regarding the glorious purpose."
            for i in range(1, 10)
        ]
        doc = NormalizedDocument(
            series_id=SeriesId("loki"),
            source_url="https://marvel.fandom.com/wiki/Loki",
            source_type="fandom",
            title="Loki",
            text="Long text",
            sections=[
                DocumentSection(
                    heading="History",
                    content=" ".join(long_sentences),
                )
            ],
            content_hash="long-hash",
            retrieval_timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        )
        chunker = DefaultEntityChunker(splitter=small_splitter)
        chunks = chunker.chunk_entity(doc, sample_entity)

        assert len(chunks) >= 2
        # Check all chunk IDs are unique
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
        for c in chunks:
            assert c.text.startswith("History\n\n")

    def test_empty_document_returns_empty_list(self, sample_entity: Entity) -> None:
        doc = NormalizedDocument(
            series_id=SeriesId("loki"),
            source_url="https://marvel.fandom.com/wiki/Empty",
            source_type="fandom",
            title="Empty",
            text="",
            sections=[],
            content_hash="empty-hash",
            retrieval_timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        )
        chunker = DefaultEntityChunker()
        assert chunker.chunk_entity(doc, sample_entity) == []
