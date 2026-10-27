"""Unit tests for DefaultEventChunker (Task 5.3)."""

from datetime import UTC, datetime

import pytest

from lore_mcp.application.chunking.event_chunker import DefaultEventChunker
from lore_mcp.application.chunking.splitter import ChunkSplitter
from lore_mcp.domain.entities import Event
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.types import EntityId, EventId, SeriesId


@pytest.mark.unit
class TestEventChunker:
    """Verifies event chunking, participant link, and chronological coordinates."""

    @pytest.fixture
    def sample_provenance(self) -> Provenance:
        return Provenance(
            source_type="fandom",
            source_url="https://marvel.fandom.com/wiki/Battle_of_New_York",
            retrieval_timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            license="CC-BY-SA-3.0",
        )

    @pytest.fixture
    def sample_event(self) -> Event:
        return Event(
            event_id=EventId("loki:s1e1:glorious-purpose"),
            series_id=SeriesId("loki"),
            name="Arrest by the TVA",
            description=(
                "Loki is apprehended by Minutemen after escaping with the Tesseract."
            ),
            season=1,
            episode=1,
            scene_index=2,
            narrative_order=105,
            participants=[EntityId("loki"), EntityId("hunter-b-15")],
            location="Gobi Desert, Mongolia",
            consequences=["Tesseract confiscated", "Brought to TVA headquarters"],
        )

    def test_event_chunk_preserves_attributes(
        self, sample_event: Event, sample_provenance: Provenance
    ) -> None:
        chunker = DefaultEventChunker()
        chunks = chunker.chunk_event(sample_event, sample_provenance)

        assert len(chunks) == 1
        c = chunks[0]
        assert c.series_id == "loki"
        assert c.entity_type == "event"
        assert c.primary_concept == "Arrest by the TVA"
        assert c.entity_ids == [EntityId("loki"), EntityId("hunter-b-15")]
        assert c.season == 1
        assert c.episode == 1
        assert c.scene_index == 2
        assert c.narrative_order == 105
        assert "Participants: loki, hunter-b-15" in c.text
        assert "Location: Gobi Desert, Mongolia" in c.text
        assert (
            "Consequences: Tesseract confiscated; Brought to TVA headquarters" in c.text
        )
        assert c.provenance.source_url == sample_provenance.source_url

    def test_oversized_event_splits_with_heading(
        self, sample_event: Event, sample_provenance: Provenance
    ) -> None:
        small_splitter = ChunkSplitter(max_tokens=30)
        chunker = DefaultEventChunker(splitter=small_splitter)
        chunks = chunker.chunk_event(sample_event, sample_provenance)

        assert len(chunks) >= 2
        for c in chunks:
            assert c.text.startswith("Arrest by the TVA\n\n")
            assert c.season == 1
            assert c.episode == 1
            assert c.narrative_order == 105

    def test_event_with_no_description_or_details(
        self, sample_provenance: Provenance
    ) -> None:
        empty_event = Event(
            event_id=EventId("loki:e1:empty"),
            series_id=SeriesId("loki"),
            name="Empty Event",
            description="",
        )
        chunker = DefaultEventChunker()
        assert chunker.chunk_event(empty_event, sample_provenance) == []

    def test_event_chunk_deterministic_ids(
        self, sample_event: Event, sample_provenance: Provenance
    ) -> None:
        chunker1 = DefaultEventChunker()
        chunker2 = DefaultEventChunker()

        chunks1 = chunker1.chunk_event(sample_event, sample_provenance)
        chunks2 = chunker2.chunk_event(sample_event, sample_provenance)

        assert len(chunks1) == len(chunks2)
        assert chunks1[0].chunk_id == chunks2[0].chunk_id
        assert chunks1[0].content_hash == chunks2[0].content_hash
