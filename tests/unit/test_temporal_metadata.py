"""Unit tests for TemporalMetadata domain model and confidence validation (Task 9.1)."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from lore_mcp.domain.chronology import Evidence, TemporalMetadata
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.entities import Event
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.types import ChunkId, EventId, SeriesId


@pytest.mark.unit
class TestTemporalMetadata:
    """Verifies temporal metadata representation, confidence validation, and helpers."""

    @pytest.fixture
    def sample_provenance(self) -> Provenance:
        return Provenance(
            source_type="fandom",
            source_url="https://marvel.fandom.com/wiki/Loki",
            retrieval_timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        )

    def test_temporal_metadata_with_all_fields(self) -> None:
        meta = TemporalMetadata(
            scene_index=4,
            season=1,
            episode=2,
            narrative_order=150,
            narrative_confidence=0.95,
        )

        assert meta.scene_index == 4
        assert meta.season == 1
        assert meta.episode == 2
        assert meta.narrative_order == 150
        assert meta.narrative_confidence == 0.95

    def test_temporal_metadata_defaults(self) -> None:
        meta = TemporalMetadata()
        assert meta.scene_index is None
        assert meta.season is None
        assert meta.episode is None
        assert meta.narrative_order is None
        assert meta.narrative_confidence is None

    def test_confidence_validation_bounds(self) -> None:
        # Valid bounds
        assert TemporalMetadata(narrative_confidence=0.0).narrative_confidence == 0.0
        assert TemporalMetadata(narrative_confidence=1.0).narrative_confidence == 1.0

        # Out of bounds
        with pytest.raises(ValidationError):
            TemporalMetadata(narrative_confidence=-0.1)

        with pytest.raises(ValidationError):
            TemporalMetadata(narrative_confidence=1.1)

    def test_evidence_temporal_metadata_helper(
        self, sample_provenance: Provenance
    ) -> None:
        ev = Evidence(
            chunk_id=ChunkId("chunk-1"),
            series_id=SeriesId("loki"),
            content="Evidence content",
            season=2,
            episode=1,
            scene_index=3,
            narrative_order=200,
            narrative_confidence=0.85,
            score=0.9,
            provenance=sample_provenance,
        )

        meta = ev.temporal_metadata()
        assert meta.season == 2
        assert meta.episode == 1
        assert meta.scene_index == 3
        assert meta.narrative_order == 200
        assert meta.narrative_confidence == 0.85

    def test_chunk_and_event_support_narrative_confidence(
        self, sample_provenance: Provenance
    ) -> None:
        chunk = Chunk(
            chunk_id=ChunkId("chunk-1"),
            series_id=SeriesId("loki"),
            text="Chunk text",
            content_hash="mock-hash",
            narrative_order=50,
            narrative_confidence=0.9,
            provenance=sample_provenance,
        )
        assert chunk.narrative_order == 50
        assert chunk.narrative_confidence == 0.9

        event = Event(
            event_id=EventId("event-1"),
            series_id=SeriesId("loki"),
            name="Timeline Bombing",
            description="Sylvie bombs timeline",
            narrative_order=75,
            narrative_confidence=0.95,
        )
        assert event.narrative_order == 75
        assert event.narrative_confidence == 0.95
