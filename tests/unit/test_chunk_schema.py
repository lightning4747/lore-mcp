"""Unit tests for Chunk domain model schema and validation (Task 5.1)."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.types import ChunkId, EntityId, SeriesId


@pytest.mark.unit
class TestChunkSchema:
    """Verifies Chunk domain model structure, immutability, and defaults."""

    @pytest.fixture
    def sample_provenance(self) -> Provenance:
        return Provenance(
            source_type="fandom",
            source_url="https://example.com/wiki/Loki",
            retrieval_timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            license="CC-BY-SA-3.0",
        )

    def test_chunk_with_all_fields_validates(
        self, sample_provenance: Provenance
    ) -> None:
        chunk = Chunk(
            chunk_id=ChunkId("chunk-123"),
            series_id=SeriesId("loki"),
            text="Loki Laufeyson was the God of Mischief.",
            season=1,
            episode=1,
            scene_index=3,
            narrative_order=10,
            schema_version=1,
            ingest_version=1,
            content_hash="abc123hash",
            entity_ids=[EntityId("loki")],
            entity_type="character",
            primary_concept="Loki",
            related_entities=[EntityId("thor"), EntityId("odin")],
            provenance=sample_provenance,
        )

        assert chunk.chunk_id == "chunk-123"
        assert chunk.series_id == "loki"
        assert chunk.season == 1
        assert chunk.episode == 1
        assert chunk.scene_index == 3
        assert chunk.narrative_order == 10
        assert chunk.entity_type == "character"
        assert chunk.primary_concept == "Loki"
        assert chunk.related_entities == [EntityId("thor"), EntityId("odin")]
        assert chunk.entity_ids == [EntityId("loki")]
        assert chunk.provenance.source_url == "https://example.com/wiki/Loki"

    def test_chunk_field_defaults(self, sample_provenance: Provenance) -> None:
        chunk = Chunk(
            chunk_id=ChunkId("chunk-default"),
            series_id=SeriesId("loki"),
            text="Short text.",
            content_hash="hash-default",
            provenance=sample_provenance,
        )

        assert chunk.season is None
        assert chunk.episode is None
        assert chunk.scene_index is None
        assert chunk.narrative_order is None
        assert chunk.schema_version == 1
        assert chunk.ingest_version == 1
        assert chunk.entity_ids == []
        assert chunk.entity_type == ""
        assert chunk.primary_concept == ""
        assert chunk.related_entities == []

    def test_chunk_is_immutable(self, sample_provenance: Provenance) -> None:
        chunk = Chunk(
            chunk_id=ChunkId("chunk-frozen"),
            series_id=SeriesId("loki"),
            text="Immutable content.",
            content_hash="frozen-hash",
            provenance=sample_provenance,
        )

        with pytest.raises(ValidationError):
            chunk.text = "Mutated text"  # type: ignore[reportAttributeAccessIssue]

    def test_chunk_extra_fields_forbidden(self, sample_provenance: Provenance) -> None:
        with pytest.raises(ValidationError):
            Chunk(
                chunk_id=ChunkId("chunk-extra"),
                series_id=SeriesId("loki"),
                text="Text",
                content_hash="hash",
                provenance=sample_provenance,
                extra_unknown_field="not_allowed",  # type: ignore[call-arg]
            )

    def test_chunk_invalid_series_id_rejected(
        self, sample_provenance: Provenance
    ) -> None:
        with pytest.raises(ValidationError):
            Chunk(
                chunk_id=ChunkId("chunk-invalid"),
                series_id="INVALID_SERIES!",  # type: ignore[arg-type]
                text="Text",
                content_hash="hash",
                provenance=sample_provenance,
            )
