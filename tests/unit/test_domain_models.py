"""Unit tests for domain shared data models and identifiers."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from lore_mcp.domain import (
    Chunk,
    ChunkId,
    DocumentSection,
    Entity,
    EntityId,
    Event,
    EventId,
    Evidence,
    EvidenceBundle,
    NormalizedDocument,
    Provenance,
    RawDocument,
    SearchHit,
    SearchQuery,
    SeriesId,
    validate_series_id,
)


@pytest.fixture
def sample_provenance() -> Provenance:
    return Provenance(
        source_type="fandom",
        source_url="https://marvel.fandom.com/wiki/Loki",
        license="CC-BY-SA-3.0",
        attribution_text="Marvel Wiki Contributors",
        retrieval_timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        revision_id="rev-12345",
    )


@pytest.fixture
def sample_chunk(sample_provenance: Provenance) -> Chunk:
    return Chunk(
        chunk_id=ChunkId("chunk-s2e6-loom-1"),
        series_id=SeriesId("loki"),
        text="Loki uses temporal slipping to return to the TVA control room.",
        season=2,
        episode=6,
        scene_index=14,
        narrative_order=108,
        schema_version=1,
        ingest_version=2,
        content_hash="sha256:abc123def456",
        entity_ids=[EntityId("loki"), EntityId("tva")],
        provenance=sample_provenance,
    )


@pytest.mark.unit
def test_series_id_slug_validation() -> None:
    # Valid slugs
    assert validate_series_id("loki") == SeriesId("loki")
    assert validate_series_id("star-wars_clone-wars") == SeriesId(
        "star-wars_clone-wars"
    )
    assert validate_series_id("  ATTACK-ON-TITAN  ") == SeriesId("attack-on-titan")

    # Invalid slugs
    with pytest.raises(ValueError, match="Invalid SeriesId slug"):
        validate_series_id("")

    with pytest.raises(ValueError, match="Invalid SeriesId slug"):
        validate_series_id("invalid slug with spaces")

    with pytest.raises(ValueError, match="Invalid SeriesId slug"):
        validate_series_id("loki@timeline!")

    with pytest.raises(TypeError, match="SeriesId must be a string"):
        validate_series_id(12345)  # type: ignore[arg-type]


@pytest.mark.unit
def test_provenance_immutability(sample_provenance: Provenance) -> None:
    assert sample_provenance.source_type == "fandom"
    with pytest.raises(ValidationError):
        sample_provenance.source_type = "wikipedia"  # type: ignore[misc]


@pytest.mark.unit
def test_raw_and_normalized_document_models() -> None:
    now = datetime.now(UTC)
    raw = RawDocument(
        series_id=SeriesId("loki"),
        source_url="https://marvel.fandom.com/wiki/Temporal_Loom",
        source_type="fandom",
        content="== Overview ==\nThe Temporal Loom manages raw time.",
        retrieval_timestamp=now,
        metadata={"author": "editor1"},
    )
    assert raw.series_id == "loki"
    assert raw.metadata["author"] == "editor1"

    section = DocumentSection(
        heading="Overview",
        content="The Temporal Loom manages raw time.",
    )
    norm = NormalizedDocument(
        series_id=SeriesId("loki"),
        source_url="https://marvel.fandom.com/wiki/Temporal_Loom",
        source_type="fandom",
        title="Temporal Loom",
        text="Overview: The Temporal Loom manages raw time.",
        sections=[section],
        content_hash="hash-loom-99",
        retrieval_timestamp=now,
    )
    assert norm.title == "Temporal Loom"
    assert len(norm.sections) == 1
    assert norm.sections[0].heading == "Overview"


@pytest.mark.unit
def test_chunk_nullable_fields_and_versions(sample_chunk: Chunk) -> None:
    assert sample_chunk.season == 2
    assert sample_chunk.episode == 6
    assert sample_chunk.scene_index == 14
    assert sample_chunk.narrative_order == 108
    assert sample_chunk.schema_version == 1
    assert sample_chunk.ingest_version == 2
    assert sample_chunk.content_hash == "sha256:abc123def456"

    # Minimal chunk with nullables as None
    minimal = Chunk(
        chunk_id=ChunkId("chunk-unscoped-1"),
        series_id=SeriesId("loki"),
        text="General lore about the Multiverse.",
        content_hash="hash-min-0",
        provenance=sample_chunk.provenance,
    )
    assert minimal.season is None
    assert minimal.episode is None
    assert minimal.scene_index is None
    assert minimal.narrative_order is None
    assert minimal.entity_ids == []


@pytest.mark.unit
def test_entity_and_event_models() -> None:
    entity = Entity(
        entity_id=EntityId("he-who-remains"),
        series_id=SeriesId("loki"),
        name="He Who Remains",
        entity_type="character",
        aliases=["Kang Variant", "Creator of TVA"],
        summary="Creator and protector of the Sacred Timeline.",
        related_entities=[EntityId("tva"), EntityId("temporal-loom")],
    )
    assert entity.entity_id == "he-who-remains"
    assert len(entity.aliases) == 2

    event = Event(
        event_id=EventId("event-loom-explosion"),
        series_id=SeriesId("loki"),
        name="Destruction of the Temporal Loom",
        description="The Temporal Loom overloads and explodes, releasing raw time.",
        season=2,
        episode=4,
        scene_index=22,
        narrative_order=95,
        participants=[EntityId("loki"), EntityId("victor-timely")],
        location="TVA Control Room",
        consequences=["Temporal radiation blast", "Spaghetti effect"],
    )
    assert event.event_id == "event-loom-explosion"
    assert len(event.participants) == 2
    assert len(event.consequences) == 2


@pytest.mark.unit
def test_search_query_requires_series() -> None:
    # Valid query
    query = SearchQuery(series_id=SeriesId("loki"), query="Temporal Loom failsafe")
    assert query.series_id == "loki"
    assert query.max_results == 8
    assert query.include_unscoped is False

    # Missing series_id raises validation error
    with pytest.raises(ValidationError):
        SearchQuery(query="Temporal Loom failsafe")  # type: ignore[call-arg]


@pytest.mark.unit
def test_search_hit_per_signal_scores(sample_chunk: Chunk) -> None:
    hit = SearchHit(
        chunk=sample_chunk,
        score=0.88,
        dense_score=0.85,
        sparse_score=0.72,
        rerank_score=0.91,
    )
    assert hit.score == 0.88
    assert hit.dense_score == 0.85
    assert hit.sparse_score == 0.72
    assert hit.rerank_score == 0.91


@pytest.mark.unit
def test_evidence_and_evidence_bundle(sample_provenance: Provenance) -> None:
    evidence = Evidence(
        chunk_id=ChunkId("chunk-s2e6-1"),
        series_id=SeriesId("loki"),
        content="Victor Timely volunteers to approach the throughput multiplier.",
        entity_id=EntityId("victor-timely"),
        season=2,
        episode=4,
        scene_index=18,
        narrative_order=92,
        score=0.94,
        dense_score=0.92,
        sparse_score=0.80,
        rerank_score=0.96,
        provenance=sample_provenance,
    )

    bundle = EvidenceBundle(
        query="Who approaches the multiplier?",
        series_id=SeriesId("loki"),
        status="ok",
        evidence=[evidence],
        truncated=False,
        chronology=["Scene 17 -> Scene 18"],
        notices=["Cached result"],
    )

    assert bundle.status == "ok"
    assert len(bundle.evidence) == 1
    assert bundle.evidence[0].entity_id == "victor-timely"
    assert bundle.notices == ["Cached result"]


@pytest.mark.unit
def test_json_roundtrip(sample_chunk: Chunk) -> None:
    json_data = sample_chunk.model_dump_json()
    reconstituted = Chunk.model_validate_json(json_data)
    assert reconstituted == sample_chunk
