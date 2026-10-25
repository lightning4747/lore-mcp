"""Unit tests for ManifestBuilder (Task 4.4 & 4.5)."""

import pytest

from lore_mcp.application.extraction.manifest_builder import ManifestBuilder
from lore_mcp.domain.entities import Entity, Event, SeriesManifest
from lore_mcp.domain.series_config import SeriesConfig, WikiEndpointConfig
from lore_mcp.domain.types import EntityId, EventId, SeriesId


@pytest.fixture
def series_config() -> SeriesConfig:
    return SeriesConfig(
        id=SeriesId("loki"),
        display_names=["Loki", "Loki (TV Series)"],
        kind="show",
        aliases=["MCU Loki", "Marvel's Loki"],
        wiki_endpoints=[
            WikiEndpointConfig(
                api_url="https://marvel.fandom.com/api.php",
                wiki_type="fandom",
                license="CC-BY-SA-3.0",
            )
        ],
        category_to_entity_type={"Characters": "character"},
    )


def _create_entity(entity_id: str, name: str) -> Entity:
    return Entity(
        entity_id=EntityId(entity_id),
        series_id=SeriesId("loki"),
        name=name,
        entity_type="character",
        summary="Summary",
    )


def _create_event(event_id: str, season: int, episode: int) -> Event:
    return Event(
        event_id=EventId(event_id),
        series_id=SeriesId("loki"),
        name=f"Event {event_id}",
        description="Event description",
        season=season,
        episode=episode,
    )


@pytest.mark.unit
def test_manifest_fields_from_config(series_config: SeriesConfig) -> None:
    builder = ManifestBuilder()
    manifest = builder.build(series_config, entities=[], events=[])

    assert manifest.series_id == SeriesId("loki")
    assert manifest.display_name == "Loki"
    assert manifest.kind == "show"
    assert manifest.aliases == ["MCU Loki", "Marvel's Loki"]


@pytest.mark.unit
def test_entity_count_in_manifest(series_config: SeriesConfig) -> None:
    builder = ManifestBuilder()
    entities = [
        _create_entity("loki", "Loki"),
        _create_entity("mobius", "Mobius"),
        _create_entity("sylvie", "Sylvie"),
    ]
    manifest = builder.build(series_config, entities=entities)

    assert manifest.entity_count == 3


@pytest.mark.unit
def test_seasons_derived_from_events(series_config: SeriesConfig) -> None:
    builder = ManifestBuilder()
    events = [
        _create_event("loki:s1e1:event1", season=1, episode=1),
        _create_event("loki:s1e2:event2", season=1, episode=2),
        _create_event("loki:s2e1:event3", season=2, episode=1),
    ]
    manifest = builder.build(series_config, entities=[], events=events)

    assert manifest.seasons == [1, 2]


@pytest.mark.unit
def test_ingest_version_starts_at_one(series_config: SeriesConfig) -> None:
    builder = ManifestBuilder()
    manifest = builder.build(series_config, entities=[])

    assert manifest.ingest_version == 1


@pytest.mark.unit
def test_ingest_version_incremented(series_config: SeriesConfig) -> None:
    builder = ManifestBuilder()
    prev = SeriesManifest(
        series_id=SeriesId("loki"),
        display_name="Loki",
        kind="show",
        aliases=[],
        seasons=[1],
        entity_count=2,
        chunk_count=25,
        ingest_version=3,
        schema_version=1,
    )
    manifest = builder.build(
        series_config,
        entities=[_create_entity("loki", "Loki")],
        previous_manifest=prev,
    )

    assert manifest.ingest_version == 4
    # Preserves chunk_count from previous manifest
    assert manifest.chunk_count == 25


@pytest.mark.unit
def test_schema_version_applied(series_config: SeriesConfig) -> None:
    builder = ManifestBuilder()
    manifest = builder.build(series_config, entities=[], schema_version=2)

    assert manifest.schema_version == 2
