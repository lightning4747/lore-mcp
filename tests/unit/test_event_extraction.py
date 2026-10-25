"""Unit tests for DefaultEventExtractor (Task 4.3 & 4.5)."""

from datetime import UTC, datetime

import pytest

from lore_mcp.application.extraction.alias_builder import AliasTableBuilder
from lore_mcp.application.extraction.event_extractor import DefaultEventExtractor
from lore_mcp.domain.documents import DocumentSection, NormalizedDocument
from lore_mcp.domain.entities import Entity
from lore_mcp.domain.series_config import SeriesConfig, WikiEndpointConfig
from lore_mcp.domain.types import EntityId, SeriesId


@pytest.fixture
def series_config() -> SeriesConfig:
    return SeriesConfig(
        id=SeriesId("loki"),
        display_names=["Loki"],
        kind="show",
        aliases=["MCU Loki"],
        wiki_endpoints=[
            WikiEndpointConfig(
                api_url="https://marvel.fandom.com/api.php",
                wiki_type="fandom",
                license="CC-BY-SA-3.0",
            )
        ],
        category_to_entity_type={"Characters": "character", "Locations": "location"},
    )


@pytest.fixture
def sample_alias_table(series_config: SeriesConfig):
    builder = AliasTableBuilder()
    entities = [
        Entity(
            entity_id=EntityId("loki"),
            series_id=SeriesId("loki"),
            name="Loki Laufeyson",
            entity_type="character",
            aliases=["Loki", "God of Mischief"],
        ),
        Entity(
            entity_id=EntityId("mobius"),
            series_id=SeriesId("loki"),
            name="Mobius M. Mobius",
            entity_type="character",
            aliases=["Mobius", "Agent Mobius"],
        ),
        Entity(
            entity_id=EntityId("gobi-desert"),
            series_id=SeriesId("loki"),
            name="Gobi Desert",
            entity_type="location",
            aliases=["Mongolia"],
        ),
        Entity(
            entity_id=EntityId("tva-hq"),
            series_id=SeriesId("loki"),
            name="TVA Headquarters",
            entity_type="location",
            aliases=["Time Variance Authority"],
        ),
    ]
    return builder.build(series_config.id, entities)


def _create_episode_doc(
    title: str,
    sections: list[DocumentSection],
    season: int = 1,
    episode: int = 1,
) -> NormalizedDocument:
    return NormalizedDocument(
        series_id=SeriesId("loki"),
        source_url=f"https://marvel.fandom.com/wiki/{title.replace(' ', '_')}",
        source_type="fandom",
        title=title,
        text="\n\n".join(s.content for s in sections),
        sections=sections,
        content_hash="eventhash123",
        retrieval_timestamp=datetime.now(UTC),
        metadata={"season": str(season), "episode": str(episode)},
    )


@pytest.mark.unit
def test_plot_section_extraction(
    series_config: SeriesConfig, sample_alias_table
) -> None:
    extractor = DefaultEventExtractor()
    sections = [
        DocumentSection(
            heading="Plot",
            content=(
                "Loki is arrested by the TVA in the Gobi Desert.\n\n"
                "Mobius interrogates Loki in the Time Theater."
            ),
        )
    ]
    doc = _create_episode_doc("Glorious Purpose", sections)
    events = extractor.extract_events(doc, series_config, sample_alias_table)

    assert len(events) == 2
    assert "arrested" in events[0].name.lower() or "loki" in events[0].name.lower()
    assert events[0].scene_index == 1
    assert events[1].scene_index == 2


@pytest.mark.unit
def test_non_plot_sections_skipped(
    series_config: SeriesConfig, sample_alias_table
) -> None:
    extractor = DefaultEventExtractor()
    sections = [
        DocumentSection(
            heading="Cast & Characters", content="Tom Hiddleston plays Loki."
        ),
        DocumentSection(
            heading="Music", content="Soundtrack composed by Natalie Holt."
        ),
    ]
    doc = _create_episode_doc("Glorious Purpose", sections)
    events = extractor.extract_events(doc, series_config, sample_alias_table)

    assert len(events) == 0


@pytest.mark.unit
def test_participant_resolution(
    series_config: SeriesConfig, sample_alias_table
) -> None:
    extractor = DefaultEventExtractor()
    sections = [
        DocumentSection(
            heading="Synopsis",
            content="[[Loki]] attempts to reason with [[Agent Mobius]].",
        )
    ]
    doc = _create_episode_doc("Glorious Purpose", sections)
    events = extractor.extract_events(doc, series_config, sample_alias_table)

    assert len(events) == 1
    event = events[0]
    assert EntityId("loki") in event.participants
    assert EntityId("mobius") in event.participants


@pytest.mark.unit
def test_location_resolution(
    series_config: SeriesConfig, sample_alias_table
) -> None:
    extractor = DefaultEventExtractor()
    sections = [
        DocumentSection(
            heading="Plot",
            content="Loki awakens in the [[Gobi Desert]] after escaping.",
        )
    ]
    doc = _create_episode_doc("Glorious Purpose", sections)
    events = extractor.extract_events(doc, series_config, sample_alias_table)

    assert len(events) == 1
    assert events[0].location == "Gobi Desert"


@pytest.mark.unit
def test_episode_and_season_association(
    series_config: SeriesConfig, sample_alias_table
) -> None:
    extractor = DefaultEventExtractor()
    sections = [DocumentSection(heading="Plot", content="Loki arrives at the TVA.")]
    doc = _create_episode_doc("Glorious Purpose", sections, season=2, episode=4)
    events = extractor.extract_events(doc, series_config, sample_alias_table)

    assert len(events) == 1
    assert events[0].season == 2
    assert events[0].episode == 4


@pytest.mark.unit
def test_scene_index_increments(
    series_config: SeriesConfig, sample_alias_table
) -> None:
    extractor = DefaultEventExtractor()
    sections = [
        DocumentSection(
            heading="Plot",
            content="Scene one happens.\n\nScene two happens.\n\nScene three happens.",
        )
    ]
    doc = _create_episode_doc("Glorious Purpose", sections)
    events = extractor.extract_events(doc, series_config, sample_alias_table)

    assert len(events) == 3
    assert [e.scene_index for e in events] == [1, 2, 3]


@pytest.mark.unit
def test_deterministic_event_id(
    series_config: SeriesConfig, sample_alias_table
) -> None:
    extractor = DefaultEventExtractor()
    sections = [
        DocumentSection(
            heading="Plot",
            content="Loki is arrested by Hunter B-15 and Minutemen in the Gobi Desert.",
        )
    ]
    doc1 = _create_episode_doc("Glorious Purpose", sections)
    doc2 = _create_episode_doc("Glorious Purpose", sections)

    events1 = extractor.extract_events(doc1, series_config, sample_alias_table)
    events2 = extractor.extract_events(doc2, series_config, sample_alias_table)

    assert events1[0].event_id == events2[0].event_id
    assert "loki:s1e1:" in str(events1[0].event_id)


@pytest.mark.unit
def test_curated_narrative_order_loaded(
    series_config: SeriesConfig, sample_alias_table
) -> None:
    extractor = DefaultEventExtractor()
    sections = [
        DocumentSection(
            heading="Plot",
            content="Loki is arrested by Hunter B-15 in the Gobi Desert.",
        )
    ]
    doc = _create_episode_doc("Glorious Purpose", sections)
    events = extractor.extract_events(doc, series_config, sample_alias_table)
    event_id_str = str(events[0].event_id)

    curated = {event_id_str: 100}
    events_with_curated = extractor.extract_events(
        doc, series_config, sample_alias_table, curated_narrative_orders=curated
    )

    assert events_with_curated[0].narrative_order == 100


@pytest.mark.unit
def test_null_narrative_order_when_not_curated(
    series_config: SeriesConfig, sample_alias_table
) -> None:
    extractor = DefaultEventExtractor()
    sections = [
        DocumentSection(heading="Plot", content="Loki enters the TVA elevator.")
    ]
    doc = _create_episode_doc("Glorious Purpose", sections)

    # Empty curated dict
    events = extractor.extract_events(
        doc, series_config, sample_alias_table, curated_narrative_orders={}
    )
    assert events[0].narrative_order is None

    # None curated dict
    events_none = extractor.extract_events(
        doc, series_config, sample_alias_table, curated_narrative_orders=None
    )
    assert events_none[0].narrative_order is None


@pytest.mark.unit
def test_no_plot_section_no_events(
    series_config: SeriesConfig, sample_alias_table
) -> None:
    extractor = DefaultEventExtractor()
    doc = _create_episode_doc("Glorious Purpose", [])
    events = extractor.extract_events(doc, series_config, sample_alias_table)
    assert events == []
