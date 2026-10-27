"""Unit tests for DefaultEntityExtractor (Task 4.1, 4.2 & 4.5)."""

import json
from datetime import UTC, datetime

import pytest

from lore_mcp.application.extraction.entity_extractor import DefaultEntityExtractor
from lore_mcp.domain.documents import DocumentSection, NormalizedDocument
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
        category_to_entity_type={
            "Characters": "character",
            "Locations": "location",
            "Devices": "device",
            "Organizations": "organization",
            "Events": "event",
            "Abilities": "ability",
            "Concepts": "concept",
        },
    )


def _create_norm_doc(
    title: str,
    text: str,
    categories: list[str] | None = None,
    redirects: list[str] | None = None,
    links: list[str] | None = None,
    extra_meta: dict[str, str] | None = None,
) -> NormalizedDocument:
    meta = {
        "title": title,
        "categories": json.dumps(categories or []),
        "redirects": json.dumps(redirects or []),
        "links": json.dumps(links or []),
    }
    if extra_meta:
        meta.update(extra_meta)

    return NormalizedDocument(
        series_id=SeriesId("loki"),
        source_url=f"https://marvel.fandom.com/wiki/{title.replace(' ', '_')}",
        source_type="fandom",
        title=title,
        text=text,
        sections=[DocumentSection(heading="Overview", content=text)],
        content_hash="abc123hash",
        retrieval_timestamp=datetime.now(UTC),
        metadata=meta,
    )


@pytest.mark.unit
def test_category_type_mapping_character(series_config: SeriesConfig) -> None:
    extractor = DefaultEntityExtractor()
    doc = _create_norm_doc("Mobius", "TVA Analyst.", categories=["Characters"])
    entity = extractor.extract_entity(doc, series_config)

    assert entity is not None
    assert entity.entity_type == "character"
    assert entity.name == "Mobius"


@pytest.mark.unit
def test_category_type_mapping_location(series_config: SeriesConfig) -> None:
    extractor = DefaultEntityExtractor()
    doc = _create_norm_doc(
        "Time Variance Authority HQ", "The TVA HQ.", categories=["Locations"]
    )
    entity = extractor.extract_entity(doc, series_config)

    assert entity is not None
    assert entity.entity_type == "location"


@pytest.mark.unit
def test_all_entity_types(series_config: SeriesConfig) -> None:
    extractor = DefaultEntityExtractor()
    type_checks = [
        ("Loki", "Characters", "character"),
        ("Asgard", "Locations", "location"),
        ("TemPad", "Devices", "device"),
        ("TVA", "Organizations", "organization"),
        ("Bombing of the Sacred Timeline", "Events", "event"),
        ("Enchantment", "Abilities", "ability"),
        ("Sacred Timeline", "Concepts", "concept"),
    ]

    for title, cat, expected_type in type_checks:
        doc = _create_norm_doc(title, f"{title} synopsis.", categories=[cat])
        entity = extractor.extract_entity(doc, series_config)
        assert entity is not None
        assert entity.entity_type == expected_type


@pytest.mark.unit
def test_unmapped_category_defaults_to_concept(
    series_config: SeriesConfig,
) -> None:
    extractor = DefaultEntityExtractor()
    doc = _create_norm_doc(
        "Quantum Realm", "A realm.", categories=["UnknownRealmCategory"]
    )
    entity = extractor.extract_entity(doc, series_config)

    assert entity is not None
    assert entity.entity_type == "concept"


@pytest.mark.unit
def test_unmapped_category_report(series_config: SeriesConfig) -> None:
    extractor = DefaultEntityExtractor()
    doc1 = _create_norm_doc("Mobius", "Analyst", categories=["Characters"])
    doc2 = _create_norm_doc(
        "Unknown Widget", "A strange tool", categories=["UnmappedWidgets", "Oddities"]
    )

    entities, report = extractor.extract_entities([doc1, doc2], series_config)

    assert len(entities) == 2
    assert "UnmappedWidgets" in report.unmapped_categories
    assert "Oddities" in report.unmapped_categories
    assert "Unknown Widget" in report.unmapped_category_documents["UnmappedWidgets"]


@pytest.mark.unit
def test_disambiguation_page_skipped(series_config: SeriesConfig) -> None:
    extractor = DefaultEntityExtractor()
    # By category
    doc1 = _create_norm_doc(
        "Loki", "Disambiguation page text", categories=["Disambiguation pages"]
    )
    assert extractor.extract_entity(doc1, series_config) is None

    # By title
    doc2 = _create_norm_doc(
        "Mobius (disambiguation)", "Disambiguation list", categories=[]
    )
    assert extractor.extract_entity(doc2, series_config) is None


@pytest.mark.unit
def test_disambiguation_skipped_in_batch(series_config: SeriesConfig) -> None:
    extractor = DefaultEntityExtractor()
    doc1 = _create_norm_doc("Mobius", "TVA Analyst", categories=["Characters"])
    doc2 = _create_norm_doc("Mobius (disambiguation)", "Pointers", categories=[])
    doc3 = _create_norm_doc("Loki", "Variants", categories=["Category:Disambiguation"])

    entities, report = extractor.extract_entities([doc1, doc2, doc3], series_config)

    assert len(entities) == 1
    assert entities[0].name == "Mobius"
    assert report.disambiguation_pages_skipped == 2
    assert report.entities_extracted == 1
    assert report.documents_processed == 3


@pytest.mark.unit
def test_redirect_aliases(series_config: SeriesConfig) -> None:
    extractor = DefaultEntityExtractor()
    doc = _create_norm_doc(
        "Sylvie Laufeydottir",
        "A female variant of Loki.",
        categories=["Characters"],
        redirects=["Lady Loki", "The Variant"],
    )
    entity = extractor.extract_entity(doc, series_config)

    assert entity is not None
    assert "Lady Loki" in entity.aliases
    assert "The Variant" in entity.aliases
    # Should not include title itself as alias
    assert "Sylvie Laufeydottir" not in entity.aliases


@pytest.mark.unit
def test_internal_links_as_related_entities(
    series_config: SeriesConfig,
) -> None:
    extractor = DefaultEntityExtractor()
    doc = _create_norm_doc(
        "Sylvie Laufeydottir",
        "A female variant.",
        categories=["Characters"],
        links=["Loki", "Mobius M. Mobius", "Time Variance Authority"],
    )
    entity = extractor.extract_entity(doc, series_config)

    assert entity is not None
    assert EntityId("loki") in entity.related_entities
    assert EntityId("mobius-m-mobius") in entity.related_entities
    assert EntityId("time-variance-authority") in entity.related_entities


@pytest.mark.unit
def test_deterministic_entity_id(series_config: SeriesConfig) -> None:
    extractor = DefaultEntityExtractor()
    doc1 = _create_norm_doc("He Who Remains", "Creator of TVA.")
    doc2 = _create_norm_doc("He Who Remains", "Different content.")

    e1 = extractor.extract_entity(doc1, series_config)
    e2 = extractor.extract_entity(doc2, series_config)

    assert e1 is not None and e2 is not None
    assert e1.entity_id == e2.entity_id == EntityId("he-who-remains")


@pytest.mark.unit
def test_entity_summary_first_paragraph(series_config: SeriesConfig) -> None:
    extractor = DefaultEntityExtractor()
    text = """# Heading 1

First real synopsis paragraph explaining who Loki is in detail.

## Section 2
Second paragraph.
"""
    doc = _create_norm_doc("Loki", text, categories=["Characters"])
    entity = extractor.extract_entity(doc, series_config)

    assert entity is not None
    expected_summary = "First real synopsis paragraph explaining who Loki is in detail."
    assert entity.summary == expected_summary


@pytest.mark.unit
def test_entity_metadata_preservation(series_config: SeriesConfig) -> None:
    extractor = DefaultEntityExtractor()
    doc = _create_norm_doc(
        "Loki",
        "God of Mischief.",
        categories=["Characters"],
        extra_meta={
            "revision_id": "rev-999",
            "license": "CC-BY-SA-3.0",
            "custom_field": "val123",
        },
    )
    entity = extractor.extract_entity(doc, series_config)

    assert entity is not None
    assert entity.metadata["revision_id"] == "rev-999"
    assert entity.metadata["license"] == "CC-BY-SA-3.0"
    assert entity.metadata["custom_field"] == "val123"
    assert entity.metadata["source_url"] == doc.source_url
