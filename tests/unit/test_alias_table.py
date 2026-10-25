"""Unit tests for AliasTable and AliasTableBuilder (Task 4.4 & 4.5)."""

import pytest

from lore_mcp.application.extraction.alias_builder import AliasTableBuilder
from lore_mcp.domain.entities import Entity
from lore_mcp.domain.types import EntityId, SeriesId


def _create_entity(
    entity_id: str,
    name: str,
    entity_type: str = "character",
    aliases: list[str] | None = None,
) -> Entity:
    return Entity(
        entity_id=EntityId(entity_id),
        series_id=SeriesId("loki"),
        name=name,
        entity_type=entity_type,
        aliases=aliases or [],
        summary=f"Summary for {name}",
    )


@pytest.mark.unit
def test_canonical_name_lookup() -> None:
    builder = AliasTableBuilder()
    entities = [
        _create_entity("mobius", "Mobius M. Mobius", aliases=["Agent Mobius"]),
        _create_entity("sylvie", "Sylvie Laufeydottir", aliases=["The Variant"]),
    ]
    table = builder.build(SeriesId("loki"), entities)

    assert table.lookup("Mobius M. Mobius") == EntityId("mobius")
    assert table.lookup("Sylvie Laufeydottir") == EntityId("sylvie")


@pytest.mark.unit
def test_alias_lookup() -> None:
    builder = AliasTableBuilder()
    entities = [
        _create_entity(
            "loki",
            "Loki Laufeyson",
            aliases=["God of Mischief", "Variant L1130", "God of Stories"],
        ),
    ]
    table = builder.build(SeriesId("loki"), entities)

    assert table.lookup("God of Mischief") == EntityId("loki")
    assert table.lookup("Variant L1130") == EntityId("loki")
    assert table.lookup("God of Stories") == EntityId("loki")


@pytest.mark.unit
def test_case_insensitive_lookup() -> None:
    builder = AliasTableBuilder()
    entities = [
        _create_entity("mobius", "Mobius M. Mobius", aliases=["Agent Mobius"]),
    ]
    table = builder.build(SeriesId("loki"), entities)

    assert table.lookup("mobius m. mobius") == EntityId("mobius")
    assert table.lookup("MOBIUS M. MOBIUS") == EntityId("mobius")
    assert table.lookup("agent mobius") == EntityId("mobius")
    assert table.lookup("  AGENT   MOBIUS  ") == EntityId("mobius")


@pytest.mark.unit
def test_diacritic_and_unicode_lookup() -> None:
    builder = AliasTableBuilder()
    # Entity with accented/decomposed character
    entities = [
        _create_entity("renslayer", "Ravonna Renslayer", aliases=["René"]),
    ]
    table = builder.build(SeriesId("loki"), entities)

    # Composed é vs decomposed e + accent
    decomposed = "Rene\u0301"
    composed = "Ren\u00e9"

    assert table.lookup(decomposed) == EntityId("renslayer")
    assert table.lookup(composed) == EntityId("renslayer")


@pytest.mark.unit
def test_miss_returns_none() -> None:
    builder = AliasTableBuilder()
    entities = [_create_entity("mobius", "Mobius M. Mobius")]
    table = builder.build(SeriesId("loki"), entities)

    assert table.lookup("Thor") is None
    assert table.lookup("") is None
    assert table.lookup("   ") is None


@pytest.mark.unit
def test_canonical_wins_over_alias() -> None:
    builder = AliasTableBuilder()
    entities = [
        _create_entity("loki-canonical", "Loki"),
        # Another entity with "Loki" as alias
        _create_entity("sylvie", "Sylvie", aliases=["Loki", "Lady Loki"]),
    ]
    table = builder.build(SeriesId("loki"), entities)

    # First registered canonical name "Loki" must map to "loki-canonical"
    assert table.lookup("Loki") == EntityId("loki-canonical")
    # Sylvie still resolves through her canonical name and distinct alias
    assert table.lookup("Sylvie") == EntityId("sylvie")
    assert table.lookup("Lady Loki") == EntityId("sylvie")
