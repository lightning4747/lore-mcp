"""Unit tests for deterministic stable identifier generators."""

import uuid

import pytest

from lore_mcp.domain import (
    ChunkId,
    EntityId,
    EventId,
    SeriesId,
    generate_chunk_id,
    generate_entity_id,
    generate_event_id,
    slugify,
)


@pytest.mark.unit
def test_slugify() -> None:
    assert slugify("Loki") == "loki"
    assert slugify("He Who Remains") == "he-who-remains"
    assert slugify("O.B.") == "o-b"
    assert slugify("Temporal Loom (S2)") == "temporal-loom-s2"
    assert slugify("  ---weird--name---  ") == "weird-name"


@pytest.mark.unit
def test_generate_entity_id_stability() -> None:
    id1 = generate_entity_id("He Who Remains")
    id2 = generate_entity_id("he who remains")
    assert id1 == id2 == EntityId("he-who-remains")

    assert generate_entity_id("Victor Timely") == EntityId("victor-timely")
    assert generate_entity_id("TVA") == EntityId("tva")

    with pytest.raises(ValueError, match="produced an empty slug"):
        generate_entity_id("   !!!   ")


@pytest.mark.unit
def test_generate_chunk_id_stability_and_determinism() -> None:
    series = SeriesId("loki")
    url = "https://marvel.fandom.com/wiki/Temporal_Loom"
    section = "Overview/Mechanics"
    split = 0

    id1 = generate_chunk_id(series, url, section, split)
    id2 = generate_chunk_id(series, url, section, split)

    # Determinism across calls
    assert id1 == id2
    # Valid UUID format
    assert uuid.UUID(id1).version == 5

    # Changing split index changes ID
    id_split1 = generate_chunk_id(series, url, section, 1)
    assert id1 != id_split1

    # Changing section changes ID
    id_diff_sec = generate_chunk_id(series, url, "History", split)
    assert id1 != id_diff_sec

    # Changing series changes ID (cross-series uniqueness)
    id_other_series = generate_chunk_id(SeriesId("star-wars"), url, section, split)
    assert id1 != id_other_series

    # Negative split index rejected
    with pytest.raises(ValueError, match="split_index must be non-negative"):
        generate_chunk_id(series, url, section, -1)


@pytest.mark.unit
def test_chunk_id_content_independence() -> None:
    # Verifies that chunk ID depends strictly on coordinates, not chunk text
    series = SeriesId("loki")
    url = "https://marvel.fandom.com/wiki/Temporal_Loom"
    section = "Overview"
    split = 0

    original_chunk_id = generate_chunk_id(series, url, section, split)
    # Even if text content is edited/updated later, chunk ID remains unchanged
    updated_chunk_id = generate_chunk_id(series, url, section, split)
    assert original_chunk_id == updated_chunk_id == ChunkId(str(original_chunk_id))


@pytest.mark.unit
def test_generate_event_id_determinism() -> None:
    # With season and episode
    ev1 = generate_event_id("loki", episode=6, event_key="loom-collapse", season=2)
    ev2 = generate_event_id("loki", episode=6, event_key="loom-collapse", season=2)
    assert ev1 == ev2 == EventId("loki:s2e6:loom-collapse")

    # With episode only
    ev3 = generate_event_id("loki", episode=4, event_key="meltdown")
    assert ev3 == EventId("loki:e4:meltdown")

    # Unscoped episode
    ev4 = generate_event_id("loki", episode=None, event_key="tva-multiverse-war")
    assert ev4 == EventId("loki:tva-multiverse-war")

    # Cross-series uniqueness
    ev_diff_series = generate_event_id(
        "doctor-who", episode=6, event_key="loom-collapse", season=2
    )
    assert ev1 != ev_diff_series

    # Empty event key rejected
    with pytest.raises(ValueError, match="produced an empty slug"):
        generate_event_id("loki", episode=1, event_key="   !!!   ")
