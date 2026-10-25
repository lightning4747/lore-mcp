"""Unit tests for CuratedTimelineLoader (Phase 4)."""

import json
from pathlib import Path

import pytest

from lore_mcp.application.extraction.timeline_loader import CuratedTimelineLoader


@pytest.mark.unit
def test_loads_narrative_orders(tmp_path: Path) -> None:
    data = [
        {
            "event_id": "loki:s01e01:arrest",
            "narrative_order": 100,
            "description": "Loki arrested in Gobi Desert",
        },
        {
            "event_id": "loki:s01e01:trial",
            "narrative_order": 110,
            "description": "Loki trial before Renslayer",
        },
    ]
    file_path = tmp_path / "timeline.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    orders = CuratedTimelineLoader.load(file_path)
    assert orders == {
        "loki:s01e01:arrest": 100,
        "loki:s01e01:trial": 110,
    }


@pytest.mark.unit
def test_loads_from_dict_wrapper(tmp_path: Path) -> None:
    data = {
        "events": [
            {"event_id": "event-1", "narrative_order": 5},
            {"event_id": "event-2", "narrative_order": 10},
        ]
    }
    file_path = tmp_path / "timeline.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    orders = CuratedTimelineLoader.load(file_path)
    assert orders == {"event-1": 5, "event-2": 10}


@pytest.mark.unit
def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        CuratedTimelineLoader.load("nonexistent_timeline_path_123.json")


@pytest.mark.unit
def test_empty_file_returns_empty(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.json"
    file_path.write_text("", encoding="utf-8")

    orders = CuratedTimelineLoader.load(file_path)
    assert orders == {}


@pytest.mark.unit
def test_entries_without_narrative_order_skipped(tmp_path: Path) -> None:
    data = [
        {"event_id": "valid-1", "narrative_order": 20},
        {"event_id": "no-order", "description": "Missing order"},
        {"event_id": "null-order", "narrative_order": None},
        {"description": "Missing event_id", "narrative_order": 30},
    ]
    file_path = tmp_path / "partial.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    orders = CuratedTimelineLoader.load(file_path)
    assert orders == {"valid-1": 20}
