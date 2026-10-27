"""Loader for curated series timeline narrative order configurations."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter


class TimelineItem(BaseModel):
    """Schema for individual timeline event entry."""

    model_config = ConfigDict(extra="ignore")

    event_id: str | None = None
    narrative_order: int | None = None


class TimelineWrapper(BaseModel):
    """Schema for timeline dictionary with 'events' array."""

    model_config = ConfigDict(extra="ignore")

    events: list[TimelineItem] = []


class CuratedTimelineLoader:
    """Loads curated narrative order mappings from a timeline JSON file."""

    @staticmethod
    def load(timeline_path: str | Path) -> dict[str, int]:
        """Load curated narrative orders from a JSON file.

        Returns a dictionary mapping event_id strings to integer narrative orders.
        Entries with missing or non-integer narrative orders are skipped.
        Raises FileNotFoundError if path does not exist.
        """
        path = Path(timeline_path)
        if not path.is_file():
            raise FileNotFoundError(f"Timeline file not found: {path}")

        raw_content = path.read_text(encoding="utf-8").strip()
        if not raw_content:
            return {}

        orders: dict[str, int] = {}

        # 1. Try parsing as a list of TimelineItem
        try:
            list_adapter = TypeAdapter(list[TimelineItem])
            items = list_adapter.validate_json(raw_content)
            for item in items:
                if item.event_id and item.narrative_order is not None:
                    orders[item.event_id.strip()] = item.narrative_order
            return orders
        except Exception:
            pass

        # 2. Try parsing as a TimelineWrapper ({events: [...]})
        try:
            wrapper_adapter = TypeAdapter(TimelineWrapper)
            wrapper = wrapper_adapter.validate_json(raw_content)
            if wrapper.events:
                for item in wrapper.events:
                    if item.event_id and item.narrative_order is not None:
                        orders[item.event_id.strip()] = item.narrative_order
                return orders
        except Exception:
            pass

        # 3. Try parsing as a direct {event_id: narrative_order} mapping
        try:
            dict_adapter = TypeAdapter(dict[str, int])
            return dict_adapter.validate_json(raw_content)
        except Exception:
            pass

        return orders
