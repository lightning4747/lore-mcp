"""Qdrant EventRegistry adapter managing events in lore_meta [S10, S11]."""

import logging
import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from lore_mcp.adapters.qdrant.errors import wrap_qdrant_error
from lore_mcp.config import QdrantSettings
from lore_mcp.domain.entities import Event
from lore_mcp.domain.identifiers import LORE_CHUNK_NAMESPACE, slugify
from lore_mcp.domain.types import EntityId, EventId, SeriesId, ValidatedSeriesId
from lore_mcp.ports.event_registry import EventRegistry

logger = logging.getLogger(__name__)


def to_event_point_id(
    series_id: SeriesId | ValidatedSeriesId | str,
    event_id: EventId | str,
) -> str:
    """Derive deterministic UUID5 point ID for an event in lore_meta."""
    return str(uuid.uuid5(LORE_CHUNK_NAMESPACE, f"event:{series_id}:{event_id}"))


class QdrantEventRegistry(EventRegistry):
    """Qdrant implementation of EventRegistry port backed by meta_collection."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        settings: QdrantSettings,
    ) -> None:
        self._client: AsyncQdrantClient = client
        self._settings: QdrantSettings = settings

    def _reconstruct_event(self, payload: dict[str, Any]) -> Event:
        """Deserialize or reconstruct Event from point payload."""
        event_json = payload.get("_event_json")
        if isinstance(event_json, str):
            return Event.model_validate_json(event_json)

        return Event(
            event_id=EventId(str(payload.get("event_id", ""))),
            series_id=SeriesId(str(payload.get("series_id", ""))),
            name=str(payload.get("name", "")),
            description=str(payload.get("description", "")),
            season=payload.get("season"),
            episode=payload.get("episode"),
            scene_index=payload.get("scene_index"),
            narrative_order=payload.get("narrative_order"),
            participants=[EntityId(str(p)) for p in payload.get("participants", [])],
            location=payload.get("location"),
            consequences=[str(c) for c in payload.get("consequences", [])],
        )

    def _build_point(self, event: Event) -> models.PointStruct:
        """Build Qdrant PointStruct for event metadata."""
        point_id = to_event_point_id(event.series_id, event.event_id)
        payload: dict[str, Any] = {
            "type": "event",
            "series_id": str(event.series_id),
            "event_id": str(event.event_id),
            "name": event.name,
            "description": event.description,
            "season": event.season,
            "episode": event.episode,
            "scene_index": event.scene_index,
            "narrative_order": event.narrative_order,
            "participants": [str(p) for p in event.participants],
            "location": event.location,
            "consequences": event.consequences,
            "_event_json": event.model_dump_json(),
        }
        return models.PointStruct(
            id=point_id,
            vector={},
            payload=payload,
        )

    async def register_event(self, event: Event) -> None:
        """Register or update a narrative lore event in meta_collection."""
        await self.register_events([event])

    async def register_events(self, events: Sequence[Event]) -> None:
        """Register or update a batch of events in meta_collection."""
        if not events:
            return

        points = [self._build_point(event) for event in events]
        try:
            await self._client.upsert(
                collection_name=self._settings.meta_collection,
                points=points,
                wait=True,
            )
        except Exception as err:
            raise wrap_qdrant_error(
                err,
                f"Failed to register events in '{self._settings.meta_collection}'",
                logger,
            ) from err

    async def get_event(self, series_id: SeriesId, event_id: EventId) -> Event | None:
        """Retrieve an event by its deterministic ID within a series."""
        point_id = to_event_point_id(series_id, event_id)
        try:
            records = await self._client.retrieve(
                collection_name=self._settings.meta_collection,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as err:
            raise wrap_qdrant_error(
                err,
                f"Failed to retrieve event '{event_id}' from "
                f"'{self._settings.meta_collection}'",
                logger,
            ) from err

        if not records:
            return None

        record = records[0]
        payload = record.payload or {}
        return self._reconstruct_event(payload)

    async def list_events(self, series_id: SeriesId) -> list[Event]:
        """List all registered events for a series."""
        filter_conditions: list[models.Condition] = [
            models.FieldCondition(
                key="type",
                match=models.MatchValue(value="event"),
            ),
            models.FieldCondition(
                key="series_id",
                match=models.MatchValue(value=str(series_id)),
            ),
        ]
        try:
            records, _ = await self._client.scroll(
                collection_name=self._settings.meta_collection,
                scroll_filter=models.Filter(must=filter_conditions),
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as err:
            raise wrap_qdrant_error(
                err,
                f"Failed to list events for series '{series_id}' from "
                f"'{self._settings.meta_collection}'",
                logger,
            ) from err

        events: list[Event] = []
        for r in records:
            if r.payload:
                events.append(self._reconstruct_event(r.payload))
        return events

    async def resolve_event(self, series_id: SeriesId, name_or_id: str) -> Event | None:
        """Resolve an event by deterministic ID, slug, or title/name."""
        clean = name_or_id.strip()
        if not clean:
            return None

        # 1. Try direct deterministic ID
        direct = await self.get_event(series_id, EventId(clean))
        if direct is not None:
            return direct

        # 2. Try slugified ID
        slug = slugify(clean)
        if slug:
            direct_slug = await self.get_event(series_id, EventId(slug))
            if direct_slug is not None:
                return direct_slug

        # 3. Try matching title/name or event_id across series events
        events = await self.list_events(series_id)
        clean_lower = clean.lower()

        for event in events:
            if (
                event.name.lower() == clean_lower
                or str(event.event_id).lower() == clean_lower
            ):
                return event

        return None
