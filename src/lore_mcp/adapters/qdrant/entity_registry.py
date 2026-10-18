"""Qdrant EntityRegistry adapter managing entities in lore_meta [S10, S11]."""

import logging
import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from lore_mcp.adapters.qdrant.errors import wrap_qdrant_error
from lore_mcp.config import QdrantSettings
from lore_mcp.domain.entities import Entity
from lore_mcp.domain.identifiers import LORE_CHUNK_NAMESPACE, slugify
from lore_mcp.domain.types import EntityId, SeriesId, ValidatedSeriesId
from lore_mcp.ports.entity_registry import EntityRegistry

logger = logging.getLogger(__name__)


def to_entity_point_id(
    series_id: SeriesId | ValidatedSeriesId | str,
    entity_id: EntityId | str,
) -> str:
    """Derive deterministic UUID5 point ID for an entity in lore_meta."""
    return str(uuid.uuid5(LORE_CHUNK_NAMESPACE, f"entity:{series_id}:{entity_id}"))


class QdrantEntityRegistry(EntityRegistry):
    """Qdrant implementation of EntityRegistry port backed by meta_collection."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        settings: QdrantSettings,
    ) -> None:
        self._client: AsyncQdrantClient = client
        self._settings: QdrantSettings = settings

    def _reconstruct_entity(self, payload: dict[str, Any]) -> Entity:
        """Deserialize or reconstruct Entity from point payload."""
        entity_json = payload.get("_entity_json")
        if isinstance(entity_json, str):
            return Entity.model_validate_json(entity_json)

        return Entity(
            entity_id=EntityId(str(payload.get("entity_id", ""))),
            series_id=SeriesId(str(payload.get("series_id", ""))),
            name=str(payload.get("name", "")),
            entity_type=str(payload.get("entity_type", "concept")),
            aliases=[str(a) for a in payload.get("aliases", [])],
            summary=str(payload.get("summary", "")),
            related_entities=[
                EntityId(str(e)) for e in payload.get("related_entities", [])
            ],
            metadata={
                str(k): str(v)
                for k, v in payload.get("metadata", {}).items()
                if isinstance(payload.get("metadata"), dict)
            },
        )

    def _build_point(self, entity: Entity) -> models.PointStruct:
        """Build Qdrant PointStruct for entity metadata."""
        point_id = to_entity_point_id(entity.series_id, entity.entity_id)
        payload: dict[str, Any] = {
            "type": "entity",
            "series_id": str(entity.series_id),
            "entity_id": str(entity.entity_id),
            "name": entity.name,
            "entity_type": entity.entity_type,
            "aliases": entity.aliases,
            "summary": entity.summary,
            "related_entities": [str(e) for e in entity.related_entities],
            "metadata": entity.metadata,
            "_entity_json": entity.model_dump_json(),
        }
        return models.PointStruct(
            id=point_id,
            vector={},
            payload=payload,
        )

    async def register_entity(self, entity: Entity) -> None:
        """Register or update a canonical lore entity in meta_collection."""
        await self.register_entities([entity])

    async def register_entities(self, entities: Sequence[Entity]) -> None:
        """Register or update a batch of entities in meta_collection."""
        if not entities:
            return

        points = [self._build_point(entity) for entity in entities]
        try:
            await self._client.upsert(
                collection_name=self._settings.meta_collection,
                points=points,
                wait=True,
            )
        except Exception as err:
            raise wrap_qdrant_error(
                err,
                f"Failed to register entities in '{self._settings.meta_collection}'",
                logger,
            ) from err

    async def get_entity(
        self, series_id: SeriesId, entity_id: EntityId
    ) -> Entity | None:
        """Retrieve an entity by its canonical slug ID within a series."""
        point_id = to_entity_point_id(series_id, entity_id)
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
                f"Failed to retrieve entity '{entity_id}' from "
                f"'{self._settings.meta_collection}'",
                logger,
            ) from err

        if not records:
            return None

        record = records[0]
        payload = record.payload or {}
        return self._reconstruct_entity(payload)

    async def list_entities(self, series_id: SeriesId) -> list[Entity]:
        """List all registered entities for a series."""
        filter_conditions: list[models.Condition] = [
            models.FieldCondition(
                key="type",
                match=models.MatchValue(value="entity"),
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
                f"Failed to list entities for series '{series_id}' from "
                f"'{self._settings.meta_collection}'",
                logger,
            ) from err

        entities: list[Entity] = []
        for r in records:
            if r.payload:
                entities.append(self._reconstruct_entity(r.payload))
        return entities

    async def resolve_entity(
        self, series_id: SeriesId, name_or_alias: str
    ) -> Entity | None:
        """Resolve a canonical entity by canonical ID, display name, or alias."""
        clean = name_or_alias.strip()
        if not clean:
            return None

        # 1. Try direct canonical slug ID
        slug = slugify(clean)
        if slug:
            direct = await self.get_entity(series_id, EntityId(slug))
            if direct is not None:
                return direct

        # 2. Try matching name or alias across series entities
        entities = await self.list_entities(series_id)
        clean_lower = clean.lower()

        # Check exact name match (case-insensitive)
        for entity in entities:
            if entity.name.lower() == clean_lower:
                return entity

        # Check aliases
        for entity in entities:
            if any(alias.lower() == clean_lower for alias in entity.aliases):
                return entity

        return None
