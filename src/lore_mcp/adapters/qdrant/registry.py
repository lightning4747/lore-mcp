"""Qdrant SeriesRegistry adapter managing manifests in lore_meta [S10, S11]."""

import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from lore_mcp.adapters.qdrant.errors import QdrantOperationError
from lore_mcp.application.errors import IncompatibleData
from lore_mcp.config import QdrantSettings
from lore_mcp.domain.entities import SeriesManifest
from lore_mcp.domain.identifiers import LORE_CHUNK_NAMESPACE
from lore_mcp.domain.types import SeriesId
from lore_mcp.ports.series_registry import SeriesRegistry

logger = logging.getLogger(__name__)


def to_manifest_point_id(series_id: SeriesId | str) -> str:
    """Derive deterministic UUID5 point ID for series manifest in lore_meta."""
    return str(uuid.uuid5(LORE_CHUNK_NAMESPACE, f"manifest:{series_id}"))


class QdrantSeriesRegistry(SeriesRegistry):
    """Qdrant implementation of SeriesRegistry port backed by meta_collection."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        settings: QdrantSettings,
    ) -> None:
        self._client: AsyncQdrantClient = client
        self._settings: QdrantSettings = settings

    def _reconstruct_manifest(self, payload: dict[str, Any]) -> SeriesManifest:
        """Deserialize or reconstruct SeriesManifest from point payload."""
        schema_version = payload.get("schema_version")
        if (
            schema_version is not None
            and int(schema_version) != self._settings.schema_version
        ):
            raise IncompatibleData(
                message=(
                    f"Manifest in '{self._settings.meta_collection}' has unsupported "
                    f"schema_version {schema_version} (configured version: "
                    f"{self._settings.schema_version})"
                ),
                expected_version=self._settings.schema_version,
                actual_version=int(schema_version),
            )

        manifest_json = payload.get("_manifest_json")
        if isinstance(manifest_json, str):
            manifest = SeriesManifest.model_validate_json(manifest_json)
            if manifest.schema_version != self._settings.schema_version:
                raise IncompatibleData(
                    message=(
                        f"Manifest for series '{manifest.series_id}' has unsupported "
                        f"schema_version {manifest.schema_version} "
                        f"(configured version: {self._settings.schema_version})"
                    ),
                    expected_version=self._settings.schema_version,
                    actual_version=manifest.schema_version,
                )
            return manifest

        # Fallback reconstruction if _manifest_json is missing
        return SeriesManifest(
            series_id=SeriesId(str(payload.get("series_id", ""))),
            display_name=str(payload.get("display_name", "")),
            kind=str(payload.get("kind", "show")),
            aliases=[str(a) for a in payload.get("aliases", [])],
            seasons=[int(s) for s in payload.get("seasons", [])],
            entity_count=int(payload.get("entity_count", 0)),
            chunk_count=int(payload.get("chunk_count", 0)),
            ingest_version=int(payload.get("ingest_version", 1)),
            schema_version=int(
                payload.get("schema_version", self._settings.schema_version)
            ),
        )

    async def register_series(self, manifest: SeriesManifest) -> None:
        """Register or update a series manifest in meta_collection."""
        if manifest.schema_version != self._settings.schema_version:
            raise IncompatibleData(
                message=(
                    f"Manifest for series '{manifest.series_id}' has unsupported "
                    f"schema_version {manifest.schema_version} (configured version: "
                    f"{self._settings.schema_version})"
                ),
                expected_version=self._settings.schema_version,
                actual_version=manifest.schema_version,
            )

        point_id = to_manifest_point_id(manifest.series_id)
        payload: dict[str, Any] = {
            "series_id": str(manifest.series_id),
            "display_name": manifest.display_name,
            "kind": manifest.kind,
            "aliases": manifest.aliases,
            "seasons": manifest.seasons,
            "entity_count": manifest.entity_count,
            "chunk_count": manifest.chunk_count,
            "ingest_version": manifest.ingest_version,
            "schema_version": manifest.schema_version,
            "_manifest_json": manifest.model_dump_json(),
        }

        point = models.PointStruct(
            id=point_id,
            vector={},
            payload=payload,
        )

        try:
            await self._client.upsert(
                collection_name=self._settings.meta_collection,
                points=[point],
                wait=True,
            )
        except Exception as err:
            raise QdrantOperationError(
                f"Failed to register series '{manifest.series_id}' in "
                f"'{self._settings.meta_collection}': {err}"
            ) from err

    async def get_series(self, series_id: SeriesId) -> SeriesManifest | None:
        """Retrieve the manifest for a given series."""
        point_id = to_manifest_point_id(series_id)
        try:
            records = await self._client.retrieve(
                collection_name=self._settings.meta_collection,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as err:
            raise QdrantOperationError(
                f"Failed to retrieve series '{series_id}' from "
                f"'{self._settings.meta_collection}': {err}"
            ) from err

        if not records:
            return None

        record = records[0]
        payload = record.payload or {}
        return self._reconstruct_manifest(payload)

    async def list_series(self) -> list[SeriesId]:
        """List all registered series identifiers."""
        manifests = await self.get_manifests()
        return sorted([m.series_id for m in manifests])

    async def delete_series(self, series_id: SeriesId) -> bool:
        """Delete a series manifest from the registry."""
        point_id = to_manifest_point_id(series_id)
        try:
            records = await self._client.retrieve(
                collection_name=self._settings.meta_collection,
                ids=[point_id],
                with_payload=False,
                with_vectors=False,
            )
            if not records:
                return False

            await self._client.delete(
                collection_name=self._settings.meta_collection,
                points_selector=models.PointIdsList(points=[point_id]),
                wait=True,
            )
            return True
        except Exception as err:
            raise QdrantOperationError(
                f"Failed to delete series '{series_id}' from "
                f"'{self._settings.meta_collection}': {err}"
            ) from err

    async def get_manifests(self) -> list[SeriesManifest]:
        """Retrieve all registered series manifests."""
        try:
            records, _ = await self._client.scroll(
                collection_name=self._settings.meta_collection,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as err:
            raise QdrantOperationError(
                f"Failed to list series manifests from "
                f"'{self._settings.meta_collection}': {err}"
            ) from err

        manifests: list[SeriesManifest] = []
        for r in records:
            if r.payload:
                manifests.append(self._reconstruct_manifest(r.payload))
        return manifests

    async def get_alias_map(self) -> dict[str, SeriesId]:
        """Return a mapping of all lowercase aliases and names to SeriesId."""
        manifests = await self.get_manifests()
        alias_map: dict[str, SeriesId] = {}
        for m in manifests:
            s_id = m.series_id
            alias_map[str(s_id).lower()] = s_id
            alias_map[m.display_name.lower()] = s_id
            for a in m.aliases:
                alias_map[a.strip().lower()] = s_id
        return alias_map

    async def resolve_alias(self, alias_or_name: str) -> SeriesId | None:
        """Resolve a display name or alias to canonical SeriesId."""
        alias_map = await self.get_alias_map()
        return alias_map.get(alias_or_name.strip().lower())
