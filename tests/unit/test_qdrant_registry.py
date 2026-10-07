"""Unit tests for QdrantSeriesRegistry adapter,
schema version rejection, and alias resolution.
"""

from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient, models

from lore_mcp.adapters.qdrant.errors import QdrantOperationError
from lore_mcp.adapters.qdrant.initializer import QdrantInitializer
from lore_mcp.adapters.qdrant.registry import QdrantSeriesRegistry, to_manifest_point_id
from lore_mcp.application.errors import DependencyUnavailable, IncompatibleData
from lore_mcp.config import QdrantSettings
from lore_mcp.domain.entities import SeriesManifest
from lore_mcp.domain.types import SeriesId


@pytest.fixture
def registry_settings() -> QdrantSettings:
    return QdrantSettings(
        url=":memory:",
        chunks_collection="meta_test_chunks",
        meta_collection="meta_test_meta",
        embedding_model="all-MiniLM-L6-v2",
        dimension=384,
        inference_backend="fastembed",
        schema_version=1,
    )


async def make_registry_client(settings: QdrantSettings) -> AsyncQdrantClient:
    client = AsyncQdrantClient(":memory:")
    await QdrantInitializer.ensure_collections_ready(client, settings)
    return client


@pytest.mark.unit
@pytest.mark.asyncio
async def test_register_and_get_manifest(
    registry_settings: QdrantSettings,
) -> None:
    client = await make_registry_client(registry_settings)
    registry = QdrantSeriesRegistry(client, registry_settings)

    manifest = SeriesManifest(
        series_id=SeriesId("loki"),
        display_name="Loki (TV Series)",
        kind="show",
        aliases=["Loki", "MCU Loki"],
        seasons=[1, 2],
        entity_count=42,
        chunk_count=150,
        ingest_version=1,
        schema_version=1,
    )

    await registry.register_series(manifest)

    fetched = await registry.get_series(SeriesId("loki"))
    assert fetched is not None
    assert fetched.series_id == "loki"
    assert fetched.display_name == "Loki (TV Series)"
    assert fetched.aliases == ["Loki", "MCU Loki"]
    assert fetched.seasons == [1, 2]
    assert fetched.entity_count == 42
    assert fetched.chunk_count == 150
    assert fetched.ingest_version == 1
    assert fetched.schema_version == 1

    assert await registry.get_series(SeriesId("nonexistent")) is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rejects_unsupported_schema_version_on_register(
    registry_settings: QdrantSettings,
) -> None:
    client = await make_registry_client(registry_settings)
    registry = QdrantSeriesRegistry(client, registry_settings)

    invalid_manifest = SeriesManifest(
        series_id=SeriesId("loki"),
        display_name="Loki",
        schema_version=99,
    )

    with pytest.raises(IncompatibleData) as exc_info:
        await registry.register_series(invalid_manifest)

    assert exc_info.value.expected_version == 1
    assert exc_info.value.actual_version == 99


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rejects_unsupported_schema_version_on_read(
    registry_settings: QdrantSettings,
) -> None:
    client = await make_registry_client(registry_settings)
    registry = QdrantSeriesRegistry(client, registry_settings)

    # Insert point directly with an incompatible schema_version
    point_id = to_manifest_point_id("future-series")
    await client.upsert(
        collection_name=registry_settings.meta_collection,
        points=[
            models.PointStruct(
                id=point_id,
                vector={},
                payload={
                    "series_id": "future-series",
                    "display_name": "Future Show",
                    "schema_version": 2,
                },
            )
        ],
    )

    with pytest.raises(IncompatibleData) as exc_info:
        await registry.get_series(SeriesId("future-series"))

    assert exc_info.value.expected_version == 1
    assert exc_info.value.actual_version == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_and_delete_series(
    registry_settings: QdrantSettings,
) -> None:
    client = await make_registry_client(registry_settings)
    registry = QdrantSeriesRegistry(client, registry_settings)

    m1 = SeriesManifest(series_id=SeriesId("loki"), display_name="Loki")
    m2 = SeriesManifest(series_id=SeriesId("andor"), display_name="Andor")
    await registry.register_series(m1)
    await registry.register_series(m2)

    series_list = await registry.list_series()
    assert series_list == [SeriesId("andor"), SeriesId("loki")]

    assert await registry.delete_series(SeriesId("loki")) is True
    assert await registry.delete_series(SeriesId("loki")) is False
    assert await registry.get_series(SeriesId("loki")) is None

    series_list_after = await registry.list_series()
    assert series_list_after == [SeriesId("andor")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_alias_resolution(
    registry_settings: QdrantSettings,
) -> None:
    client = await make_registry_client(registry_settings)
    registry = QdrantSeriesRegistry(client, registry_settings)

    manifest = SeriesManifest(
        series_id=SeriesId("loki"),
        display_name="Loki (TV Series)",
        aliases=["God of Mischief", "MCU Loki", "Loki Season 2"],
    )
    await registry.register_series(manifest)

    alias_map = await registry.get_alias_map()
    assert alias_map["loki"] == SeriesId("loki")
    assert alias_map["loki (tv series)"] == SeriesId("loki")
    assert alias_map["god of mischief"] == SeriesId("loki")
    assert alias_map["mcu loki"] == SeriesId("loki")

    # resolve_alias tests
    assert await registry.resolve_alias("loki") == SeriesId("loki")
    assert await registry.resolve_alias("Loki (TV Series)") == SeriesId("loki")
    assert await registry.resolve_alias("  MCU Loki  ") == SeriesId("loki")
    assert await registry.resolve_alias("nonexistent") is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_registry_error_wrapping(
    registry_settings: QdrantSettings,
) -> None:
    mock_client = AsyncMock(spec=AsyncQdrantClient)
    mock_client.upsert.side_effect = RuntimeError("Upsert network error")
    mock_client.retrieve.side_effect = RuntimeError("Retrieve network error")
    mock_client.scroll.side_effect = RuntimeError("Scroll network error")

    registry = QdrantSeriesRegistry(mock_client, registry_settings)
    manifest = SeriesManifest(series_id=SeriesId("loki"), display_name="Loki")

    with pytest.raises(QdrantOperationError) as exc_upsert:
        await registry.register_series(manifest)
    assert isinstance(exc_upsert.value, DependencyUnavailable)

    with pytest.raises(QdrantOperationError) as exc_get:
        await registry.get_series(SeriesId("loki"))
    assert isinstance(exc_get.value, DependencyUnavailable)

    with pytest.raises(QdrantOperationError) as exc_list:
        await registry.list_series()
    assert isinstance(exc_list.value, DependencyUnavailable)
