"""Qdrant collection initializer and metadata validator [S4, S5, S9-S11]."""

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from lore_mcp.adapters.qdrant.errors import (
    QdrantInitializationError,
    QdrantMetadataMismatchError,
)
from lore_mcp.config import QdrantSettings

logger = logging.getLogger(__name__)


class QdrantInitializer:
    """Manages Qdrant collections initialization and schema compatibility checks."""

    @classmethod
    async def validate_chunks_metadata(
        cls,
        client: AsyncQdrantClient,
        settings: QdrantSettings,
    ) -> None:
        """Validate existing chunks collection metadata against configured settings."""
        try:
            info = await client.get_collection(settings.chunks_collection)
        except Exception as err:
            raise QdrantInitializationError(
                f"Failed to inspect collection '{settings.chunks_collection}': {err}"
            ) from err

        metadata: dict[str, Any] = info.config.metadata or {} if info.config else {}

        expected_checks: list[tuple[str, Any]] = [
            ("embedding_model", settings.embedding_model),
            ("dimension", settings.dimension),
            ("inference_backend", settings.inference_backend),
            ("schema_version", settings.schema_version),
        ]

        for prop_name, expected_val in expected_checks:
            actual_val = metadata.get(prop_name)
            if actual_val is None:
                raise QdrantMetadataMismatchError(
                    collection_name=settings.chunks_collection,
                    property_name=prop_name,
                    expected_value=expected_val,
                    actual_value=None,
                )
            # Normalize numeric types for comparison
            if isinstance(expected_val, int):
                try:
                    actual_val = int(actual_val)
                except (ValueError, TypeError):
                    pass

            if actual_val != expected_val:
                raise QdrantMetadataMismatchError(
                    collection_name=settings.chunks_collection,
                    property_name=prop_name,
                    expected_value=expected_val,
                    actual_value=actual_val,
                )

    @classmethod
    async def validate_meta_metadata(
        cls,
        client: AsyncQdrantClient,
        settings: QdrantSettings,
    ) -> None:
        """Validate existing meta collection metadata against configured settings."""
        try:
            info = await client.get_collection(settings.meta_collection)
        except Exception as err:
            raise QdrantInitializationError(
                f"Failed to inspect meta collection '{settings.meta_collection}': {err}"
            ) from err

        metadata: dict[str, Any] = info.config.metadata or {} if info.config else {}

        actual_version = metadata.get("schema_version")
        if actual_version is not None:
            try:
                actual_version = int(actual_version)
            except (ValueError, TypeError):
                pass
            if actual_version != settings.schema_version:
                raise QdrantMetadataMismatchError(
                    collection_name=settings.meta_collection,
                    property_name="schema_version",
                    expected_value=settings.schema_version,
                    actual_value=actual_version,
                )

    @classmethod
    async def create_chunks_collection(
        cls,
        client: AsyncQdrantClient,
        settings: QdrantSettings,
    ) -> None:
        """Create chunks collection with named vectors, indexes, and metadata."""
        vectors_config = {
            settings.dense_vector_name: models.VectorParams(
                size=settings.dimension,
                distance=models.Distance.COSINE,
            )
        }
        sparse_vectors_config = {
            settings.sparse_vector_name: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }
        metadata = {
            "embedding_model": settings.embedding_model,
            "dimension": settings.dimension,
            "inference_backend": settings.inference_backend,
            "schema_version": settings.schema_version,
        }

        try:
            await client.create_collection(
                collection_name=settings.chunks_collection,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
                metadata=metadata,
            )
        except Exception as err:
            raise QdrantInitializationError(
                f"Failed to create collection '{settings.chunks_collection}': {err}"
            ) from err

        # Create payload indexes before ingestion [S11]
        try:
            # 1. series: keyword with is_tenant=True [S10]
            await client.create_payload_index(
                collection_name=settings.chunks_collection,
                field_name="series",
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    is_tenant=True,
                ),
            )
            # 2. entity_id, entity_type, source_type: keyword
            for keyword_field in ("entity_id", "entity_type", "source_type"):
                await client.create_payload_index(
                    collection_name=settings.chunks_collection,
                    field_name=keyword_field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            # 3. season, episode, narrative_order: integer
            for int_field in ("season", "episode", "narrative_order"):
                await client.create_payload_index(
                    collection_name=settings.chunks_collection,
                    field_name=int_field,
                    field_schema=models.PayloadSchemaType.INTEGER,
                )
        except Exception as err:
            raise QdrantInitializationError(
                f"Failed to create payload indexes on "
                f"'{settings.chunks_collection}': {err}"
            ) from err

    @classmethod
    async def create_meta_collection(
        cls,
        client: AsyncQdrantClient,
        settings: QdrantSettings,
    ) -> None:
        """Create payload-only metadata collection for series manifests."""
        try:
            await client.create_collection(
                collection_name=settings.meta_collection,
                vectors_config={},
                metadata={"schema_version": settings.schema_version},
            )
        except Exception as err:
            raise QdrantInitializationError(
                f"Failed to create meta collection '{settings.meta_collection}': {err}"
            ) from err

    @classmethod
    async def ensure_collections_ready(
        cls,
        client: AsyncQdrantClient,
        settings: QdrantSettings,
    ) -> None:
        """Ensure chunks and meta collections are ready, initialized, and verified."""
        try:
            chunks_exists = await client.collection_exists(settings.chunks_collection)
        except Exception as err:
            raise QdrantInitializationError(
                f"Failed to reach Qdrant server: {err}"
            ) from err

        if not chunks_exists:
            logger.info(
                "Initializing Qdrant chunks collection '%s'...",
                settings.chunks_collection,
            )
            await cls.create_chunks_collection(client, settings)
        else:
            await cls.validate_chunks_metadata(client, settings)

        try:
            meta_exists = await client.collection_exists(settings.meta_collection)
        except Exception as err:
            raise QdrantInitializationError(
                f"Failed to reach Qdrant server: {err}"
            ) from err

        if not meta_exists:
            logger.info(
                "Initializing Qdrant meta collection '%s'...",
                settings.meta_collection,
            )
            await cls.create_meta_collection(client, settings)
        else:
            await cls.validate_meta_metadata(client, settings)
