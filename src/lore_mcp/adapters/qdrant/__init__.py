"""Qdrant vector store adapter implementing [S4], [S5], [S9], [S10], [S11]."""

from lore_mcp.adapters.qdrant.client import create_async_qdrant_client
from lore_mcp.adapters.qdrant.entity_registry import (
    QdrantEntityRegistry,
    to_entity_point_id,
)
from lore_mcp.adapters.qdrant.errors import (
    QdrantAdapterError,
    QdrantConnectionError,
    QdrantInitializationError,
    QdrantMetadataMismatchError,
    QdrantOperationError,
    is_qdrant_connection_error,
    wrap_qdrant_error,
)
from lore_mcp.adapters.qdrant.event_registry import (
    QdrantEventRegistry,
    to_event_point_id,
)
from lore_mcp.adapters.qdrant.filters import (
    build_filter_from_query,
    build_metadata_filter,
)
from lore_mcp.adapters.qdrant.index import (
    QdrantVectorIndex,
    resolve_embedding_model_name,
    to_point_id,
)
from lore_mcp.adapters.qdrant.initializer import QdrantInitializer
from lore_mcp.adapters.qdrant.registry import (
    QdrantSeriesRegistry,
    to_manifest_point_id,
)

__all__ = [
    "QdrantAdapterError",
    "QdrantConnectionError",
    "QdrantEntityRegistry",
    "QdrantEventRegistry",
    "QdrantInitializationError",
    "QdrantInitializer",
    "QdrantMetadataMismatchError",
    "QdrantOperationError",
    "QdrantSeriesRegistry",
    "QdrantVectorIndex",
    "build_filter_from_query",
    "build_metadata_filter",
    "create_async_qdrant_client",
    "is_qdrant_connection_error",
    "resolve_embedding_model_name",
    "to_entity_point_id",
    "to_event_point_id",
    "to_manifest_point_id",
    "to_point_id",
    "wrap_qdrant_error",
]
