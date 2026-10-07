"""Qdrant client factory functions."""

from qdrant_client import AsyncQdrantClient

from lore_mcp.config import QdrantSettings


def create_async_qdrant_client(settings: QdrantSettings) -> AsyncQdrantClient:
    """Create and configure an AsyncQdrantClient instance from settings."""
    if settings.url == ":memory:":
        return AsyncQdrantClient(":memory:")

    api_key_str = (
        settings.api_key.get_secret_value() if settings.api_key is not None else None
    )
    return AsyncQdrantClient(
        url=settings.url,
        api_key=api_key_str,
    )
