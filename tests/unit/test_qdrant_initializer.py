"""Unit tests for Qdrant collection initializer and metadata validation."""

import pytest
from qdrant_client import AsyncQdrantClient, models

from lore_mcp.adapters.qdrant import (
    QdrantInitializer,
    QdrantMetadataMismatchError,
    create_async_qdrant_client,
)
from lore_mcp.application.errors import IncompatibleData
from lore_mcp.config import QdrantSettings


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fresh_collection_initialization() -> None:
    client = AsyncQdrantClient(":memory:")
    settings = QdrantSettings(
        url=":memory:",
        chunks_collection="test_chunks",
        meta_collection="test_meta",
        embedding_model="all-MiniLM-L6-v2",
        dimension=384,
        inference_backend="cloud",
        schema_version=1,
    )

    await QdrantInitializer.ensure_collections_ready(client, settings)

    # 1. Chunks collection verification
    assert await client.collection_exists("test_chunks")
    chunks_info = await client.get_collection("test_chunks")
    assert chunks_info.config is not None

    # Verify named dense vector
    vectors_config = chunks_info.config.params.vectors
    assert isinstance(vectors_config, dict)
    assert "dense" in vectors_config
    dense_params = vectors_config["dense"]
    assert dense_params.size == 384
    assert dense_params.distance == models.Distance.COSINE

    # Verify named sparse vector
    sparse_config = chunks_info.config.params.sparse_vectors
    assert isinstance(sparse_config, dict)
    assert "bm25" in sparse_config
    assert sparse_config["bm25"].modifier == models.Modifier.IDF

    # Verify metadata
    metadata = chunks_info.config.metadata
    assert metadata == {
        "embedding_model": "all-MiniLM-L6-v2",
        "dimension": 384,
        "inference_backend": "cloud",
        "schema_version": 1,
    }

    # 2. Meta collection verification
    assert await client.collection_exists("test_meta")
    meta_info = await client.get_collection("test_meta")
    assert meta_info.config is not None
    assert meta_info.config.metadata == {"schema_version": 1}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_idempotent_initialization() -> None:
    client = AsyncQdrantClient(":memory:")
    settings = QdrantSettings(
        url=":memory:",
        chunks_collection="idempotent_chunks",
        meta_collection="idempotent_meta",
    )

    # First call initializes
    await QdrantInitializer.ensure_collections_ready(client, settings)
    # Second call verifies without re-creating or failing
    await QdrantInitializer.ensure_collections_ready(client, settings)

    assert await client.collection_exists("idempotent_chunks")
    assert await client.collection_exists("idempotent_meta")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_metadata_mismatch_embedding_model() -> None:
    client = AsyncQdrantClient(":memory:")
    settings = QdrantSettings(
        url=":memory:",
        chunks_collection="mismatch_model_chunks",
        meta_collection="mismatch_meta",
        embedding_model="all-MiniLM-L6-v2",
    )

    # Pre-create collection with wrong embedding model
    await client.create_collection(
        collection_name="mismatch_model_chunks",
        vectors_config={
            "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
        },
        metadata={
            "embedding_model": "text-embedding-3-small",
            "dimension": 384,
            "inference_backend": "cloud",
            "schema_version": 1,
        },
    )

    with pytest.raises(QdrantMetadataMismatchError) as exc_info:
        await QdrantInitializer.ensure_collections_ready(client, settings)

    assert isinstance(exc_info.value, IncompatibleData)
    assert exc_info.value.property_name == "embedding_model"
    assert exc_info.value.expected_value == "all-MiniLM-L6-v2"
    assert exc_info.value.actual_value == "text-embedding-3-small"
    assert "Refusing to start" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_metadata_mismatch_dimension() -> None:
    client = AsyncQdrantClient(":memory:")
    settings = QdrantSettings(
        url=":memory:",
        chunks_collection="mismatch_dim_chunks",
        meta_collection="mismatch_meta",
        dimension=384,
    )

    await client.create_collection(
        collection_name="mismatch_dim_chunks",
        vectors_config={
            "dense": models.VectorParams(size=768, distance=models.Distance.COSINE)
        },
        metadata={
            "embedding_model": "all-MiniLM-L6-v2",
            "dimension": 768,
            "inference_backend": "cloud",
            "schema_version": 1,
        },
    )

    with pytest.raises(QdrantMetadataMismatchError) as exc_info:
        await QdrantInitializer.ensure_collections_ready(client, settings)

    assert exc_info.value.property_name == "dimension"
    assert exc_info.value.expected_value == 384
    assert exc_info.value.actual_value == 768


@pytest.mark.unit
@pytest.mark.asyncio
async def test_metadata_mismatch_backend() -> None:
    client = AsyncQdrantClient(":memory:")
    settings = QdrantSettings(
        url=":memory:",
        chunks_collection="mismatch_backend_chunks",
        meta_collection="mismatch_meta",
        inference_backend="cloud",
    )

    await client.create_collection(
        collection_name="mismatch_backend_chunks",
        vectors_config={
            "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
        },
        metadata={
            "embedding_model": "all-MiniLM-L6-v2",
            "dimension": 384,
            "inference_backend": "fastembed",
            "schema_version": 1,
        },
    )

    with pytest.raises(QdrantMetadataMismatchError) as exc_info:
        await QdrantInitializer.ensure_collections_ready(client, settings)

    assert exc_info.value.property_name == "inference_backend"
    assert exc_info.value.expected_value == "cloud"
    assert exc_info.value.actual_value == "fastembed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_metadata_mismatch_schema_version() -> None:
    client = AsyncQdrantClient(":memory:")
    settings = QdrantSettings(
        url=":memory:",
        chunks_collection="mismatch_ver_chunks",
        meta_collection="mismatch_meta",
        schema_version=1,
    )

    await client.create_collection(
        collection_name="mismatch_ver_chunks",
        vectors_config={
            "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
        },
        metadata={
            "embedding_model": "all-MiniLM-L6-v2",
            "dimension": 384,
            "inference_backend": "cloud",
            "schema_version": 2,
        },
    )

    with pytest.raises(QdrantMetadataMismatchError) as exc_info:
        await QdrantInitializer.ensure_collections_ready(client, settings)

    assert exc_info.value.property_name == "schema_version"
    assert exc_info.value.expected_version == 1
    assert exc_info.value.actual_version == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_metadata_missing_entirely() -> None:
    client = AsyncQdrantClient(":memory:")
    settings = QdrantSettings(
        url=":memory:",
        chunks_collection="no_meta_chunks",
        meta_collection="mismatch_meta",
    )

    await client.create_collection(
        collection_name="no_meta_chunks",
        vectors_config={
            "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
        },
    )

    with pytest.raises(QdrantMetadataMismatchError) as exc_info:
        await QdrantInitializer.ensure_collections_ready(client, settings)

    assert "metadata mismatch" in str(exc_info.value)


@pytest.mark.unit
def test_create_async_qdrant_client() -> None:
    settings = QdrantSettings(url=":memory:")
    client = create_async_qdrant_client(settings)
    assert isinstance(client, AsyncQdrantClient)
