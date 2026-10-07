"""Contract test verifying QdrantVectorIndex satisfies VectorIndexContractSuite."""

import asyncio
from typing import Any

import pytest
from qdrant_client import AsyncQdrantClient, models

from lore_mcp.adapters.qdrant.index import QdrantVectorIndex
from lore_mcp.adapters.qdrant.initializer import QdrantInitializer
from lore_mcp.adapters.qdrant.registry import QdrantSeriesRegistry
from lore_mcp.config import QdrantSettings
from lore_mcp.ports.series_registry import SeriesRegistry
from lore_mcp.ports.vector_index import VectorIndex
from lore_mcp.testing import SeriesRegistryContractSuite, VectorIndexContractSuite


@pytest.fixture
def contract_settings() -> QdrantSettings:
    return QdrantSettings(
        url=":memory:",
        chunks_collection="contract_chunks",
        meta_collection="contract_meta",
        embedding_model="all-MiniLM-L6-v2",
        dimension=384,
        inference_backend="fastembed",
        schema_version=1,
    )


@pytest.mark.contract
class TestQdrantVectorIndexContract(VectorIndexContractSuite):
    @pytest.fixture
    def index(self, contract_settings: QdrantSettings) -> VectorIndex:
        async def _setup() -> VectorIndex:
            client = AsyncQdrantClient(":memory:")
            await QdrantInitializer.ensure_collections_ready(client, contract_settings)

            def fake_embed_models(
                points_or_queries: Any, is_query: bool = False, batch_size: int = 8
            ) -> Any:
                if is_query:
                    yield [0.0] * 384
                    return
                for p in points_or_queries:
                    yield models.PointStruct(
                        id=p.id,
                        vector={
                            "dense": [0.01] * 384,
                            "bm25": models.SparseVector(
                                indices=[0, 1],
                                values=[1.0, 0.5],
                            ),
                        },
                        payload=p.payload,
                    )

            client._embed_models = fake_embed_models  # type: ignore[assignment]
            return QdrantVectorIndex(client, contract_settings)

        return asyncio.run(_setup())


@pytest.mark.contract
class TestQdrantSeriesRegistryContract(SeriesRegistryContractSuite):
    @pytest.fixture
    def registry(self, contract_settings: QdrantSettings) -> SeriesRegistry:
        async def _setup() -> SeriesRegistry:
            client = AsyncQdrantClient(":memory:")
            await QdrantInitializer.ensure_collections_ready(client, contract_settings)
            return QdrantSeriesRegistry(client, contract_settings)

        return asyncio.run(_setup())
