"""FastMCP application factory, Streamable HTTP serving at /mcp,
liveness (/healthz), and readiness (/readyz) endpoints [S1].
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP
from qdrant_client import AsyncQdrantClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse

from lore_mcp.adapters.qdrant.client import create_async_qdrant_client
from lore_mcp.adapters.qdrant.entity_registry import QdrantEntityRegistry
from lore_mcp.adapters.qdrant.index import QdrantVectorIndex
from lore_mcp.adapters.qdrant.registry import QdrantSeriesRegistry
from lore_mcp.application.chunk_service import ChunkService
from lore_mcp.application.entity_service import EntityService
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.application.retrieval_service import LoreRetrievalService
from lore_mcp.application.scope import ScopeResolver
from lore_mcp.config import Settings, get_settings
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.search import EvidenceBundle
from lore_mcp.interfaces.mcp.models import EntityDetails, SeriesSummary
from lore_mcp.interfaces.mcp.readiness import ReadinessChecker
from lore_mcp.ports.budget import BudgetCounter
from lore_mcp.ports.cache import Cache
from lore_mcp.ports.clock import Clock
from lore_mcp.ports.entity_registry import EntityRegistry
from lore_mcp.ports.series_registry import SeriesRegistry
from lore_mcp.ports.vector_index import VectorIndex
from lore_mcp.ports.web_search import WebSearch

logger = logging.getLogger(__name__)


def create_mcp_server(
    settings: Settings | None = None,
    client: AsyncQdrantClient | None = None,
    registry: SeriesRegistry | None = None,
    entity_registry: EntityRegistry | None = None,
    vector_index: VectorIndex | None = None,
    retrieval_service: LoreRetrievalService | None = None,
    entity_service: EntityService | None = None,
    chunk_service: ChunkService | None = None,
    cache: Cache | None = None,
    web_search: WebSearch | None = None,
    budget_counter: BudgetCounter | None = None,
    clock: Clock | None = None,
    readiness_checker: ReadinessChecker | None = None,
) -> FastMCP:
    """Create and configure the FastMCP application.

    Configures:
    - Lifespan startup and shutdown hooks for resource management.
    - Touchless liveness probe at /healthz (returns 200 OK immediately).
    - Cached readiness probe at /readyz checking Qdrant & manifest
      compatibility (~30s TTL).
    - Tool `list_series`: discover registered series ids, display names,
      aliases, and kind.
    - Tool `search_series_lore`: scope-resolved, cached, hybrid retrieval with
      provenance and temporal context.
    - Tool `get_entity`: resolve canonical entity by id/alias and retrieve
      associated evidence with provenance.
    - Tool `get_chunk`: fetch a full chunk by id to expand truncated evidence,
      verifying series scope [S11].
    """

    app_settings = settings or get_settings()
    owns_client = client is None
    qdrant_client = client or create_async_qdrant_client(app_settings.qdrant)

    series_registry = registry or QdrantSeriesRegistry(
        client=qdrant_client,
        settings=app_settings.qdrant,
    )

    entity_reg = entity_registry or QdrantEntityRegistry(
        client=qdrant_client,
        settings=app_settings.qdrant,
    )

    index = vector_index or QdrantVectorIndex(
        client=qdrant_client,
        settings=app_settings.qdrant,
    )

    scope_resolver = ScopeResolver(
        registry=series_registry,
        clock=clock,
    )

    hybrid_retriever = HybridRetriever(
        index=index,
        rrf_k=app_settings.qdrant.rrf_k,
    )

    retrieval_orch = retrieval_service or LoreRetrievalService(
        scope_resolver=scope_resolver,
        retriever=hybrid_retriever,
        cache=cache,
        web_search=web_search,
        budget_counter=budget_counter,
        clock=clock,
        web_search_settings=app_settings.web_search,
        limits_settings=app_settings.limits,
    )

    entity_svc = entity_service or EntityService(
        scope_resolver=scope_resolver,
        retriever=hybrid_retriever,
        entity_registry=entity_reg,
        limits_settings=app_settings.limits,
    )

    chunk_svc = chunk_service or ChunkService(
        scope_resolver=scope_resolver,
        vector_index=index,
    )

    checker = readiness_checker or ReadinessChecker(
        client=qdrant_client,
        settings=app_settings.qdrant,
        clock=clock,
        ttl_seconds=30.0,
    )

    @asynccontextmanager
    async def server_lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
        logger.info("Initializing lore-mcp server resources...")
        try:
            yield {
                "settings": app_settings,
                "qdrant_client": qdrant_client,
                "series_registry": series_registry,
                "entity_registry": entity_reg,
                "readiness_checker": checker,
            }

        finally:
            logger.info("Shutting down lore-mcp server resources...")
            if owns_client:
                try:
                    await qdrant_client.close()
                except Exception as err:
                    logger.warning(
                        "Error closing Qdrant client during shutdown: %s", err
                    )

    mcp = FastMCP(
        name="lore-mcp",
        instructions=(
            "Lore retrieval server providing context-aware evidence from "
            "franchise wikis and transcripts."
        ),
        lifespan=server_lifespan,
    )

    # Attach checker for direct access if needed
    mcp._readiness_checker = checker  # type: ignore[attr-defined]

    # 1. Liveness probe: touches NO dependencies
    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"}, status_code=200)

    # 2. Readiness probe: checks Qdrant reachability and manifest
    # compatibility (cached ~30s)
    @mcp.custom_route("/readyz", methods=["GET"])
    async def readyz(request: Request) -> JSONResponse:
        is_ready, details = await checker.check_readiness()
        status_code = 200 if is_ready else 503
        return JSONResponse(
            {"ready": is_ready, "details": details}, status_code=status_code
        )

    # 3. Discovery Tool: list_series
    @mcp.tool()
    async def list_series() -> list[SeriesSummary]:
        """List all indexed franchises/series with their canonical IDs, display names,
        aliases, and kind.

        Use this tool to discover valid series IDs before querying lore.
        """
        manifests = await series_registry.get_manifests()
        return [
            SeriesSummary(
                series_id=str(m.series_id),
                display_name=m.display_name,
                aliases=list(m.aliases),
                kind=m.kind,
            )
            for m in manifests
        ]

    # 4. Retrieval Tool: search_series_lore
    @mcp.tool()
    async def search_series_lore(
        query: str,
        series: str,
        season: int | None = None,
        episode: int | None = None,
        max_season: int | None = None,
        max_episode: int | None = None,
        include_unscoped: bool = False,
        entity: str | None = None,
        max_results: int | None = None,
    ) -> EvidenceBundle:
        """Search lore evidence for a specific series/franchise scope.

        Executes scope-resolved, cached, hybrid dense+sparse retrieval with
        heuristic reranking, spoiler bounds, sufficiency check, and chronological
        timeline context.
        """
        return await retrieval_orch.search(
            query=query,
            series=series,
            season=season,
            episode=episode,
            max_season=max_season,
            max_episode=max_episode,
            include_unscoped=include_unscoped,
            entity=entity,
            max_results=max_results,
        )

    # 5. Entity Tool: get_entity
    @mcp.tool()
    async def get_entity(
        entity: str,
        series: str,
        max_results: int | None = 8,
    ) -> EntityDetails:
        """Resolve a canonical entity by ID or alias, retrieve associated evidence,
        and return entity metadata with source provenance.
        """
        return await entity_svc.get_entity(
            entity=entity,
            series=series,
            max_results=max_results,
        )

    # 6. Chunk Tool: get_chunk
    @mcp.tool()
    async def get_chunk(
        chunk_id: str,
        series: str,
    ) -> Chunk:
        """Fetch a full chunk by ID to expand truncated evidence,
        verifying series scope ownership [S11].
        """
        return await chunk_svc.get_chunk(
            chunk_id=chunk_id,
            series=series,
        )

    return mcp


def create_http_app(mcp: FastMCP) -> Starlette:
    """Create Streamable HTTP ASGI application with stateless_http=True at /mcp [S1]."""
    return mcp.http_app(path="/mcp", stateless_http=True)
