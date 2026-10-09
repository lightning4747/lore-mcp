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
from lore_mcp.config import Settings, get_settings
from lore_mcp.interfaces.mcp.readiness import ReadinessChecker
from lore_mcp.ports.clock import Clock

logger = logging.getLogger(__name__)


def create_mcp_server(
    settings: Settings | None = None,
    client: AsyncQdrantClient | None = None,
    clock: Clock | None = None,
    readiness_checker: ReadinessChecker | None = None,
) -> FastMCP:
    """Create and configure the FastMCP application.

    Configures:
    - Lifespan startup and shutdown hooks for resource management.
    - Touchless liveness probe at /healthz (returns 200 OK immediately).
    - Cached readiness probe at /readyz checking Qdrant & manifest
      compatibility (~30s TTL).
    """
    app_settings = settings or get_settings()
    owns_client = client is None
    qdrant_client = client or create_async_qdrant_client(app_settings.qdrant)

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

    return mcp


def create_http_app(mcp: FastMCP) -> Starlette:
    """Create Streamable HTTP ASGI application with stateless_http=True at /mcp [S1]."""
    return mcp.http_app(path="/mcp", stateless_http=True)
