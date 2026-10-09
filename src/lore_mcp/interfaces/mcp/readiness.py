"""Readiness checker verifying Qdrant reachability and manifest compatibility.

Uses an in-process TTL cache to prevent hammering Qdrant.
"""

from datetime import datetime
from typing import Any

from qdrant_client import AsyncQdrantClient

from lore_mcp.adapters.qdrant.initializer import QdrantInitializer
from lore_mcp.config import QdrantSettings
from lore_mcp.ports.clock import Clock, SystemClock


class ReadinessChecker:
    """Verifies vector store reachability and metadata compatibility.

    Caches check results in-process for a configured TTL (default ~30 seconds)
    to prevent hammering external dependencies during frequent container health checks.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        settings: QdrantSettings,
        clock: Clock | None = None,
        ttl_seconds: float = 30.0,
    ) -> None:
        self._client = client
        self._settings = settings
        self._clock = clock or SystemClock()
        self._ttl_seconds = ttl_seconds

        self._cached_result: tuple[bool, dict[str, Any]] | None = None
        self._cached_at: datetime | None = None

    async def check_readiness(
        self, force_refresh: bool = False
    ) -> tuple[bool, dict[str, Any]]:
        """Check Qdrant reachability and meta collection schema compatibility.

        Returns:
            A tuple of (is_ready, details_dict).
        """
        now = self._clock.now()
        if (
            not force_refresh
            and self._cached_result is not None
            and self._cached_at is not None
        ):
            age = (now - self._cached_at).total_seconds()
            if age < self._ttl_seconds:
                return self._cached_result

        try:
            # 1. Test basic network reachability
            await self._client.get_collections()
        except Exception as err:
            result = (
                False,
                {
                    "qdrant_reachable": False,
                    "manifest_compatible": False,
                    "error": str(err),
                },
            )
            self._cached_result = result
            self._cached_at = now
            return result

        try:
            # 2. Test metadata & manifest compatibility
            await QdrantInitializer.validate_meta_metadata(self._client, self._settings)
            result = (
                True,
                {
                    "qdrant_reachable": True,
                    "manifest_compatible": True,
                    "meta_collection": self._settings.meta_collection,
                    "schema_version": self._settings.schema_version,
                },
            )
        except Exception as err:
            result = (
                False,
                {
                    "qdrant_reachable": True,
                    "manifest_compatible": False,
                    "error": str(err),
                },
            )

        self._cached_result = result
        self._cached_at = now
        return result
