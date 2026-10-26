"""Redis client connection factory with TLS support and short timeouts (Task 12.1)."""

import logging
from typing import Any

import redis.asyncio as aioredis

from lore_mcp.config import RedisSettings

logger = logging.getLogger(__name__)


def create_redis_client(
    settings: RedisSettings | None = None,
) -> aioredis.Redis | None:
    """Create an asynchronous Redis client with TLS and configured timeouts.

    Returns None if settings.url is not configured or fails to initialize.
    """
    if settings is None or not settings.url or not settings.url.strip():
        logger.info("Redis URL is not configured. Redis adapter will operate disabled.")
        return None

    raw_url = settings.url.strip()

    # Convert Upstash REST URL (https://host) to standard TLS Redis (rediss://host:6379)
    if raw_url.startswith("https://"):
        host = raw_url.removeprefix("https://").rstrip("/")
        raw_url = f"rediss://{host}:6379"
    elif raw_url.startswith("http://"):
        host = raw_url.removeprefix("http://").rstrip("/")
        raw_url = f"redis://{host}:6379"

    kwargs: dict[str, Any] = {
        "socket_timeout": settings.timeout_seconds,
        "socket_connect_timeout": settings.connect_timeout_seconds,
        "decode_responses": True,
    }

    if settings.token:
        kwargs["password"] = settings.token.get_secret_value()

    # Enable SSL cert requirements bypass for managed cloud free tier if rediss://
    if raw_url.startswith("rediss://"):
        kwargs["ssl_cert_reqs"] = None

    try:
        client: aioredis.Redis = aioredis.from_url(raw_url, **kwargs)
        return client
    except Exception as err:
        logger.warning(
            "Failed to initialize Redis client from URL '%s': %s. "
            "Operating in degraded mode with cache disabled.",
            raw_url,
            err,
        )
        return None
