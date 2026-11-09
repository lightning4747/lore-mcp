"""Automated keep-alive and anti-suspension script for Render and Qdrant Cloud.

Tasks 17.1, 17.3, 17.5:
1. Pings the Render service /healthz endpoint to prevent the 15-minute idle spin-down.
2. Pings Qdrant Cloud /collections to prevent the 7-day inactivity cluster suspension.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("lore_mcp.scripts.keep_alive")


def ping_healthz(service_url: str, timeout: float = 15.0) -> bool:
    """Send a lightweight GET request to the /healthz endpoint."""
    base_url = service_url.rstrip("/")
    target = f"{base_url}/healthz"
    logger.info("Pinging service healthz at: %s", target)

    start_time = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(target)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if response.status_code == 200:
            data: Any = (
                response.json()
                if response.headers.get("content-type", "").startswith(
                    "application/json"
                )
                else {}
            )
            version = (
                data.get("version", "unknown") if isinstance(data, dict) else "unknown"
            )
            logger.info(
                "Healthz ping successful! Status: 200 OK | "
                "Version: %s | Latency: %.1fms",
                version,
                elapsed_ms,
            )
            return True
        else:
            logger.error(
                "Healthz ping failed with HTTP %d: %s",
                response.status_code,
                response.text,
            )
            return False
    except httpx.RequestError as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "Healthz ping request failed after %.1fms: %s",
            elapsed_ms,
            exc,
        )
        return False


def ping_qdrant(
    qdrant_url: str,
    api_key: str | None = None,
    timeout: float = 15.0,
) -> bool:
    """Send a low-cost heartbeat to Qdrant Cloud to prevent 7-day suspension."""
    base_url = qdrant_url.rstrip("/")
    target = f"{base_url}/collections"
    logger.info("Sending anti-suspension heartbeat to Qdrant at: %s", target)

    headers = {"api-key": api_key} if api_key else {}
    start_time = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(target, headers=headers)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if response.status_code == 200:
            data: Any = (
                response.json()
                if response.headers.get("content-type", "").startswith(
                    "application/json"
                )
                else {}
            )
            collections = (
                data.get("result", {}).get("collections", [])
                if isinstance(data, dict) and isinstance(data.get("result"), dict)
                else []
            )
            logger.info(
                "Qdrant heartbeat successful! Status: 200 OK | "
                "Collections: %d | Latency: %.1fms",
                len(collections),
                elapsed_ms,
            )
            return True
        else:
            logger.error(
                "Qdrant heartbeat failed with HTTP %d: %s",
                response.status_code,
                response.text,
            )
            return False
    except httpx.RequestError as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "Qdrant heartbeat request failed after %.1fms: %s",
            elapsed_ms,
            exc,
        )
        return False


def run_keep_alive(
    service_url: str | None = None,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    timeout: float = 15.0,
) -> int:
    """Run keep-alive probes for configured services."""
    if not service_url and not qdrant_url:
        logger.warning("No service_url or qdrant_url provided. Nothing to ping.")
        return 0

    all_success = True

    if service_url:
        ok = ping_healthz(service_url=service_url, timeout=timeout)
        if not ok:
            all_success = False

    if qdrant_url:
        key = qdrant_api_key or os.environ.get("QDRANT_API_KEY")
        ok = ping_qdrant(qdrant_url=qdrant_url, api_key=key, timeout=timeout)
        if not ok:
            all_success = False

    return 0 if all_success else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lore MCP keep-alive and anti-suspension runner",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("RENDER_SERVICE_URL") or os.environ.get("SERVICE_URL"),
        help="Base service URL (e.g. https://lore-mcp.onrender.com)",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL") or os.environ.get("CLUSTER_URL"),
        help="Qdrant Cloud URL for anti-suspension heartbeat",
    )
    parser.add_argument(
        "--qdrant-api-key",
        default=os.environ.get("QDRANT_API_KEY"),
        help="Qdrant Cloud API Key",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP request timeout in seconds (default: 15.0)",
    )

    args = parser.parse_args()
    code = run_keep_alive(
        service_url=args.url,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        timeout=args.timeout,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
