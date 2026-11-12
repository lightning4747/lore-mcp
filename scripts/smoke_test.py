"""Smoke test utility script for container startup checks and post-deploy validation.

Tasks 18.2, 18.3, 18.5:
- Probes touchless /healthz endpoint.
- Probes MCP initialize over Streamable HTTP POST /mcp.
- Probes MCP tools/list to verify registered retrieval tools.
- (Deploy mode) Executes a real search_series_lore query.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

try:
    import httpx
except ImportError:
    sys.stderr.write(
        "Error: 'httpx' is required to run smoke_test.py. "
        "Install it via 'pip install httpx'.\n"
    )
    sys.exit(1)


def parse_mcp_response(response: httpx.Response) -> dict[str, Any]:
    """Parse JSON or SSE text/event-stream response from MCP Streamable HTTP."""
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        for line in response.text.splitlines():
            if line.startswith("data: "):
                data_str = line.removeprefix("data: ").strip()
                if data_str:
                    return json.loads(data_str)
        raise ValueError(f"No valid SSE data line found in response:\n{response.text}")
    return response.json()


def check_healthz(
    client: httpx.Client,
    base_url: str,
    path_token: str | None = None,
) -> dict[str, Any]:
    """Check touchless liveness probe at /healthz."""
    url = f"{base_url.rstrip('/')}/healthz"
    if path_token:
        url = f"{url}?token={path_token}"

    res = client.get(url, timeout=10.0)
    if res.status_code != 200:
        raise RuntimeError(f"/healthz returned HTTP {res.status_code}: {res.text}")

    data = res.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"/healthz status not ok: {data}")
    return data


def check_mcp_initialize(
    client: httpx.Client,
    base_url: str,
    path_token: str | None = None,
) -> dict[str, Any]:
    """Send MCP JSON-RPC initialize request to /mcp."""
    url = f"{base_url.rstrip('/')}/mcp"
    if path_token:
        url = f"{url}?token={path_token}"

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "lore-smoke-test", "version": "1.0.0"},
        },
    }
    headers = {"Accept": "application/json, text/event-stream"}
    res = client.post(url, json=payload, headers=headers, timeout=10.0)
    if res.status_code != 200:
        raise RuntimeError(
            f"MCP initialize failed with HTTP {res.status_code}: {res.text}"
        )

    data = parse_mcp_response(res)
    if "result" not in data:
        raise RuntimeError(f"MCP initialize response missing result: {data}")
    return data


def check_mcp_tools_list(
    client: httpx.Client,
    base_url: str,
    path_token: str | None = None,
) -> list[str]:
    """Send MCP JSON-RPC tools/list request to /mcp."""
    url = f"{base_url.rstrip('/')}/mcp"
    if path_token:
        url = f"{url}?token={path_token}"

    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }
    headers = {"Accept": "application/json, text/event-stream"}
    res = client.post(url, json=payload, headers=headers, timeout=10.0)
    if res.status_code != 200:
        raise RuntimeError(f"tools/list failed with HTTP {res.status_code}: {res.text}")

    data = parse_mcp_response(res)
    tools = data.get("result", {}).get("tools", [])
    tool_names: list[str] = [
        str(t["name"]) for t in tools if isinstance(t, dict) and "name" in t
    ]

    expected = [
        "list_series",
        "search_series_lore",
        "get_entity",
        "get_chunk",
        "trace_event",
        "compare_events",
        "search_dialogue",
    ]
    missing = [name for name in expected if name not in tool_names]
    if missing:
        raise RuntimeError(f"tools/list is missing expected tools: {missing}")

    return tool_names


def check_mcp_search(
    client: httpx.Client,
    base_url: str,
    series: str = "loki",
    query: str = "Loki",
    path_token: str | None = None,
) -> dict[str, Any]:
    """Execute a test search_series_lore tool invocation via tools/call."""
    url = f"{base_url.rstrip('/')}/mcp"
    if path_token:
        url = f"{url}?token={path_token}"

    payload = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "search_series_lore",
            "arguments": {
                "query": query,
                "series": series,
                "max_results": 2,
            },
        },
    }
    headers = {"Accept": "application/json, text/event-stream"}
    res = client.post(url, json=payload, headers=headers, timeout=15.0)
    if res.status_code != 200:
        raise RuntimeError(f"tools/call failed with HTTP {res.status_code}: {res.text}")

    data = parse_mcp_response(res)
    result = data.get("result", {})
    if result.get("isError"):
        raise RuntimeError(f"tools/call returned error result: {result}")
    return data


def run_smoke_tests(
    base_url: str,
    mode: str = "startup",
    series: str = "loki",
    path_token: str | None = None,
    timeout: float = 30.0,
    retries: int | None = None,
    delay: float | None = None,
) -> int:
    """Run smoke test suite with retries until service becomes available."""
    print(
        f"\n[smoke-test] Starting smoke test suite against {base_url} (mode={mode})..."
    )
    start_time = time.monotonic()
    last_err: Exception | None = None

    effective_delay = delay if delay is not None else 1.0
    effective_max_time = timeout
    if retries is not None and delay is not None:
        effective_max_time = max(timeout, retries * delay)

    with httpx.Client() as client:
        # 1. Warm-up poll on /healthz
        print("[smoke-test] 1. Probing /healthz...")
        health_data: dict[str, Any] | None = None
        attempt = 0
        while time.monotonic() - start_time < effective_max_time:
            attempt += 1
            try:
                health_data = check_healthz(client, base_url, path_token=path_token)
                break
            except Exception as err:
                last_err = err
                if retries is not None and attempt >= retries:
                    break
                time.sleep(effective_delay)

        if health_data is None:
            print(
                f"[smoke-test] FAILED: /healthz probe timed out "
                f"after {effective_max_time}s ({attempt} attempts): {last_err}"
            )
            return 1

        print(
            f"[smoke-test]    OK (version={health_data.get('version')}, "
            f"ingest_version={health_data.get('ingest_version')})"
        )

        # 2. MCP initialize
        print("[smoke-test] 2. Probing MCP initialize...")
        try:
            init_data = check_mcp_initialize(client, base_url, path_token=path_token)
            server_info = init_data.get("result", {}).get("serverInfo", {})
            print(f"[smoke-test]    OK (serverInfo={server_info})")
        except Exception as err:
            print(f"[smoke-test] FAILED: MCP initialize error: {err}")
            return 1

        # 3. MCP tools/list
        print("[smoke-test] 3. Probing MCP tools/list...")
        try:
            tool_names = check_mcp_tools_list(client, base_url, path_token=path_token)
            print(f"[smoke-test]    OK (found {len(tool_names)} tools: {tool_names})")
        except Exception as err:
            print(f"[smoke-test] FAILED: tools/list error: {err}")
            return 1

        # 4. Search query (Deploy mode only)
        if mode == "deploy":
            print(
                f"[smoke-test] 4. Probing search_series_lore for series='{series}'..."
            )
            try:
                check_mcp_search(
                    client,
                    base_url,
                    series=series,
                    query="Loki",
                    path_token=path_token,
                )
                print("[smoke-test]    OK (search execution returned valid result)")
            except Exception as err:
                print(f"[smoke-test] FAILED: search_series_lore test error: {err}")
                return 1

    print("\n[smoke-test] PASSED: All smoke test checks completed successfully.\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Lore MCP Smoke Test Runner")
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Base service URL (e.g. http://localhost:8080)",
    )
    parser.add_argument(
        "--mode",
        choices=["startup", "deploy"],
        default="startup",
        help="Test mode: 'startup' (container startup) or 'deploy' (post-deploy)",
    )
    parser.add_argument(
        "--series",
        default="loki",
        help="Series identifier for search tool call in deploy mode",
    )
    parser.add_argument(
        "--path-token",
        default=None,
        help="Optional path token for secured MCP access",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for readiness check",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        help="Maximum retry attempts for initial health check",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Delay in seconds between health check retries",
    )

    args = parser.parse_args()
    code = run_smoke_tests(
        base_url=args.url,
        mode=args.mode,
        series=args.series,
        path_token=args.path_token,
        timeout=args.timeout,
        retries=args.retries,
        delay=args.delay,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
