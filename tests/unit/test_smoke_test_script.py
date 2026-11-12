"""Unit tests for scripts/smoke_test.py (Tasks 18.2, 18.3, 18.5)."""

from unittest.mock import patch

import httpx
import pytest
from scripts.smoke_test import (
    check_healthz,
    check_mcp_initialize,
    check_mcp_search,
    check_mcp_tools_list,
    parse_mcp_response,
    run_smoke_tests,
)


@pytest.mark.unit
def test_parse_mcp_response_json() -> None:
    res = httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        text='{"jsonrpc": "2.0", "result": {"ok": true}}',
    )
    data = parse_mcp_response(res)
    assert data["result"]["ok"] is True


@pytest.mark.unit
def test_parse_mcp_response_sse() -> None:
    sse_text = (
        "event: message\n"
        'data: {"jsonrpc": "2.0", "id": 1, "result": {"status": "success"}}\n\n'
    )
    res = httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        text=sse_text,
    )
    data = parse_mcp_response(res)
    assert data["result"]["status"] == "success"


@pytest.mark.unit
def test_parse_mcp_response_invalid_sse() -> None:
    res = httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        text="no data line here",
    )
    with pytest.raises(ValueError, match="No valid SSE data line found"):
        parse_mcp_response(res)


@pytest.mark.unit
def test_check_healthz_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/healthz"
        return httpx.Response(
            200, json={"status": "ok", "version": "v1.0", "ingest_version": 1}
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        data = check_healthz(client, "http://testserver")
        assert data["status"] == "ok"
        assert data["version"] == "v1.0"


@pytest.mark.unit
def test_check_healthz_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        with pytest.raises(RuntimeError, match="/healthz returned HTTP 500"):
            check_healthz(client, "http://testserver")


@pytest.mark.unit
def test_check_mcp_tools_list_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/mcp"
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "tools": [
                        {"name": "list_series"},
                        {"name": "search_series_lore"},
                        {"name": "get_entity"},
                        {"name": "get_chunk"},
                        {"name": "trace_event"},
                        {"name": "compare_events"},
                        {"name": "search_dialogue"},
                    ]
                },
            },
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        tools = check_mcp_tools_list(client, "http://testserver")
        assert "search_series_lore" in tools
        assert len(tools) == 7


@pytest.mark.unit
def test_check_mcp_tools_list_missing_tools() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "other"}]}},
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        with pytest.raises(RuntimeError, match="missing expected tools"):
            check_mcp_tools_list(client, "http://testserver")


@pytest.mark.unit
def test_check_mcp_search_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "result": {"content": [{"type": "text", "text": "result"}]},
            },
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        res = check_mcp_search(client, "http://testserver", series="loki")
        assert "result" in res


@pytest.mark.unit
def test_check_mcp_search_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "result": {"isError": True, "content": [{"text": "Failed"}]},
            },
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        with pytest.raises(RuntimeError, match="tools/call returned error result"):
            check_mcp_search(client, "http://testserver", series="loki")


@pytest.mark.unit
def test_check_mcp_initialize_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"serverInfo": {"name": "test", "version": "1.0"}},
            },
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        res = check_mcp_initialize(client, "http://testserver")
        assert "result" in res


@pytest.mark.unit
def test_run_smoke_tests_success() -> None:
    with (
        patch("scripts.smoke_test.check_healthz") as mock_h,
        patch("scripts.smoke_test.check_mcp_initialize") as mock_init,
        patch("scripts.smoke_test.check_mcp_tools_list") as mock_tools,
        patch("scripts.smoke_test.check_mcp_search") as mock_search,
    ):
        mock_h.return_value = {"status": "ok", "version": "1.0", "ingest_version": 1}
        mock_init.return_value = {"result": {"serverInfo": {"name": "lore"}}}
        mock_tools.return_value = ["list_series", "search_series_lore"]
        mock_search.return_value = {"result": {"ok": True}}

        code = run_smoke_tests("http://localhost:8080", mode="deploy", timeout=5.0)
        assert code == 0
        mock_h.assert_called_once()
        mock_init.assert_called_once()
        mock_tools.assert_called_once()
        mock_search.assert_called_once()


@pytest.mark.unit
def test_run_smoke_tests_with_retries_and_delay() -> None:
    with (
        patch("scripts.smoke_test.check_healthz") as mock_h,
        patch("scripts.smoke_test.check_mcp_initialize") as mock_init,
        patch("scripts.smoke_test.check_mcp_tools_list") as mock_tools,
    ):
        mock_h.return_value = {"status": "ok", "version": "1.0", "ingest_version": 1}
        mock_init.return_value = {"result": {"serverInfo": {"name": "lore"}}}
        mock_tools.return_value = ["list_series"]

        code = run_smoke_tests(
            "http://localhost:8080",
            mode="startup",
            retries=3,
            delay=0.01,
        )
        assert code == 0
        mock_h.assert_called_once()


@pytest.mark.unit
def test_smoke_test_cli_retries_and_delay() -> None:
    from scripts.smoke_test import main

    with (
        patch("sys.argv", ["smoke_test.py", "--retries", "15", "--delay", "2"]),
        patch("scripts.smoke_test.run_smoke_tests", return_value=0) as mock_run,
        patch("sys.exit") as mock_exit,
    ):
        main()
        mock_run.assert_called_once_with(
            base_url="http://localhost:8080",
            mode="startup",
            series="loki",
            path_token=None,
            timeout=30.0,
            retries=15,
            delay=2.0,
        )
        mock_exit.assert_called_once_with(0)
