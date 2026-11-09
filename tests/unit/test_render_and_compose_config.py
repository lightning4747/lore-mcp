"""Unit tests for docker-compose, render.yaml, and keep_alive script.

Covers Tasks 16.3, 17.1, 17.3.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
import yaml
from scripts.keep_alive import ping_healthz, ping_qdrant, run_keep_alive


@pytest.mark.unit
def test_docker_compose_structure() -> None:
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists(), "docker-compose.yml should exist"

    with open(compose_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "services" in data
    services = data["services"]

    # Verify lore-mcp service
    assert "lore-mcp" in services
    mcp = services["lore-mcp"]
    assert mcp.get("build", {}).get("target") == "runtime"
    assert "qdrant" in mcp.get("depends_on", {})
    assert "redis" in mcp.get("depends_on", {})

    # Verify qdrant service
    assert "qdrant" in services
    qdrant = services["qdrant"]
    assert "image" in qdrant
    assert "qdrant_data:/qdrant/storage" in qdrant.get("volumes", [])
    assert "healthcheck" in qdrant

    # Verify redis service
    assert "redis" in services
    redis = services["redis"]
    assert "image" in redis
    assert "redis_data:/data" in redis.get("volumes", [])
    assert "healthcheck" in redis

    # Verify volumes
    assert "volumes" in data
    assert "qdrant_data" in data["volumes"]
    assert "redis_data" in data["volumes"]


@pytest.mark.unit
def test_render_blueprint_structure() -> None:
    render_path = Path("render.yaml")
    assert render_path.exists(), "render.yaml should exist"

    with open(render_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "services" in data
    assert len(data["services"]) >= 1

    svc = data["services"][0]
    assert svc["type"] == "web"
    assert svc["name"] == "lore-mcp"
    assert svc["runtime"] == "docker"
    assert svc["dockerfilePath"] == "./Dockerfile"
    assert svc["healthCheckPath"] == "/healthz"
    assert svc["plan"] == "free"

    # Verify sensitive environment variables are non-synced
    env_vars = {item["key"]: item for item in svc.get("envVars", [])}
    assert env_vars["QDRANT_URL"]["sync"] is False
    assert env_vars["QDRANT_API_KEY"]["sync"] is False
    assert env_vars["REDIS_URL"]["sync"] is False
    assert env_vars["TAVILY_API_KEY"]["sync"] is False


@pytest.mark.unit
def test_ping_healthz_success() -> None:
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"status": "ok", "version": "0.1.0"}

    with patch("httpx.Client.get", return_value=mock_response):
        result = ping_healthz("https://lore-mcp.onrender.com")
        assert result is True


@pytest.mark.unit
def test_ping_healthz_non_200() -> None:
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 503
    mock_response.text = "Service Unavailable"

    with patch("httpx.Client.get", return_value=mock_response):
        result = ping_healthz("https://lore-mcp.onrender.com")
        assert result is False


@pytest.mark.unit
def test_ping_healthz_network_error() -> None:
    with patch(
        "httpx.Client.get",
        side_effect=httpx.ConnectError("Failed to connect"),
    ):
        result = ping_healthz("https://lore-mcp.onrender.com")
        assert result is False


@pytest.mark.unit
def test_ping_qdrant_success() -> None:
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"result": {"collections": [{"name": "lore"}]}}

    with patch("httpx.Client.get", return_value=mock_response) as mock_get:
        result = ping_qdrant(
            qdrant_url="https://cluster.qdrant.io:6333",
            api_key="secret-key",
        )
        assert result is True
        mock_get.assert_called_once()
        headers = mock_get.call_args[1].get("headers", {})
        assert headers.get("api-key") == "secret-key"


@pytest.mark.unit
def test_ping_qdrant_failure() -> None:
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 403
    mock_response.text = "Forbidden"

    with patch("httpx.Client.get", return_value=mock_response):
        result = ping_qdrant("https://cluster.qdrant.io:6333")
        assert result is False


@pytest.mark.unit
def test_ping_qdrant_network_error() -> None:
    with patch(
        "httpx.Client.get",
        side_effect=httpx.ConnectTimeout("Timeout"),
    ):
        result = ping_qdrant("https://cluster.qdrant.io:6333")
        assert result is False


@pytest.mark.unit
def test_run_keep_alive_no_urls() -> None:
    assert run_keep_alive() == 0


@pytest.mark.unit
def test_run_keep_alive_all_success() -> None:
    with (
        patch("scripts.keep_alive.ping_healthz", return_value=True),
        patch("scripts.keep_alive.ping_qdrant", return_value=True),
    ):
        assert (
            run_keep_alive(
                service_url="https://lore.onrender.com",
                qdrant_url="https://qdrant.cloud:6333",
            )
            == 0
        )


@pytest.mark.unit
def test_run_keep_alive_partial_failure() -> None:
    with (
        patch("scripts.keep_alive.ping_healthz", return_value=True),
        patch("scripts.keep_alive.ping_qdrant", return_value=False),
    ):
        assert (
            run_keep_alive(
                service_url="https://lore.onrender.com",
                qdrant_url="https://qdrant.cloud:6333",
            )
            == 1
        )
