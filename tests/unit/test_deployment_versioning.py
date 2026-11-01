"""Unit tests for deployment versioning and healthz endpoint (Task 18.4)."""

import logging
from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient
from starlette.testclient import TestClient

from lore_mcp.config import Settings, VersionSettings
from lore_mcp.interfaces.mcp import create_http_app, create_mcp_server


@pytest.mark.unit
def test_version_settings_defaults() -> None:
    version = VersionSettings()
    assert version.app_version == "0.1.0-dev"
    assert version.ingest_version == 1


@pytest.mark.unit
def test_version_settings_validation() -> None:
    with pytest.raises(ValueError, match="ingest_version must be greater than 0"):
        VersionSettings(ingest_version=0)

    with pytest.raises(ValueError, match="ingest_version must be greater than 0"):
        VersionSettings(ingest_version=-1)


@pytest.mark.unit
def test_version_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_VERSION", "git-commit-abc1234")
    monkeypatch.setenv("INGEST_VERSION", "3")
    version = VersionSettings()
    assert version.app_version == "git-commit-abc1234"
    assert version.ingest_version == 3


@pytest.mark.unit
def test_healthz_returns_version_metadata() -> None:
    settings = Settings(
        version=VersionSettings(
            app_version="git-sha-test-999",
            ingest_version=42,
        )
    )
    mock_client = AsyncMock(spec=AsyncQdrantClient)
    mcp = create_mcp_server(settings=settings, client=mock_client)
    app = create_http_app(mcp)

    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "git-sha-test-999"
        assert data["ingest_version"] == 42


@pytest.mark.unit
def test_server_lifespan_logs_version(caplog: pytest.LogCaptureFixture) -> None:
    settings = Settings(
        version=VersionSettings(
            app_version="git-sha-log-check",
            ingest_version=7,
        )
    )
    mock_client = AsyncMock(spec=AsyncQdrantClient)
    mcp = create_mcp_server(settings=settings, client=mock_client)
    app = create_http_app(mcp)

    with caplog.at_level(logging.INFO), TestClient(app) as client:
        res = client.get("/healthz")
        assert res.status_code == 200

    assert "git-sha-log-check" in caplog.text
    assert "ingest_version=7" in caplog.text
