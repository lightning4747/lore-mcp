"""Unit tests for lore serve CLI subcommand and create_app factory
(Tasks 18.2, 18.4).
"""

from unittest.mock import patch

import pytest
from starlette.applications import Starlette

from lore_mcp.interfaces.cli.main import main, run_serve_command
from lore_mcp.interfaces.mcp import create_app


@pytest.mark.unit
def test_create_app_factory() -> None:
    app = create_app()
    assert isinstance(app, Starlette)


@pytest.mark.unit
def test_run_serve_command_invokes_uvicorn() -> None:
    with patch("uvicorn.run") as mock_run:
        exit_code = run_serve_command(host="127.0.0.1", port=9090, log_level="DEBUG")
        assert exit_code == 0
        mock_run.assert_called_once_with(
            "lore_mcp.interfaces.mcp.server:create_app",
            factory=True,
            host="127.0.0.1",
            port=9090,
            log_level="debug",
        )


@pytest.mark.unit
def test_cli_serve_subcommand_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    with (
        patch("lore_mcp.interfaces.cli.main.run_serve_command") as mock_serve,
        patch("sys.exit") as mock_exit,
    ):
        mock_serve.return_value = 0
        main(["serve"])
        mock_serve.assert_called_once_with(
            host="0.0.0.0",
            port=8080,
            log_level="info",
        )
        mock_exit.assert_called_once_with(0)


@pytest.mark.unit
def test_cli_serve_subcommand_custom_args() -> None:
    with (
        patch("lore_mcp.interfaces.cli.main.run_serve_command") as mock_serve,
        patch("sys.exit") as mock_exit,
    ):
        mock_serve.return_value = 0
        main(
            ["serve", "--host", "10.0.0.1", "--port", "5000", "--log-level", "warning"]
        )
        mock_serve.assert_called_once_with(
            host="10.0.0.1",
            port=5000,
            log_level="warning",
        )
        mock_exit.assert_called_once_with(0)


@pytest.mark.unit
def test_cli_serve_subcommand_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "7070")
    monkeypatch.setenv("LOG_LEVEL", "error")
    with (
        patch("lore_mcp.interfaces.cli.main.run_serve_command") as mock_serve,
        patch("sys.exit") as mock_exit,
    ):
        mock_serve.return_value = 0
        main(["serve"])
        mock_serve.assert_called_once_with(
            host="0.0.0.0",
            port=7070,
            log_level="error",
        )
        mock_exit.assert_called_once_with(0)
