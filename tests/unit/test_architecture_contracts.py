"""Unit tests verifying architectural contracts and enforcement mechanisms."""

import subprocess
from pathlib import Path

import pytest


@pytest.mark.unit
def test_import_linter_contracts_pass() -> None:
    result = subprocess.run(
        ["uv", "run", "lint-imports"],
        capture_output=True,
        text=True,
    )
    msg = f"import-linter failed:\n{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, msg
    assert "Contracts: 5 kept, 0 broken." in result.stdout


@pytest.mark.unit
@pytest.mark.parametrize(
    "relative_file,violation_import,expected_broken",
    [
        (
            "src/lore_mcp/domain/__init__.py",
            "import lore_mcp.application",
            "Hexagonal layers architecture BROKEN",
        ),
        (
            "src/lore_mcp/adapters/qdrant/__init__.py",
            "import lore_mcp.adapters.redis",
            "Adapter independence BROKEN",
        ),
        (
            "src/lore_mcp/domain/__init__.py",
            "import httpx",
            "Domain imports no I/O libraries BROKEN",
        ),
        (
            "src/lore_mcp/interfaces/cli/__init__.py",
            "import fastmcp",
            "Ingest CLI imports no FastMCP BROKEN",
        ),
    ],
)
def test_deliberate_import_violations_fail_linter(
    relative_file: str,
    violation_import: str,
    expected_broken: str,
) -> None:
    target_path = Path(relative_file)
    original_content = target_path.read_text(encoding="utf-8")
    try:
        target_path.write_text(
            original_content + f"\n{violation_import}\n", encoding="utf-8"
        )
        result = subprocess.run(
            ["uv", "run", "lint-imports"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, (
            f"Expected import-linter to fail for {violation_import}, "
            f"but passed with output:\n{result.stdout}"
        )
        assert (
            expected_broken in result.stdout
        ), f"Expected {expected_broken!r} in stdout, got:\n{result.stdout}"
    finally:
        target_path.write_text(original_content, encoding="utf-8")
