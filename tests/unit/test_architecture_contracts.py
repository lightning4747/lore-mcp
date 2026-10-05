import subprocess

import pytest


@pytest.mark.unit
def test_import_linter_contracts_pass():
    result = subprocess.run(
        ["uv", "run", "lint-imports"],
        capture_output=True,
        text=True,
    )
    msg = f"import-linter failed:\n{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, msg
    assert "Contracts: 5 kept, 0 broken." in result.stdout
