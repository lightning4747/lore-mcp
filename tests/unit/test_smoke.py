import pytest

import lore_mcp


@pytest.mark.unit
def test_package_import():
    assert lore_mcp.__version__ == "0.1.0"
