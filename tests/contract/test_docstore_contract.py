"""Contract test verifying that FilesystemDocumentStore satisfies DocumentStore port."""

from pathlib import Path

import pytest

from lore_mcp.adapters.docstore import FilesystemDocumentStore
from lore_mcp.testing import DocumentStoreContractSuite


@pytest.mark.contract
class TestFilesystemDocumentStoreContract(DocumentStoreContractSuite):
    @pytest.fixture
    def store(self, tmp_path: Path) -> FilesystemDocumentStore:
        return FilesystemDocumentStore(base_dir=tmp_path)
