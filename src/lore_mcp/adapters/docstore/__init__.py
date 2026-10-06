"""Content-addressed filesystem document store adapter."""

from lore_mcp.adapters.docstore.errors import DocstoreError, DocumentCorruptedError
from lore_mcp.adapters.docstore.store import FilesystemDocumentStore, compute_sha256

__all__ = [
    "DocstoreError",
    "DocumentCorruptedError",
    "FilesystemDocumentStore",
    "compute_sha256",
]
