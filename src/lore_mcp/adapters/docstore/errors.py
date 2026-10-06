"""Exceptions for filesystem document store adapter."""


class DocstoreError(Exception):
    """Base exception for all document store adapter errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message: str = message

    def __str__(self) -> str:
        return self.message


class DocumentCorruptedError(DocstoreError):
    """Raised when a stored document fails integrity or hash verification."""

    def __init__(self, path: str, expected_hash: str, actual_hash: str) -> None:
        self.path: str = path
        self.expected_hash: str = expected_hash
        self.actual_hash: str = actual_hash
        super().__init__(
            f"Document corrupted at '{path}'. "
            f"Expected hash {expected_hash}, got {actual_hash}"
        )
