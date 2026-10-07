"""Exceptions for Qdrant vector store adapter."""

from typing import Any

from lore_mcp.application.errors import DependencyUnavailable, IncompatibleData


class QdrantAdapterError(Exception):
    """Base exception for all Qdrant adapter errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message: str = message

    def __str__(self) -> str:
        return self.message


class QdrantMetadataMismatchError(IncompatibleData):
    """Raised when existing collection metadata does not match configuration."""

    def __init__(
        self,
        collection_name: str,
        property_name: str,
        expected_value: Any,
        actual_value: Any,
    ) -> None:
        self.collection_name = collection_name
        self.property_name = property_name
        self.expected_value = expected_value
        self.actual_value = actual_value

        message = (
            f"Collection '{collection_name}' metadata mismatch for '{property_name}': "
            f"configured {expected_value!r}, but found {actual_value!r}. "
            f"Refusing to start."
        )
        exp_ver = int(expected_value) if property_name == "schema_version" else None
        act_ver = (
            int(actual_value)
            if property_name == "schema_version" and actual_value is not None
            else None
        )
        super().__init__(
            message=message,
            expected_version=exp_ver,
            actual_version=act_ver,
        )


class QdrantInitializationError(DependencyUnavailable):
    """Raised when initializing Qdrant collections fails."""

    def __init__(self, message: str) -> None:
        super().__init__(service_name="qdrant", message=message)


class QdrantOperationError(DependencyUnavailable):
    """Raised when an operation against Qdrant fails."""

    def __init__(self, message: str) -> None:
        super().__init__(service_name="qdrant", message=message)
