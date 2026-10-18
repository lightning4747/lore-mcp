import logging
from typing import Any

import httpx
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from lore_mcp.application.errors import (
    ApplicationError,
    DependencyUnavailable,
    IncompatibleData,
)

logger = logging.getLogger(__name__)


class QdrantAdapterError(ApplicationError):
    """Base exception for all Qdrant adapter errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


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


class QdrantConnectionError(DependencyUnavailable):
    """Raised when connection or network communication with Qdrant fails."""

    def __init__(self, message: str) -> None:
        super().__init__(service_name="qdrant", message=message)


class QdrantOperationError(QdrantAdapterError):
    """Raised when an operation against Qdrant fails due to operational or query
    errors.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


def is_qdrant_connection_error(exc: BaseException) -> bool:
    """Determine whether an exception is a genuine connectivity or timeout failure."""
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.TimeoutException,
            httpx.NetworkError,
            ConnectionError,
            TimeoutError,
            ResponseHandlingException,
        ),
    ):
        return True

    if isinstance(exc, UnexpectedResponse) and exc.status_code in (502, 503, 504):
        return True

    cause = exc.__cause__ or exc.__context__
    if cause is not None and is_qdrant_connection_error(cause):
        return True

    return False


def wrap_qdrant_error(
    err: Exception,
    context_msg: str,
    log: logging.Logger | None = None,
) -> QdrantConnectionError | QdrantOperationError:
    """Classify and log a Qdrant exception, returning domain exception.

    Genuine connection or timeout failures produce QdrantConnectionError
    (a DependencyUnavailable). Operational, query, or validation failures
    produce QdrantOperationError (an ApplicationError) and are logged at
    ERROR level with stack trace.
    """
    active_logger = log or logger
    if is_qdrant_connection_error(err):
        active_logger.warning(
            "Qdrant connectivity failure during %s: %s", context_msg, err
        )
        return QdrantConnectionError(f"{context_msg}: {err}")

    active_logger.error(
        "Qdrant operational failure during %s: %s", context_msg, err, exc_info=True
    )
    return QdrantOperationError(f"{context_msg}: {err}")
