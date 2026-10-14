"""Structured error handling and taxonomy mapping for FastMCP tools."""

import functools
import logging
import re
from collections.abc import Callable, Coroutine
from enum import StrEnum
from typing import Any

from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field

from lore_mcp.application.errors import (
    ApplicationError,
    DataUnavailable,
    DependencyUnavailable,
    IncompatibleData,
    InvalidArgument,
    InvalidScope,
    RateLimited,
)

logger = logging.getLogger(__name__)


_SECRET_PATTERN = re.compile(
    r"(?i)((?:api[_-]?key|password|secret|token|auth|credential)\s*[:=]\s*(?:bearer\s+)?['\"]?[^\s'\",]+|bearer\s+[^\s'\",]+)",
)


def sanitize_error_message(message: str) -> str:
    """Mask accidental secrets, tokens, or credentials in error messages."""
    return _SECRET_PATTERN.sub("[REDACTED]", message)


class McpErrorCode(StrEnum):
    """Machine-readable error codes mapped from domain taxonomy."""

    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    INVALID_SCOPE = "INVALID_SCOPE"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    INCOMPATIBLE_DATA = "INCOMPATIBLE_DATA"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class McpErrorResponse(BaseModel):
    """Structured error payload returned to MCP clients."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: McpErrorCode = Field(description="Structured machine-readable error code")
    error_type: str = Field(description="Error category classification")
    message: str = Field(description="Safe human-readable error description")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured contextual details without internal secrets",
    )


def map_exception_to_mcp_error(exc: Exception) -> McpErrorResponse:
    """Map domain and application exceptions to structured MCP error payloads.

    Never exposes raw exceptions, internal traces, or credentials to clients.
    """
    if isinstance(exc, InvalidScope):
        return McpErrorResponse(
            code=McpErrorCode.INVALID_SCOPE,
            error_type="invalid_scope",
            message=sanitize_error_message(exc.message),
            details={
                "scope": exc.scope,
                "suggestions": exc.suggestions,
            },
        )

    if isinstance(exc, DataUnavailable):
        details: dict[str, Any] = {}
        if exc.resource_id:
            details["resource_id"] = exc.resource_id
        return McpErrorResponse(
            code=McpErrorCode.DATA_UNAVAILABLE,
            error_type="data_unavailable",
            message=sanitize_error_message(exc.message),
            details=details,
        )

    if isinstance(exc, RateLimited):
        details = {}
        if exc.retry_after_seconds is not None:
            details["retry_after_seconds"] = exc.retry_after_seconds
        return McpErrorResponse(
            code=McpErrorCode.RATE_LIMITED,
            error_type="rate_limited",
            message=sanitize_error_message(exc.message),
            details=details,
        )

    if isinstance(exc, DependencyUnavailable):
        return McpErrorResponse(
            code=McpErrorCode.DEPENDENCY_UNAVAILABLE,
            error_type="dependency_unavailable",
            message=(
                f"Backing dependency '{exc.service_name}' is currently unavailable."
            ),
            details={"service_name": exc.service_name},
        )

    if isinstance(exc, InvalidArgument):
        details = {}
        if exc.argument_name:
            details["argument_name"] = exc.argument_name
        return McpErrorResponse(
            code=McpErrorCode.INVALID_ARGUMENT,
            error_type="invalid_argument",
            message=sanitize_error_message(exc.message),
            details=details,
        )

    if isinstance(exc, IncompatibleData):
        return McpErrorResponse(
            code=McpErrorCode.INCOMPATIBLE_DATA,
            error_type="incompatible_data",
            message="Data compatibility issue encountered.",
            details={
                "expected_version": exc.expected_version,
                "actual_version": exc.actual_version,
            },
        )

    if isinstance(exc, ApplicationError):
        return McpErrorResponse(
            code=McpErrorCode.INTERNAL_ERROR,
            error_type="application_error",
            message=sanitize_error_message(exc.message),
            details={},
        )

    # Any unknown or unexpected internal exception
    logger.exception("Unexpected error in MCP tool: %s", exc)
    return McpErrorResponse(
        code=McpErrorCode.INTERNAL_ERROR,
        error_type="internal_error",
        message="An internal error occurred while processing the request.",
        details={},
    )


def handle_mcp_errors[**P, T](
    fn: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[P, Coroutine[Any, Any, T]]:
    """Catch application and internal errors, raising structured ToolError."""

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:
            mcp_err = map_exception_to_mcp_error(exc)
            raise ToolError(mcp_err.model_dump_json()) from exc

    return wrapper


def validate_non_empty(value: str, name: str) -> str:
    """Validate that a string argument is non-empty and non-whitespace."""
    if not isinstance(value, str) or not value.strip():
        raise InvalidArgument(
            f"Argument '{name}' must be a non-empty string.",
            argument_name=name,
        )
    return value.strip()


def validate_optional_positive_int(value: int | None, name: str) -> int | None:
    """Validate that an optional integer argument is >= 1 if provided."""
    if value is not None:
        if not isinstance(value, int) or value < 1:
            raise InvalidArgument(
                f"Argument '{name}' must be a positive integer (>= 1).",
                argument_name=name,
            )
    return value
