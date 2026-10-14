"""Unit tests for MCP error mapping, sanitization, and decorators."""

import json

import pytest
from fastmcp.exceptions import ToolError

from lore_mcp.application.errors import (
    ApplicationError,
    DataUnavailable,
    DependencyUnavailable,
    IncompatibleData,
    InvalidArgument,
    InvalidScope,
    RateLimited,
)
from lore_mcp.interfaces.mcp.errors import (
    McpErrorCode,
    handle_mcp_errors,
    map_exception_to_mcp_error,
    sanitize_error_message,
    validate_non_empty,
    validate_optional_positive_int,
)


@pytest.mark.unit
def test_sanitize_error_message_masks_secrets() -> None:
    """Accidental secrets or credentials in error messages are redacted."""
    raw = (
        "Failed connecting to host: password='SuperSecretPassword!' "
        "auth=Bearer abc123xyz"
    )
    sanitized = sanitize_error_message(raw)
    assert "SuperSecretPassword!" not in sanitized
    assert "abc123xyz" not in sanitized
    assert "[REDACTED]" in sanitized


@pytest.mark.unit
def test_map_invalid_scope() -> None:
    """InvalidScope maps to INVALID_SCOPE code with suggestions."""
    exc = InvalidScope("loky", suggestions=["loki"])
    resp = map_exception_to_mcp_error(exc)
    assert resp.code == McpErrorCode.INVALID_SCOPE
    assert resp.error_type == "invalid_scope"
    assert "loky" in resp.message
    assert resp.details["scope"] == "loky"
    assert resp.details["suggestions"] == ["loki"]


@pytest.mark.unit
def test_map_data_unavailable() -> None:
    """DataUnavailable maps to DATA_UNAVAILABLE code with resource_id."""
    exc = DataUnavailable("Chunk not found", resource_id="chunk_99")
    resp = map_exception_to_mcp_error(exc)
    assert resp.code == McpErrorCode.DATA_UNAVAILABLE
    assert resp.error_type == "data_unavailable"
    assert resp.details["resource_id"] == "chunk_99"


@pytest.mark.unit
def test_map_rate_limited() -> None:
    """RateLimited maps to RATE_LIMITED code with retry_after_seconds."""
    exc = RateLimited("Too many requests", retry_after_seconds=42.0)
    resp = map_exception_to_mcp_error(exc)
    assert resp.code == McpErrorCode.RATE_LIMITED
    assert resp.error_type == "rate_limited"
    assert resp.details["retry_after_seconds"] == 42.0


@pytest.mark.unit
def test_map_dependency_unavailable() -> None:
    """DependencyUnavailable maps to DEPENDENCY_UNAVAILABLE without internal secrets."""
    exc = DependencyUnavailable("qdrant", "connection url http://admin:pass@db:6333")
    resp = map_exception_to_mcp_error(exc)
    assert resp.code == McpErrorCode.DEPENDENCY_UNAVAILABLE
    assert resp.error_type == "dependency_unavailable"
    assert "qdrant" in resp.message
    assert "pass@db" not in resp.message
    assert resp.details == {"service_name": "qdrant"}


@pytest.mark.unit
def test_map_invalid_argument() -> None:
    """InvalidArgument maps to INVALID_ARGUMENT with argument_name."""
    exc = InvalidArgument("Argument must be positive", argument_name="season")
    resp = map_exception_to_mcp_error(exc)
    assert resp.code == McpErrorCode.INVALID_ARGUMENT
    assert resp.error_type == "invalid_argument"
    assert resp.details["argument_name"] == "season"


@pytest.mark.unit
def test_map_incompatible_data() -> None:
    """IncompatibleData maps to INCOMPATIBLE_DATA with versions."""
    exc = IncompatibleData("Version mismatch", expected_version=2, actual_version=1)
    resp = map_exception_to_mcp_error(exc)
    assert resp.code == McpErrorCode.INCOMPATIBLE_DATA
    assert resp.error_type == "incompatible_data"
    assert resp.details["expected_version"] == 2
    assert resp.details["actual_version"] == 1


@pytest.mark.unit
def test_map_unexpected_exception_masks_trace() -> None:
    """Unexpected internal exceptions map to INTERNAL_ERROR and hide trace/secrets."""
    exc = RuntimeError("Secret DB password=secret123 crashed the system")
    resp = map_exception_to_mcp_error(exc)
    assert resp.code == McpErrorCode.INTERNAL_ERROR
    assert resp.error_type == "internal_error"
    assert "secret123" not in resp.message
    assert "password" not in resp.message
    assert "internal error occurred" in resp.message
    assert resp.details == {}


@pytest.mark.unit
def test_map_generic_application_error() -> None:
    """Generic ApplicationError maps to application_error category."""
    exc = ApplicationError("Custom domain failure")
    resp = map_exception_to_mcp_error(exc)
    assert resp.code == McpErrorCode.INTERNAL_ERROR
    assert resp.error_type == "application_error"
    assert "Custom domain failure" in resp.message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_mcp_errors_decorator() -> None:
    """Decorator converts caught application exception to structured ToolError JSON."""

    @handle_mcp_errors
    async def sample_tool(val: str) -> str:
        if val == "bad":
            raise InvalidArgument("Invalid input", argument_name="val")
        if val == "crash":
            raise RuntimeError("Secret leaked: token=xyz999")
        return f"ok:{val}"

    # Success case
    assert await sample_tool("good") == "ok:good"

    # Handled application error
    with pytest.raises(ToolError) as exc_info:
        await sample_tool("bad")
    payload = json.loads(str(exc_info.value))
    assert payload["code"] == "INVALID_ARGUMENT"
    assert payload["details"]["argument_name"] == "val"

    # Handled unexpected internal error
    with pytest.raises(ToolError) as exc_info:
        await sample_tool("crash")
    payload = json.loads(str(exc_info.value))
    assert payload["code"] == "INTERNAL_ERROR"
    assert "xyz999" not in str(payload)


@pytest.mark.unit
def test_validation_helpers() -> None:
    """Validation helper functions enforce non-empty strings and positive ints."""
    assert validate_non_empty("  valid  ", "param") == "valid"
    with pytest.raises(InvalidArgument) as exc_info:
        validate_non_empty("   ", "param")
    assert exc_info.value.argument_name == "param"

    assert validate_optional_positive_int(None, "count") is None
    assert validate_optional_positive_int(5, "count") == 5
    with pytest.raises(InvalidArgument) as exc_info:
        validate_optional_positive_int(0, "count")
    assert exc_info.value.argument_name == "count"

    with pytest.raises(InvalidArgument) as exc_info:
        validate_optional_positive_int(-3, "count")
    assert exc_info.value.argument_name == "count"
