"""Unit tests for application error taxonomy."""

import pytest

from lore_mcp.application import (
    ApplicationError,
    DataUnavailable,
    DependencyUnavailable,
    IncompatibleData,
    InvalidArgument,
    InvalidScope,
    RateLimited,
)


@pytest.mark.unit
def test_application_error_hierarchy() -> None:
    errors = [
        InvalidArgument("Invalid value"),
        InvalidScope("unknown-series"),
        DataUnavailable("Missing entity"),
        RateLimited("Budget exceeded"),
        DependencyUnavailable("qdrant", "connection refused"),
        IncompatibleData("Schema mismatch"),
    ]
    for err in errors:
        assert isinstance(err, ApplicationError)
        assert isinstance(err, Exception)
        assert len(str(err)) > 0


@pytest.mark.unit
def test_invalid_argument() -> None:
    err = InvalidArgument("max_results must be > 0", argument_name="max_results")
    assert err.message == "max_results must be > 0"
    assert err.argument_name == "max_results"
    assert str(err) == "max_results must be > 0"


@pytest.mark.unit
def test_invalid_scope_with_suggestions() -> None:
    # With suggestions
    err = InvalidScope("loky", suggestions=["loki", "loki-season-2"])
    assert err.scope == "loky"
    assert err.suggestions == ["loki", "loki-season-2"]
    assert "Did you mean: 'loki', 'loki-season-2'?" in str(err)

    # Without suggestions
    err_no_sug = InvalidScope("completely-unknown")
    assert err_no_sug.suggestions == []
    assert str(err_no_sug) == "Unknown or invalid series scope 'completely-unknown'."

    # Suggestions capped at 3
    err_many = InvalidScope("test", suggestions=["s1", "s2", "s3", "s4", "s5"])
    assert len(err_many.suggestions) == 3
    assert err_many.suggestions == ["s1", "s2", "s3"]


@pytest.mark.unit
def test_data_unavailable() -> None:
    err = DataUnavailable("Chunk not found", resource_id="chunk-123")
    assert err.resource_id == "chunk-123"
    assert str(err) == "Chunk not found"


@pytest.mark.unit
def test_rate_limited() -> None:
    err = RateLimited("Daily web search budget exhausted", retry_after_seconds=3600.0)
    assert err.retry_after_seconds == 3600.0
    assert str(err) == "Daily web search budget exhausted"


@pytest.mark.unit
def test_dependency_unavailable() -> None:
    err = DependencyUnavailable("redis", "timeout connecting to host")
    assert err.service_name == "redis"
    assert "Dependency 'redis' is unavailable: timeout connecting to host" in str(err)


@pytest.mark.unit
def test_incompatible_data() -> None:
    err = IncompatibleData(
        "Manifest schema version unsupported",
        expected_version=1,
        actual_version=2,
    )
    assert err.expected_version == 1
    assert err.actual_version == 2
    assert "Manifest schema version unsupported" in str(err)
