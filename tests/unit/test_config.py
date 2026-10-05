"""Unit tests for typed configuration system."""

import pytest
from pydantic import SecretStr, ValidationError

from lore_mcp.config import (
    QdrantSettings,
    RedisSettings,
    RequestLimitsSettings,
    SecuritySettings,
    Settings,
    WebSearchSettings,
    get_settings,
)


@pytest.mark.unit
def test_default_settings_instantiation() -> None:
    settings = Settings()
    assert settings.port == 8080
    assert settings.log_level in {"DEBUG", "INFO", "WARNING", "ERROR"}
    assert settings.qdrant.chunks_collection == "lore_chunks"
    assert settings.qdrant.meta_collection == "lore_meta"
    assert settings.security.access_mode in {"open", "path_token"}
    assert settings.limits.default_max_results == 8
    assert settings.limits.hard_max_results == 20


@pytest.mark.unit
def test_secret_str_redaction() -> None:
    qdrant = QdrantSettings(api_key=SecretStr("super_secret_qdrant_key"))
    redis = RedisSettings(token=SecretStr("super_secret_redis_token"))
    web = WebSearchSettings(api_key=SecretStr("super_secret_tavily_key"))
    sec = SecuritySettings(
        access_mode="path_token", path_token=SecretStr("my_path_token")
    )

    for secret_obj, raw_val in [
        (qdrant.api_key, "super_secret_qdrant_key"),
        (redis.token, "super_secret_redis_token"),
        (web.api_key, "super_secret_tavily_key"),
        (sec.path_token, "my_path_token"),
    ]:
        assert secret_obj is not None
        assert raw_val not in str(secret_obj)
        assert raw_val not in repr(secret_obj)
        assert secret_obj.get_secret_value() == raw_val


@pytest.mark.unit
def test_security_settings_fail_fast_on_missing_path_token() -> None:
    # Open mode does not require path_token
    open_sec = SecuritySettings(access_mode="open", path_token=None)
    assert open_sec.access_mode == "open"

    # path_token mode requires non-empty path_token
    with pytest.raises(ValidationError, match="path_token must be provided"):
        SecuritySettings(access_mode="path_token", path_token=None)

    with pytest.raises(ValidationError, match="path_token must be provided"):
        SecuritySettings(access_mode="path_token", path_token=SecretStr("   "))

    valid_sec = SecuritySettings(
        access_mode="path_token", path_token=SecretStr("valid-token-123")
    )
    assert valid_sec.path_token is not None
    assert valid_sec.path_token.get_secret_value() == "valid-token-123"


@pytest.mark.unit
def test_security_settings_origin_allowlist_parsing() -> None:
    # String with comma separation
    sec_str = SecuritySettings(
        origin_allowlist="https://claude.ai, https://chat.anthropic.com"  # type: ignore[arg-type]
    )
    assert sec_str.origin_allowlist == [
        "https://claude.ai",
        "https://chat.anthropic.com",
    ]

    # List of strings
    sec_list = SecuritySettings(origin_allowlist=["http://localhost:3000"])
    assert sec_list.origin_allowlist == ["http://localhost:3000"]


@pytest.mark.unit
def test_request_limits_validation() -> None:
    # Valid limits
    limits = RequestLimitsSettings(
        default_max_results=5,
        hard_max_results=10,
        max_query_length=200,
        deadline_seconds=10.0,
    )
    assert limits.default_max_results == 5
    assert limits.hard_max_results == 10

    # Non-positive default_max_results
    with pytest.raises(
        ValidationError, match="default_max_results must be greater than 0"
    ):
        RequestLimitsSettings(default_max_results=0)

    # hard_max_results less than default_max_results
    with pytest.raises(
        ValidationError,
        match="hard_max_results cannot be less than default_max_results",
    ):
        RequestLimitsSettings(default_max_results=10, hard_max_results=5)

    # Non-positive max_query_length
    with pytest.raises(
        ValidationError, match="max_query_length must be greater than 0"
    ):
        RequestLimitsSettings(max_query_length=0)

    # Non-positive deadline_seconds
    with pytest.raises(
        ValidationError, match="deadline_seconds must be greater than 0"
    ):
        RequestLimitsSettings(deadline_seconds=0.0)


@pytest.mark.unit
def test_redis_timeout_validation() -> None:
    with pytest.raises(ValidationError, match="Timeout values must be greater than 0"):
        RedisSettings(timeout_seconds=0.0)

    with pytest.raises(ValidationError, match="Timeout values must be greater than 0"):
        RedisSettings(connect_timeout_seconds=-1.0)


@pytest.mark.unit
def test_web_search_budget_validation() -> None:
    with pytest.raises(ValidationError, match="daily_budget must be greater than 0"):
        WebSearchSettings(daily_budget=0)

    valid_web = WebSearchSettings(daily_budget=50)
    assert valid_web.daily_budget == 50


@pytest.mark.unit
def test_alias_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLUSTER_URL", "https://custom-cluster.qdrant.io")
    monkeypatch.setenv("QDRUNT_API_KEY", "typo_qdrunt_api_key")
    monkeypatch.setenv("TAVILY_API", "tavily_legacy_api_key")

    qdrant = QdrantSettings()
    assert qdrant.url == "https://custom-cluster.qdrant.io"
    assert qdrant.api_key is not None
    assert qdrant.api_key.get_secret_value() == "typo_qdrunt_api_key"

    web = WebSearchSettings()
    assert web.api_key is not None
    assert web.api_key.get_secret_value() == "tavily_legacy_api_key"


@pytest.mark.unit
def test_get_settings_caching() -> None:
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
