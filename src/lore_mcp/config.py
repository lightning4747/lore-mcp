"""Application configuration and environment settings."""

import functools
from typing import Literal

from pydantic import (
    AliasChoices,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantSettings(BaseSettings):
    """Qdrant vector store connection and collection configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    url: str = Field(
        default="http://localhost:6333",
        validation_alias=AliasChoices("QDRANT_URL", "CLUSTER_URL"),
        description="Qdrant cluster endpoint or local server URL",
    )
    api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("QDRANT_API_KEY", "QDRUNT_API_KEY"),
        description="Qdrant Cloud API key for authenticated access",
    )
    chunks_collection: str = Field(
        default="lore_chunks",
        validation_alias=AliasChoices("QDRANT_CHUNKS_COLLECTION"),
        description="Qdrant collection name for vector chunks",
    )
    meta_collection: str = Field(
        default="lore_meta",
        validation_alias=AliasChoices("QDRANT_META_COLLECTION"),
        description="Qdrant collection name for series manifests and metadata",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        validation_alias=AliasChoices("QDRANT_EMBEDDING_MODEL"),
        description="Dense embedding model name",
    )
    dimension: int = Field(
        default=384,
        validation_alias=AliasChoices("QDRANT_DIMENSION"),
        description="Dense vector dimension",
    )
    inference_backend: Literal["cloud", "fastembed"] = Field(
        default="cloud",
        validation_alias=AliasChoices("QDRANT_INFERENCE_BACKEND"),
        description="Embedding inference backend (cloud or fastembed)",
    )
    schema_version: int = Field(
        default=1,
        validation_alias=AliasChoices("QDRANT_SCHEMA_VERSION"),
        description="Vector store schema version",
    )
    dense_vector_name: str = Field(
        default="dense",
        validation_alias=AliasChoices("QDRANT_DENSE_VECTOR_NAME"),
        description="Named dense vector identifier in collection",
    )
    sparse_vector_name: str = Field(
        default="bm25",
        validation_alias=AliasChoices("QDRANT_SPARSE_VECTOR_NAME"),
        description="Named sparse vector identifier in collection",
    )

    @field_validator("dimension", "schema_version")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Value must be greater than 0")
        return value


class RedisSettings(BaseSettings):
    """Redis / Upstash caching and budget counter configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("UPSTASH_REDIS_REST_URL", "REDIS_URL"),
        description="Upstash REST URL or standard redis:// connection string",
    )
    token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("UPSTASH_REDIS_REST_TOKEN", "REDIS_TOKEN"),
        description="Upstash REST access token or Redis password",
    )
    timeout_seconds: float = Field(
        default=2.0,
        validation_alias=AliasChoices("REDIS_TIMEOUT_SECONDS"),
        description="Redis socket/read timeout in seconds",
    )
    connect_timeout_seconds: float = Field(
        default=2.0,
        validation_alias=AliasChoices("REDIS_CONNECT_TIMEOUT_SECONDS"),
        description="Redis connection timeout in seconds",
    )

    @field_validator("timeout_seconds", "connect_timeout_seconds")
    @classmethod
    def validate_positive_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Timeout values must be greater than 0")
        return value


class WebSearchSettings(BaseSettings):
    """Dynamic web search credentials and budget configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("TAVILY_API_KEY", "TAVILY_API"),
        description="Tavily API key for web search fallback",
    )
    daily_budget: int = Field(
        default=30,
        validation_alias=AliasChoices("WEB_SEARCH_DAILY_BUDGET"),
        description="Max fallback web search queries per day",
    )

    @field_validator("daily_budget")
    @classmethod
    def validate_daily_budget(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("daily_budget must be greater than 0")
        return value


class SecuritySettings(BaseSettings):
    """MCP server access mode, path token, and Origin allowlist."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    access_mode: Literal["open", "path_token"] = Field(
        default="open",
        validation_alias=AliasChoices("MCP_ACCESS_MODE", "ACCESS_MODE"),
        description=(
            "Access mode: 'open' for authless dev, 'path_token' for secured URL"
        ),
    )
    path_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("MCP_PATH_TOKEN", "PATH_TOKEN"),
        description="Secret path token required when access_mode is 'path_token'",
    )
    origin_allowlist: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("MCP_ORIGIN_ALLOWLIST", "ORIGIN_ALLOWLIST"),
        description="Allowed Origin header values for browser/connector requests",
    )

    @field_validator("origin_allowlist", mode="before")
    @classmethod
    def parse_origin_allowlist(cls, value: object) -> list[str]:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @model_validator(mode="after")
    def validate_path_token(self) -> "SecuritySettings":
        if self.access_mode == "path_token":
            if not self.path_token or not self.path_token.get_secret_value().strip():
                raise ValueError(
                    "path_token must be provided and non-empty when "
                    "access_mode is 'path_token'"
                )
        return self


class RequestLimitsSettings(BaseSettings):
    """Request limits, query constraints, and execution deadlines."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    default_max_results: int = Field(
        default=8,
        validation_alias=AliasChoices("DEFAULT_MAX_RESULTS"),
        description="Default number of search candidates returned",
    )
    hard_max_results: int = Field(
        default=20,
        validation_alias=AliasChoices("HARD_MAX_RESULTS"),
        description="Hard cap on maximum search candidates requested",
    )
    max_query_length: int = Field(
        default=500,
        validation_alias=AliasChoices("MAX_QUERY_LENGTH"),
        description="Maximum allowed character length for query text",
    )
    deadline_seconds: float = Field(
        default=15.0,
        validation_alias=AliasChoices("REQUEST_DEADLINE_SECONDS"),
        description="Total per-request deadline in seconds",
    )

    @model_validator(mode="after")
    def validate_limits(self) -> "RequestLimitsSettings":
        if self.default_max_results <= 0:
            raise ValueError("default_max_results must be greater than 0")
        if self.hard_max_results < self.default_max_results:
            raise ValueError("hard_max_results cannot be less than default_max_results")
        if self.max_query_length <= 0:
            raise ValueError("max_query_length must be greater than 0")
        if self.deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be greater than 0")
        return self


class Settings(BaseSettings):
    """Root application configuration aggregating all component settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    port: int = Field(
        default=8080,
        validation_alias=AliasChoices("PORT"),
        description="Port for FastMCP streamable HTTP server",
    )
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("LOG_LEVEL"),
        description="Application logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    limits: RequestLimitsSettings = Field(default_factory=RequestLimitsSettings)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()
