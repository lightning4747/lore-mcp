"""Unit tests for Redis key generation and invalidation (Tasks 12.2-12.4)."""

import pytest

from lore_mcp.adapters.redis.keys import (
    build_budget_key,
    build_retrieval_cache_key,
    build_web_cache_key,
)


@pytest.mark.unit
class TestRedisKeys:
    """Verifies key format, parameter sensitivity, normalization, and invalidation."""

    def test_retrieval_key_format(self) -> None:
        key = build_retrieval_cache_key(
            tool_name="search_series_lore",
            series="loki",
            ingest_version=1,
            query="Glorious Purpose",
            season=1,
            episode=1,
        )
        assert key.startswith("lore:v1:ret:loki:1:")
        parts = key.split(":")
        assert len(parts) == 6
        assert len(parts[5]) == 64  # SHA256 hex digest length

    def test_retrieval_key_normalization_invariance(self) -> None:
        key1 = build_retrieval_cache_key(
            tool_name="search_series_lore",
            series="Loki",
            ingest_version=1,
            query="  GLORIOUS PURPOSE  ",
            entity=" TVA ",
        )
        key2 = build_retrieval_cache_key(
            tool_name="search_series_lore",
            series="loki",
            ingest_version=1,
            query="glorious purpose",
            entity="tva",
        )
        assert key1 == key2

    def test_parameter_change_produces_new_key(self) -> None:
        base_key = build_retrieval_cache_key(
            tool_name="search_series_lore",
            series="loki",
            ingest_version=1,
            query="coffee",
            season=1,
            episode=1,
            max_results=8,
        )

        # Changing season
        k_season = build_retrieval_cache_key(
            tool_name="search_series_lore",
            series="loki",
            ingest_version=1,
            query="coffee",
            season=2,
            episode=1,
            max_results=8,
        )
        assert k_season != base_key

        # Changing max_results
        k_limit = build_retrieval_cache_key(
            tool_name="search_series_lore",
            series="loki",
            ingest_version=1,
            query="coffee",
            season=1,
            episode=1,
            max_results=20,
        )
        assert k_limit != base_key

        # Changing schema_version
        k_schema = build_retrieval_cache_key(
            tool_name="search_series_lore",
            series="loki",
            ingest_version=1,
            query="coffee",
            season=1,
            episode=1,
            max_results=8,
            schema_version=2,
        )
        assert k_schema != base_key

    def test_ingest_version_invalidates_key(self) -> None:
        key_v1 = build_retrieval_cache_key(
            tool_name="search_series_lore",
            series="loki",
            ingest_version=1,
            query="timeline",
        )
        key_v2 = build_retrieval_cache_key(
            tool_name="search_series_lore",
            series="loki",
            ingest_version=2,
            query="timeline",
        )
        assert key_v1.startswith("lore:v1:ret:loki:1:")
        assert key_v2.startswith("lore:v1:ret:loki:2:")
        assert key_v1 != key_v2

    def test_web_cache_key_format(self) -> None:
        key = build_web_cache_key(series="global", query="  MCU Timeline Order  ")
        assert key.startswith("lore:v1:web:global:")
        parts = key.split(":")
        assert len(parts) == 5
        assert len(parts[4]) == 64

    def test_budget_key_format(self) -> None:
        daily_key = build_budget_key("web_search_daily", "2026-10-26")
        assert daily_key == "lore:v1:budget:web_search_daily:2026-10-26"

        rate_key = build_budget_key("global_rate", "2026-10-26-12-00")
        assert rate_key == "lore:v1:budget:global_rate:2026-10-26-12-00"
