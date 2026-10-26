"""Hierarchical Redis key generators for cache and budgets (Tasks 12.2-12.4)."""

import hashlib
import json
from typing import Any


def build_retrieval_cache_key(
    tool_name: str,
    series: str,
    ingest_version: int,
    query: str,
    season: int | None = None,
    episode: int | None = None,
    max_season: int | None = None,
    max_episode: int | None = None,
    include_unscoped: bool = False,
    entity: str | None = None,
    source_type: str | None = None,
    max_results: int = 8,
    schema_version: int = 1,
) -> str:
    """Generate deterministic retrieval cache key with ingest_version and query hash.

    Format: lore:v1:ret:{series}:{ingest_version}:{query_hash}
    """
    canonical_payload: dict[str, Any] = {
        "tool_name": tool_name.strip(),
        "query": query.strip().lower(),
        "season": season,
        "episode": episode,
        "max_season": max_season,
        "max_episode": max_episode,
        "include_unscoped": include_unscoped,
        "entity": entity.strip().lower() if entity else None,
        "source_type": source_type.strip().lower() if source_type else None,
        "max_results": max_results,
        "schema_version": schema_version,
    }
    encoded = json.dumps(canonical_payload, sort_keys=True).encode("utf-8")
    query_hash = hashlib.sha256(encoded).hexdigest()
    normalized_series = series.strip().lower()

    return f"lore:v1:ret:{normalized_series}:{ingest_version}:{query_hash}"


def build_web_cache_key(series: str, query: str) -> str:
    """Generate web search cache key with query hash.

    Format: lore:v1:web:{series}:{query_hash}
    """
    normalized_query = query.strip().lower().encode("utf-8")
    query_hash = hashlib.sha256(normalized_query).hexdigest()
    normalized_series = series.strip().lower()

    return f"lore:v1:web:{normalized_series}:{query_hash}"


def build_budget_key(name: str, window: str) -> str:
    """Generate atomic budget counter key.

    Format: lore:v1:budget:{name}:{window}
    """
    normalized_name = name.strip().lower()
    normalized_window = window.strip()

    return f"lore:v1:budget:{normalized_name}:{normalized_window}"
