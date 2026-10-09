"""Scope resolution service for series-locked lore retrieval."""

import difflib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from lore_mcp.application.errors import InvalidScope
from lore_mcp.domain.entities import SeriesManifest
from lore_mcp.domain.types import SeriesId
from lore_mcp.ports.clock import Clock, SystemClock
from lore_mcp.ports.series_registry import SeriesRegistry


@dataclass(frozen=True)
class ResolvedScope:
    """Resolved and validated series scope paired with its active manifest."""

    series_id: SeriesId
    manifest: SeriesManifest


class ScopeResolver:
    """Resolves and caches series scope with alias expansion and suggestion ranking."""

    def __init__(
        self,
        registry: SeriesRegistry,
        clock: Clock | None = None,
        ttl_seconds: float = 60.0,
    ) -> None:
        self._registry = registry
        self._clock: Clock = clock or SystemClock()
        self._ttl_seconds = ttl_seconds
        self._cached_manifests: dict[SeriesId, SeriesManifest] = {}
        self._alias_to_series: dict[str, set[SeriesId]] = {}
        self._cached_at: datetime | None = None

    def _is_cache_valid(self) -> bool:
        if self._cached_at is None:
            return False
        age = (self._clock.now() - self._cached_at).total_seconds()
        return age < self._ttl_seconds

    async def _refresh_cache(self) -> None:
        manifests = await self._registry.get_manifests()
        self._cached_manifests = {m.series_id: m for m in manifests}

        alias_map: dict[str, set[SeriesId]] = {}
        for manifest in manifests:
            s_id = manifest.series_id
            keys: set[str] = {
                str(s_id).lower(),
                manifest.display_name.lower(),
            }
            for alias in manifest.aliases:
                cleaned = alias.strip().lower()
                if cleaned:
                    keys.add(cleaned)

            for key in keys:
                if key not in alias_map:
                    alias_map[key] = set()
                alias_map[key].add(s_id)

        self._alias_to_series = alias_map
        self._cached_at = self._clock.now()

    async def _ensure_cache(self) -> None:
        if not self._is_cache_valid():
            await self._refresh_cache()

    def _compute_suggestions(
        self, normalized_scope: str, available_series: Sequence[SeriesId]
    ) -> list[str]:
        if not available_series:
            return []

        matched_series: list[SeriesId] = []

        # 1. Prefix matches on series_id, display_name, or aliases
        for s_id in available_series:
            manifest = self._cached_manifests.get(s_id)
            if not manifest:
                continue
            candidates = [
                str(manifest.series_id).lower(),
                manifest.display_name.lower(),
                *[a.lower() for a in manifest.aliases],
            ]
            if any(c.startswith(normalized_scope) for c in candidates):
                if s_id not in matched_series:
                    matched_series.append(s_id)

        # 2. Substring containment
        for s_id in available_series:
            manifest = self._cached_manifests.get(s_id)
            if not manifest or s_id in matched_series:
                continue
            candidates = [
                str(manifest.series_id).lower(),
                manifest.display_name.lower(),
                *[a.lower() for a in manifest.aliases],
            ]
            if any(normalized_scope in c or c in normalized_scope for c in candidates):
                matched_series.append(s_id)

        # 3. Fuzzy matching on alias map keys
        close_keys = difflib.get_close_matches(
            normalized_scope,
            list(self._alias_to_series.keys()),
            n=5,
            cutoff=0.4,
        )
        for key in close_keys:
            for s_id in self._alias_to_series.get(key, ()):
                if s_id not in matched_series:
                    matched_series.append(s_id)

        return [str(s) for s in matched_series[:3]]

    async def resolve(self, scope: str | None) -> ResolvedScope:
        """Resolve a user-provided series identifier or alias into a validated scope.

        Raises:
            InvalidScope: If the scope is omitted/empty, ambiguous, or not found.
        """
        if scope is None or not scope.strip():
            await self._ensure_cache()
            suggestions = [str(s) for s in list(self._cached_manifests.keys())[:3]]
            raise InvalidScope(
                scope="" if scope is None else scope,
                suggestions=suggestions,
                message="Series scope is required for all lore retrieval.",
            )

        cleaned_scope = scope.strip()
        normalized = cleaned_scope.lower()

        await self._ensure_cache()

        # Direct match in alias map
        if normalized in self._alias_to_series:
            candidates = self._alias_to_series[normalized]
            if len(candidates) == 1:
                series_id = next(iter(candidates))
                manifest = self._cached_manifests[series_id]
                return ResolvedScope(series_id=series_id, manifest=manifest)

            # Ambiguous alias mapping to multiple series
            suggestions = sorted(str(s) for s in candidates)[:3]
            formatted = ", ".join(repr(s) for s in suggestions)
            raise InvalidScope(
                scope=cleaned_scope,
                suggestions=suggestions,
                message=(
                    f"Ambiguous series scope '{cleaned_scope}'. "
                    f"Matches multiple series: {formatted}."
                ),
            )

        # Unknown scope -> compute suggestions
        available = list(self._cached_manifests.keys())
        suggestions = self._compute_suggestions(normalized, available)
        raise InvalidScope(scope=cleaned_scope, suggestions=suggestions)

    async def get_manifest(self, series_id: SeriesId) -> SeriesManifest | None:
        """Retrieve the cached manifest for a series or refresh if missing."""
        await self._ensure_cache()
        manifest = self._cached_manifests.get(series_id)
        if manifest is not None:
            return manifest

        # If not present in current cache, query registry directly
        fresh = await self._registry.get_series(series_id)
        if fresh is not None:
            self._cached_manifests[series_id] = fresh
        return fresh

    async def list_series(self) -> list[SeriesId]:
        """Return list of all registered series IDs."""
        await self._ensure_cache()
        return sorted(self._cached_manifests.keys())

    def invalidate_cache(self) -> None:
        """Invalidate the in-process cache, forcing a refresh on next access."""
        self._cached_at = None
