"""Unit tests for ScopeResolver service, alias resolution,
suggestions, and TTL caching.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from lore_mcp.application.errors import InvalidScope
from lore_mcp.application.scope import ResolvedScope, ScopeResolver
from lore_mcp.domain.entities import SeriesManifest
from lore_mcp.domain.types import SeriesId
from lore_mcp.testing.fakes import FakeClock, FakeSeriesRegistry


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock(start_time=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC))


@pytest.fixture
def sample_manifests() -> list[SeriesManifest]:
    return [
        SeriesManifest(
            series_id=SeriesId("loki"),
            display_name="Loki (TV Series)",
            kind="show",
            aliases=["MCU Loki", "God of Stories"],
            seasons=[1, 2],
            chunk_count=100,
        ),
        SeriesManifest(
            series_id=SeriesId("andor"),
            display_name="Andor",
            kind="show",
            aliases=["Cassian Andor", "Star Wars: Andor"],
            seasons=[1, 2],
            chunk_count=80,
        ),
        SeriesManifest(
            series_id=SeriesId("arcane"),
            display_name="Arcane",
            kind="anime",
            aliases=["League of Legends: Arcane"],
            seasons=[1, 2],
            chunk_count=120,
        ),
    ]


@pytest.fixture
def fake_registry(sample_manifests: list[SeriesManifest]) -> FakeSeriesRegistry:
    registry = FakeSeriesRegistry()
    for m in sample_manifests:
        registry._manifests[m.series_id] = m
    return registry


@pytest.fixture
def resolver(fake_registry: FakeSeriesRegistry, fake_clock: FakeClock) -> ScopeResolver:
    return ScopeResolver(registry=fake_registry, clock=fake_clock, ttl_seconds=60.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_exact_series_id(resolver: ScopeResolver) -> None:
    res = await resolver.resolve("loki")
    assert isinstance(res, ResolvedScope)
    assert res.series_id == SeriesId("loki")
    assert res.manifest.display_name == "Loki (TV Series)"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_case_insensitive(resolver: ScopeResolver) -> None:
    for variant in ["LOKI", "LoKi", "  loki  "]:
        res = await resolver.resolve(variant)
        assert res.series_id == SeriesId("loki")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_display_name(resolver: ScopeResolver) -> None:
    res = await resolver.resolve("Loki (TV Series)")
    assert res.series_id == SeriesId("loki")

    res_lower = await resolver.resolve("loki (tv series)")
    assert res_lower.series_id == SeriesId("loki")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_aliases(resolver: ScopeResolver) -> None:
    res1 = await resolver.resolve("MCU Loki")
    assert res1.series_id == SeriesId("loki")

    res2 = await resolver.resolve("god of stories")
    assert res2.series_id == SeriesId("loki")

    res3 = await resolver.resolve("Star Wars: Andor")
    assert res3.series_id == SeriesId("andor")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_empty_or_none_raises_invalid_scope(
    resolver: ScopeResolver,
) -> None:
    for invalid in [None, "", "   ", "\t\n"]:
        with pytest.raises(InvalidScope) as exc_info:
            await resolver.resolve(invalid)
        err = exc_info.value
        assert "Series scope is required" in str(err)
        assert len(err.suggestions) > 0
        assert "loki" in err.suggestions


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_unknown_with_close_suggestions(
    resolver: ScopeResolver,
) -> None:
    with pytest.raises(InvalidScope) as exc_info:
        await resolver.resolve("loky")
    err = exc_info.value
    assert err.scope == "loky"
    assert "loki" in err.suggestions
    assert len(err.suggestions) <= 3
    assert "Did you mean: 'loki'" in str(err)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_unknown_prefix_suggestions(
    resolver: ScopeResolver,
) -> None:
    with pytest.raises(InvalidScope) as exc_info:
        await resolver.resolve("arc")
    err = exc_info.value
    assert "arcane" in err.suggestions


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_unknown_no_suggestions(
    resolver: ScopeResolver,
) -> None:
    with pytest.raises(InvalidScope) as exc_info:
        await resolver.resolve("xyzzy-completely-unknown-series-999")
    err = exc_info.value
    assert err.suggestions == []
    assert "Unknown or invalid series scope" in str(err)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_ambiguous_alias(
    fake_clock: FakeClock,
) -> None:
    registry = FakeSeriesRegistry()
    m1 = SeriesManifest(
        series_id=SeriesId("mcu-prime"),
        display_name="MCU Prime",
        aliases=["Marvel", "Earth-616"],
    )
    m2 = SeriesManifest(
        series_id=SeriesId("mcu-variant"),
        display_name="MCU Variant",
        aliases=["Marvel", "Earth-838"],
    )
    registry._manifests[m1.series_id] = m1
    registry._manifests[m2.series_id] = m2

    resolver = ScopeResolver(registry=registry, clock=fake_clock, ttl_seconds=60.0)

    with pytest.raises(InvalidScope) as exc_info:
        await resolver.resolve("Marvel")
    err = exc_info.value
    assert err.scope == "Marvel"
    assert "Ambiguous series scope 'Marvel'" in str(err)
    assert set(err.suggestions) == {"mcu-prime", "mcu-variant"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_manifest_cache_in_process_ttl(
    fake_registry: FakeSeriesRegistry, fake_clock: FakeClock
) -> None:
    # Wrap get_manifests with spy
    original_get_manifests = fake_registry.get_manifests
    spy_get_manifests = AsyncMock(side_effect=original_get_manifests)
    fake_registry.get_manifests = spy_get_manifests  # type: ignore[method-assign]

    resolver = ScopeResolver(registry=fake_registry, clock=fake_clock, ttl_seconds=60.0)

    # 1. First resolution: queries registry
    res1 = await resolver.resolve("loki")
    assert res1.series_id == SeriesId("loki")
    assert spy_get_manifests.call_count == 1

    # 2. Subsequent resolutions within TTL (0-59s): uses in-process cache
    fake_clock.advance(timedelta(seconds=30))
    res2 = await resolver.resolve("andor")
    assert res2.series_id == SeriesId("andor")
    res3 = await resolver.resolve("arcane")
    assert res3.series_id == SeriesId("arcane")
    assert spy_get_manifests.call_count == 1

    # 3. Advance past TTL (61s from start): next resolution refreshes cache
    fake_clock.advance(timedelta(seconds=31))  # total 61s
    res4 = await resolver.resolve("loki")
    assert res4.series_id == SeriesId("loki")
    assert spy_get_manifests.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_invalidate_cache(
    fake_registry: FakeSeriesRegistry, fake_clock: FakeClock
) -> None:
    original_get_manifests = fake_registry.get_manifests
    spy = AsyncMock(side_effect=original_get_manifests)
    fake_registry.get_manifests = spy  # type: ignore[method-assign]

    resolver = ScopeResolver(registry=fake_registry, clock=fake_clock, ttl_seconds=60.0)

    await resolver.resolve("loki")
    assert spy.call_count == 1

    # Manual invalidation forces refresh even at 0 elapsed time
    resolver.invalidate_cache()
    await resolver.resolve("andor")
    assert spy.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_manifest_and_list_series(
    resolver: ScopeResolver,
) -> None:
    series_list = await resolver.list_series()
    assert series_list == [SeriesId("andor"), SeriesId("arcane"), SeriesId("loki")]

    manifest = await resolver.get_manifest(SeriesId("loki"))
    assert manifest is not None
    assert manifest.series_id == SeriesId("loki")

    missing = await resolver.get_manifest(SeriesId("nonexistent"))
    assert missing is None
