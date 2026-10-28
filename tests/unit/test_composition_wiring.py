"""Unit tests for composition root wiring of TavilyWebSearch and CachedWebSearch."""

from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from lore_mcp.adapters.redis import CachedWebSearch
from lore_mcp.adapters.tavily import TavilyWebSearch
from lore_mcp.composition import build_application_services
from lore_mcp.config import RedisSettings, Settings, WebSearchSettings
from lore_mcp.testing.fakes import FakeCache, FakeVectorIndex


@pytest.mark.unit
def test_composition_wires_tavily_when_api_key_provided() -> None:
    settings = Settings(
        redis=RedisSettings(url=None),
        web_search=WebSearchSettings(api_key=SecretStr("tvly-test-12345")),
    )
    mock_client = MagicMock()

    services = build_application_services(
        settings=settings,
        client=mock_client,
        vector_index=FakeVectorIndex(),
    )

    retrieval = services.retrieval_service
    # Underlying search adapter should be TavilyWebSearch
    assert isinstance(retrieval._web_search, TavilyWebSearch)
    assert retrieval._web_search._api_key == "tvly-test-12345"


@pytest.mark.unit
def test_composition_decorates_web_search_with_cache() -> None:
    settings = Settings(
        web_search=WebSearchSettings(api_key=SecretStr("tvly-test-12345"))
    )
    mock_client = MagicMock()
    fake_cache = FakeCache()

    services = build_application_services(
        settings=settings,
        client=mock_client,
        vector_index=FakeVectorIndex(),
        cache=fake_cache,
    )

    retrieval = services.retrieval_service
    # Should be decorated with CachedWebSearch wrapping TavilyWebSearch
    assert isinstance(retrieval._web_search, CachedWebSearch)
    assert isinstance(retrieval._web_search._inner, TavilyWebSearch)
