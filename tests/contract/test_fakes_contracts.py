"""Contract tests verifying that in-memory fakes satisfy all port contracts."""

from datetime import UTC, datetime

import pytest

from lore_mcp.domain.documents import RawDocument
from lore_mcp.domain.search import WebSearchResult
from lore_mcp.domain.types import SeriesId
from lore_mcp.testing import (
    BudgetCounterContractSuite,
    CacheContractSuite,
    ClockContractSuite,
    DocumentSourceContractSuite,
    DocumentStoreContractSuite,
    FakeBudgetCounter,
    FakeCache,
    FakeClock,
    FakeDocumentSource,
    FakeDocumentStore,
    FakeSeriesRegistry,
    FakeVectorIndex,
    FakeWebSearch,
    SeriesRegistryContractSuite,
    VectorIndexContractSuite,
    WebSearchContractSuite,
)


@pytest.mark.contract
class TestFakeVectorIndexContract(VectorIndexContractSuite):
    @pytest.fixture
    def index(self) -> FakeVectorIndex:
        return FakeVectorIndex()


@pytest.mark.contract
class TestFakeSeriesRegistryContract(SeriesRegistryContractSuite):
    @pytest.fixture
    def registry(self) -> FakeSeriesRegistry:
        return FakeSeriesRegistry()


@pytest.mark.contract
class TestFakeCacheContract(CacheContractSuite):
    @pytest.fixture
    def cache(self) -> FakeCache:
        return FakeCache()


@pytest.mark.contract
class TestFakeBudgetCounterContract(BudgetCounterContractSuite):
    @pytest.fixture
    def counter(self) -> FakeBudgetCounter:
        return FakeBudgetCounter()


@pytest.mark.contract
class TestFakeWebSearchContract(WebSearchContractSuite):
    @pytest.fixture
    def search(self) -> FakeWebSearch:
        fake = FakeWebSearch()
        fake.set_default_results(
            [
                WebSearchResult(
                    url="https://example.com/1",
                    title="Result 1",
                    snippet="Snippet 1",
                ),
                WebSearchResult(
                    url="https://example.com/2",
                    title="Result 2",
                    snippet="Snippet 2",
                ),
            ]
        )
        return fake


@pytest.mark.contract
class TestFakeDocumentSourceContract(DocumentSourceContractSuite):
    @pytest.fixture
    def source(self) -> FakeDocumentSource:
        fake = FakeDocumentSource()
        fake.add_document(
            RawDocument(
                series_id=SeriesId("loki"),
                source_url="https://marvel.fandom.com/wiki/Loki",
                source_type="fandom",
                content="Loki source content",
                retrieval_timestamp=datetime.now(UTC),
            )
        )
        return fake


@pytest.mark.contract
class TestFakeDocumentStoreContract(DocumentStoreContractSuite):
    @pytest.fixture
    def store(self) -> FakeDocumentStore:
        return FakeDocumentStore()


@pytest.mark.contract
class TestFakeClockContract(ClockContractSuite):
    @pytest.fixture
    def clock(self) -> FakeClock:
        return FakeClock()
