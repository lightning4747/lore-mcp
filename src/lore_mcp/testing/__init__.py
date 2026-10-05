"""Testing utilities, in-memory fakes, and contract test suites."""

from lore_mcp.testing.contracts import (
    BudgetCounterContractSuite,
    CacheContractSuite,
    ClockContractSuite,
    DocumentSourceContractSuite,
    DocumentStoreContractSuite,
    SeriesRegistryContractSuite,
    VectorIndexContractSuite,
    WebSearchContractSuite,
)
from lore_mcp.testing.fakes import (
    FakeBudgetCounter,
    FakeCache,
    FakeClock,
    FakeDocumentSource,
    FakeDocumentStore,
    FakeSeriesRegistry,
    FakeVectorIndex,
    FakeWebSearch,
)

__all__ = [
    "BudgetCounterContractSuite",
    "CacheContractSuite",
    "ClockContractSuite",
    "DocumentSourceContractSuite",
    "DocumentStoreContractSuite",
    "FakeBudgetCounter",
    "FakeCache",
    "FakeClock",
    "FakeDocumentSource",
    "FakeDocumentStore",
    "FakeSeriesRegistry",
    "FakeVectorIndex",
    "FakeWebSearch",
    "SeriesRegistryContractSuite",
    "VectorIndexContractSuite",
    "WebSearchContractSuite",
]
