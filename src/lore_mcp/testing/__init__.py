"""Testing utilities, in-memory fakes, and contract test suites."""

from lore_mcp.testing.contracts import (
    BudgetCounterContractSuite,
    CacheContractSuite,
    ClockContractSuite,
    DocumentSourceContractSuite,
    DocumentStoreContractSuite,
    EntityRegistryContractSuite,
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
    FakeEntityRegistry,
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
    "EntityRegistryContractSuite",
    "FakeBudgetCounter",
    "FakeCache",
    "FakeClock",
    "FakeDocumentSource",
    "FakeDocumentStore",
    "FakeEntityRegistry",
    "FakeSeriesRegistry",
    "FakeVectorIndex",
    "FakeWebSearch",
    "SeriesRegistryContractSuite",
    "VectorIndexContractSuite",
    "WebSearchContractSuite",
]
