"""Testing utilities, in-memory fakes, and contract test suites."""

from lore_mcp.testing.contracts import (
    BudgetCounterContractSuite,
    CacheContractSuite,
    ClockContractSuite,
    DocumentSourceContractSuite,
    DocumentStoreContractSuite,
    EntityRegistryContractSuite,
    EventRegistryContractSuite,
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
    FakeEventRegistry,
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
    "EventRegistryContractSuite",
    "FakeBudgetCounter",
    "FakeCache",
    "FakeClock",
    "FakeDocumentSource",
    "FakeDocumentStore",
    "FakeEntityRegistry",
    "FakeEventRegistry",
    "FakeSeriesRegistry",
    "FakeVectorIndex",
    "FakeWebSearch",
    "SeriesRegistryContractSuite",
    "VectorIndexContractSuite",
    "WebSearchContractSuite",
]
