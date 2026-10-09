"""Application layer: orchestrations, use cases, and ingestion stages."""

from lore_mcp.application.dense import DenseRetriever
from lore_mcp.application.errors import (
    ApplicationError,
    DataUnavailable,
    DependencyUnavailable,
    IncompatibleData,
    InvalidArgument,
    InvalidScope,
    RateLimited,
)
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.application.scope import ResolvedScope, ScopeResolver
from lore_mcp.application.series_loader import (
    load_all_series_configs,
    load_series_config,
)
from lore_mcp.application.sparse import SparseRetriever

__all__ = [
    "ApplicationError",
    "DataUnavailable",
    "DenseRetriever",
    "DependencyUnavailable",
    "HybridRetriever",
    "IncompatibleData",
    "InvalidArgument",
    "InvalidScope",
    "RateLimited",
    "ResolvedScope",
    "ScopeResolver",
    "SparseRetriever",
    "load_all_series_configs",
    "load_series_config",
]
