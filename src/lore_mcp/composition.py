"""Application composition root assembling domain, ports, adapters, and services."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient

from lore_mcp.adapters.qdrant.client import create_async_qdrant_client
from lore_mcp.adapters.qdrant.entity_registry import QdrantEntityRegistry
from lore_mcp.adapters.qdrant.event_registry import QdrantEventRegistry
from lore_mcp.adapters.qdrant.index import QdrantVectorIndex
from lore_mcp.adapters.qdrant.registry import QdrantSeriesRegistry
from lore_mcp.adapters.redis import (
    CachedRetrievalService,
    CachedWebSearch,
    CircuitBreaker,
    CommandTracker,
    GlobalRateGuard,
    RedisBudgetCounter,
    RedisCache,
    create_redis_client,
)
from lore_mcp.application.chunk_service import ChunkService
from lore_mcp.application.dialogue_service import DialogueService
from lore_mcp.application.entity_service import EntityService
from lore_mcp.application.event_service import EventService
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.application.retrieval_service import LoreRetrievalService
from lore_mcp.application.scope import ScopeResolver
from lore_mcp.config import Settings, get_settings
from lore_mcp.domain.evaluation import EvaluationConfig
from lore_mcp.ports.budget import BudgetCounter
from lore_mcp.ports.cache import Cache
from lore_mcp.ports.clock import Clock
from lore_mcp.ports.entity_registry import EntityRegistry
from lore_mcp.ports.event_registry import EventRegistry
from lore_mcp.ports.series_registry import SeriesRegistry
from lore_mcp.ports.vector_index import VectorIndex
from lore_mcp.ports.web_search import WebSearch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApplicationServices:
    """Composition container for all initialized domain and application services."""

    settings: Settings
    qdrant_client: AsyncQdrantClient
    series_registry: SeriesRegistry
    entity_registry: EntityRegistry
    event_registry: EventRegistry
    vector_index: VectorIndex
    scope_resolver: ScopeResolver
    hybrid_retriever: HybridRetriever
    retrieval_service: LoreRetrievalService
    entity_service: EntityService
    event_service: EventService
    chunk_service: ChunkService
    dialogue_service: DialogueService
    owns_client: bool
    cache: Cache | None = None
    budget_counter: BudgetCounter | None = None
    rate_guard: GlobalRateGuard | None = None


def build_application_services(
    settings: Settings | None = None,
    client: AsyncQdrantClient | None = None,
    registry: SeriesRegistry | None = None,
    entity_registry: EntityRegistry | None = None,
    event_registry: EventRegistry | None = None,
    vector_index: VectorIndex | None = None,
    retrieval_service: LoreRetrievalService | None = None,
    entity_service: EntityService | None = None,
    event_service: EventService | None = None,
    chunk_service: ChunkService | None = None,
    dialogue_service: DialogueService | None = None,
    cache: Cache | None = None,
    web_search: WebSearch | None = None,
    budget_counter: BudgetCounter | None = None,
    clock: Clock | None = None,
    overrides: EvaluationConfig | None = None,
) -> ApplicationServices:
    """Build and wire all application services from configuration and adapters."""
    app_settings = settings or get_settings()
    owns_client = client is None
    qdrant_client = client or create_async_qdrant_client(app_settings.qdrant)

    active_cache = cache
    active_budget_counter = budget_counter
    active_web_search = web_search

    # Wire Redis adapter when URL is configured and ports not explicitly provided
    if app_settings.redis.url and active_cache is None:
        try:
            redis_client = create_redis_client(app_settings.redis)
            if redis_client is not None:
                circuit_breaker = CircuitBreaker(
                    failure_threshold=app_settings.redis.circuit_breaker_failure_threshold,
                    reset_timeout=app_settings.redis.circuit_breaker_reset_seconds,
                )
                command_tracker = CommandTracker(
                    monthly_budget=app_settings.redis.monthly_command_budget,
                    shedding_ratio=app_settings.redis.command_budget_shedding_ratio,
                )
                active_cache = RedisCache(
                    client=redis_client,
                    circuit_breaker=circuit_breaker,
                    command_tracker=command_tracker,
                    settings=app_settings.redis,
                )
                if active_budget_counter is None:
                    active_budget_counter = RedisBudgetCounter(
                        client=redis_client,
                        circuit_breaker=circuit_breaker,
                        command_tracker=command_tracker,
                        settings=app_settings.redis,
                    )
        except Exception as err:
            logger.warning(
                "Failed to initialize Redis cache adapter: %s. "
                "Operating with cache disabled.",
                err,
            )

    series_registry = registry or QdrantSeriesRegistry(
        client=qdrant_client,
        settings=app_settings.qdrant,
    )

    entity_reg = entity_registry or QdrantEntityRegistry(
        client=qdrant_client,
        settings=app_settings.qdrant,
    )

    event_reg = event_registry or QdrantEventRegistry(
        client=qdrant_client,
        settings=app_settings.qdrant,
    )

    index = vector_index or QdrantVectorIndex(
        client=qdrant_client,
        settings=app_settings.qdrant,
    )

    scope_resolver = ScopeResolver(
        registry=series_registry,
        clock=clock,
    )

    rrf_k = overrides.rrf_k if overrides else app_settings.qdrant.rrf_k
    hybrid_retriever = HybridRetriever(
        index=index,
        rrf_k=rrf_k,
    )

    # Decorate web search with CachedWebSearch if web_search and cache are available
    if active_web_search is not None and active_cache is not None:
        active_web_search = CachedWebSearch(
            inner=active_web_search,
            cache=active_cache,
            settings=app_settings.redis,
        )

    base_retrieval_orch = retrieval_service or LoreRetrievalService(
        scope_resolver=scope_resolver,
        retriever=hybrid_retriever,
        cache=None,
        web_search=active_web_search,
        budget_counter=active_budget_counter,
        clock=clock,
        web_search_settings=app_settings.web_search,
        limits_settings=app_settings.limits,
    )

    # Decorate retrieval_orch with CachedRetrievalService if cache is available
    retrieval_orch: LoreRetrievalService = base_retrieval_orch
    if active_cache is not None and retrieval_service is None:
        retrieval_orch = CachedRetrievalService(
            inner=base_retrieval_orch,
            cache=active_cache,
            settings=app_settings.redis,
        )

    rate_guard: GlobalRateGuard | None = None
    if active_budget_counter is not None:
        rate_guard = GlobalRateGuard(
            budget_counter=active_budget_counter,
            rate_limit_per_minute=app_settings.redis.global_rate_limit_per_minute,
            clock=clock,
        )

    entity_svc = entity_service or EntityService(
        scope_resolver=scope_resolver,
        retriever=hybrid_retriever,
        entity_registry=entity_reg,
        limits_settings=app_settings.limits,
    )

    event_svc = event_service or EventService(
        scope_resolver=scope_resolver,
        retriever=hybrid_retriever,
        event_registry=event_reg,
        entity_registry=entity_reg,
        limits_settings=app_settings.limits,
    )

    chunk_svc = chunk_service or ChunkService(
        scope_resolver=scope_resolver,
        vector_index=index,
    )

    dialogue_svc = dialogue_service or DialogueService(
        scope_resolver=scope_resolver,
        retriever=hybrid_retriever,
        limits_settings=app_settings.limits,
    )

    return ApplicationServices(
        settings=app_settings,
        qdrant_client=qdrant_client,
        series_registry=series_registry,
        entity_registry=entity_reg,
        event_registry=event_reg,
        vector_index=index,
        scope_resolver=scope_resolver,
        hybrid_retriever=hybrid_retriever,
        retrieval_service=retrieval_orch,
        entity_service=entity_svc,
        event_service=event_svc,
        chunk_service=chunk_svc,
        dialogue_service=dialogue_svc,
        owns_client=owns_client,
        cache=active_cache,
        budget_counter=active_budget_counter,
        rate_guard=rate_guard,
    )
