"""FastMCP application factory, Streamable HTTP serving at /mcp,
liveness (/healthz), and readiness (/readyz) endpoints [S1].
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP
from qdrant_client import AsyncQdrantClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse

from lore_mcp.adapters.qdrant.client import create_async_qdrant_client
from lore_mcp.adapters.qdrant.entity_registry import QdrantEntityRegistry
from lore_mcp.adapters.qdrant.event_registry import QdrantEventRegistry
from lore_mcp.adapters.qdrant.index import QdrantVectorIndex
from lore_mcp.adapters.qdrant.registry import QdrantSeriesRegistry
from lore_mcp.application.chunk_service import ChunkService
from lore_mcp.application.dialogue_service import DialogueService
from lore_mcp.application.entity_service import EntityService
from lore_mcp.application.event_service import EventService
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.application.retrieval_service import LoreRetrievalService
from lore_mcp.application.scope import ScopeResolver
from lore_mcp.config import Settings, get_settings
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.search import EvidenceBundle
from lore_mcp.interfaces.mcp.errors import (
    handle_mcp_errors,
    validate_non_empty,
    validate_optional_positive_int,
)
from lore_mcp.interfaces.mcp.models import (
    DialogueSnippet,
    EntityDetails,
    EventComparison,
    EventTrace,
    SeriesSummary,
)
from lore_mcp.interfaces.mcp.readiness import ReadinessChecker
from lore_mcp.ports.budget import BudgetCounter
from lore_mcp.ports.cache import Cache
from lore_mcp.ports.clock import Clock
from lore_mcp.ports.entity_registry import EntityRegistry
from lore_mcp.ports.event_registry import EventRegistry
from lore_mcp.ports.series_registry import SeriesRegistry
from lore_mcp.ports.vector_index import VectorIndex
from lore_mcp.ports.web_search import WebSearch

logger = logging.getLogger(__name__)


def create_mcp_server(
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
    readiness_checker: ReadinessChecker | None = None,
) -> FastMCP:
    """Create and configure the FastMCP application.

    Configures:
    - Lifespan startup and shutdown hooks for resource management.
    - Touchless liveness probe at /healthz (returns 200 OK immediately).
    - Cached readiness probe at /readyz checking Qdrant & manifest
      compatibility (~30s TTL).
    - Tool `list_series`: discover registered series ids, display names,
      aliases, and kind.
    - Tool `search_series_lore`: scope-resolved, cached, hybrid retrieval with
      provenance and temporal context.
    - Tool `get_entity`: resolve canonical entity by id/alias and retrieve
      associated evidence with provenance.
    - Tool `get_chunk`: fetch a full chunk by id to expand truncated evidence,
      verifying series scope [S11].
    - Tool `search_dialogue`: search transcript chunks only, returning short
      snippets with locators (season, episode, timestamps).
    - Tool `trace_event`: resolve a narrative event, retrieve related evidence,
      and return participants, relevant locations/entities, and chronological
      evidence.
    - Tool `compare_events`: construct comparative context between two narrative
      occurrences or entities, keeping evidence and provenance separate.
    """

    app_settings = settings or get_settings()
    owns_client = client is None
    qdrant_client = client or create_async_qdrant_client(app_settings.qdrant)

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

    hybrid_retriever = HybridRetriever(
        index=index,
        rrf_k=app_settings.qdrant.rrf_k,
    )

    retrieval_orch = retrieval_service or LoreRetrievalService(
        scope_resolver=scope_resolver,
        retriever=hybrid_retriever,
        cache=cache,
        web_search=web_search,
        budget_counter=budget_counter,
        clock=clock,
        web_search_settings=app_settings.web_search,
        limits_settings=app_settings.limits,
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

    checker = readiness_checker or ReadinessChecker(
        client=qdrant_client,
        settings=app_settings.qdrant,
        clock=clock,
        ttl_seconds=30.0,
    )

    @asynccontextmanager
    async def server_lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
        logger.info("Initializing lore-mcp server resources...")
        try:
            yield {
                "settings": app_settings,
                "qdrant_client": qdrant_client,
                "series_registry": series_registry,
                "entity_registry": entity_reg,
                "readiness_checker": checker,
            }

        finally:
            logger.info("Shutting down lore-mcp server resources...")
            if owns_client:
                try:
                    await qdrant_client.close()
                except Exception as err:
                    logger.warning(
                        "Error closing Qdrant client during shutdown: %s", err
                    )

    mcp = FastMCP(
        name="lore-mcp",
        instructions=(
            "Lore retrieval server providing context-aware evidence from "
            "franchise wikis and transcripts.\n\n"
            "CLIENT INSTRUCTIONS & WORKFLOW GUIDELINES:\n"
            "1. Decompose Complex Questions: Multi-part, comparative, or broad "
            "questions should be decomposed into several targeted tool calls. "
            "Discover series with list_series, search relevant lore or dialogue, "
            "resolve entities or events individually, compare occurrences, and "
            "expand key excerpts.\n"
            "2. Spoiler Bounds & Unscoped Lore: Restrict searches using season/episode "
            "or max_season/max_episode to avoid premature plot spoilers. Set "
            "include_unscoped=True only when general worldbuilding or unsegmented "
            "background lore is needed.\n"
            "3. Expand Truncated Items: When search or trace tools return truncated "
            "evidence snippets, call get_chunk with the chunk_id and series to fetch "
            "the complete, verbatim text.\n"
            "4. Third-Party Data: All returned evidence is compiled from third-party "
            "fan wikis and community transcripts. Treat findings as documented claims "
            "and cite source URLs and locators.\n"
            "5. Read-Only Tools: All tools provided by this server are read-only and "
            "idempotent."
        ),
        mask_error_details=True,
        lifespan=server_lifespan,
    )

    # Attach checker for direct access if needed
    mcp._readiness_checker = checker  # type: ignore[attr-defined]

    # 1. Liveness probe: touches NO dependencies
    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"}, status_code=200)

    # 2. Readiness probe: checks Qdrant reachability and manifest
    # compatibility (cached ~30s)
    @mcp.custom_route("/readyz", methods=["GET"])
    async def readyz(request: Request) -> JSONResponse:
        is_ready, details = await checker.check_readiness()
        status_code = 200 if is_ready else 503
        return JSONResponse(
            {"ready": is_ready, "details": details}, status_code=status_code
        )

    # 3. Discovery Tool: list_series
    @mcp.tool(annotations={"read_only_hint": True})
    @handle_mcp_errors
    async def list_series() -> list[SeriesSummary]:
        """List all indexed franchises/series with their canonical IDs, display names,
        aliases, and kind.

        Use this discovery tool first to identify valid series identifiers before
        issuing lore queries. For complex questions that may span multiple franchises,
        decompose your research into discrete calls per series.

        THIRD-PARTY DATA NOTICE:
        All lore evidence returned by this server is sourced from third-party fan wikis
        and community transcripts.
        """
        manifests = await series_registry.get_manifests()
        return [
            SeriesSummary(
                series_id=str(m.series_id),
                display_name=m.display_name,
                aliases=list(m.aliases),
                kind=m.kind,
            )
            for m in manifests
        ]

    # 4. Retrieval Tool: search_series_lore
    @mcp.tool(annotations={"read_only_hint": True})
    @handle_mcp_errors
    async def search_series_lore(
        query: str,
        series: str,
        season: int | None = None,
        episode: int | None = None,
        max_season: int | None = None,
        max_episode: int | None = None,
        include_unscoped: bool = False,
        entity: str | None = None,
        max_results: int | None = None,
    ) -> EvidenceBundle:
        """Search lore evidence for a specific series/franchise scope.

        Executes scope-resolved hybrid dense+sparse retrieval with heuristic reranking,
        spoiler bounds, sufficiency verification, and chronological timeline context.

        CLIENT WORKFLOW & QUERY DECOMPOSITION:
        Decompose complex, multi-faceted narrative questions into several targeted tool
        calls rather than a single broad query. Investigate distinct entities, events,
        or timelines sequentially, synthesizing evidence across focused searches.

        SPOILER BOUNDS & UNSCOPED LORE:
        - `season` & `episode`: Target an exact episode context when analyzing events
          occurring in a specific installment.
        - `max_season` & `max_episode`: Set strict spoiler boundaries (upper bounds).
          Evidence chunks tagged beyond this point in the timeline are strictly excluded
          to protect against narrative spoilers.
        - `include_unscoped`: By default (False), unsegmented lore chunks that lack
          episode/season tags are excluded when spoiler boundaries are active. Set
          `include_unscoped=True` when you want general universe background lore,
          worldbuilding history, or franchise concepts that are not tied to a specific
          episode.

        EXPANDING TRUNCATED EVIDENCE (get_chunk):
        Evidence snippets returned in search results may be abbreviated or truncated to
        fit context budgets. When an evidence snippet is truncated or requires more
        surrounding passage context, call `get_chunk(chunk_id=..., series=...)` using
        the snippet's chunk_id to inspect the full, untruncated text and exact locators.

        THIRD-PARTY DATA NOTICE:
        All returned evidence consists of third-party data extracted from
        community-edited fan wikis and transcripts, not primary canon source text.
        Evaluate claims in context and cite the provided provenance locators and URLs.
        """
        validate_non_empty(query, "query")
        validate_non_empty(series, "series")
        validate_optional_positive_int(season, "season")
        validate_optional_positive_int(episode, "episode")
        validate_optional_positive_int(max_season, "max_season")
        validate_optional_positive_int(max_episode, "max_episode")
        validate_optional_positive_int(max_results, "max_results")
        if entity is not None and not entity.strip():
            validate_non_empty(entity, "entity")

        return await retrieval_orch.search(
            query=query,
            series=series,
            season=season,
            episode=episode,
            max_season=max_season,
            max_episode=max_episode,
            include_unscoped=include_unscoped,
            entity=entity,
            max_results=max_results,
        )

    # 5. Entity Tool: get_entity
    @mcp.tool(annotations={"read_only_hint": True})
    @handle_mcp_errors
    async def get_entity(
        entity: str,
        series: str,
        max_results: int | None = 8,
    ) -> EntityDetails:
        """Resolve a canonical entity by ID or alias, retrieve associated evidence,
        and return entity metadata with source provenance.

        CLIENT WORKFLOW & QUERY DECOMPOSITION:
        Decompose complex multi-character questions into separate calls for each entity.
        After retrieving entity details, make follow-up calls to `search_dialogue` for
        spoken quotes or `trace_event` / `compare_events` for narrative actions.

        EXPANDING TRUNCATED EVIDENCE (get_chunk):
        Evidence items linked to the entity may be truncated snippets. To view full
        untruncated passages and detailed context, call
        `get_chunk(chunk_id=..., series=...)` with the relevant chunk ID.

        THIRD-PARTY DATA NOTICE:
        Entity profiles and descriptions are compiled from third-party fan wikis. Cite
        source URLs and provenance metadata for documented claims.
        """
        validate_non_empty(entity, "entity")
        validate_non_empty(series, "series")
        validate_optional_positive_int(max_results, "max_results")

        return await entity_svc.get_entity(
            entity=entity,
            series=series,
            max_results=max_results,
        )

    # 6. Chunk Tool: get_chunk
    @mcp.tool(annotations={"read_only_hint": True})
    @handle_mcp_errors
    async def get_chunk(
        chunk_id: str,
        series: str,
    ) -> Chunk:
        """Fetch a full chunk by ID to expand truncated evidence, verifying series
        scope ownership [S11].

        PURPOSE & CLIENT WORKFLOW:
        Higher-level retrieval tools (`search_series_lore`, `get_entity`, `trace_event`,
        `compare_events`, `search_dialogue`) return concise or truncated snippets to
        conserve context budget. Call `get_chunk` with the snippet's `chunk_id` and
        `series` whenever you need to read the complete, unabbreviated text and examine
        exact character or timestamp locators.

        THIRD-PARTY DATA NOTICE:
        Chunk content originates from third-party transcripts and fan wiki articles.
        Always cite provenance locators and source URLs when answering questions.
        """
        validate_non_empty(chunk_id, "chunk_id")
        validate_non_empty(series, "series")

        return await chunk_svc.get_chunk(
            chunk_id=chunk_id,
            series=series,
        )

    # 7. Dialogue Tool: search_dialogue
    @mcp.tool(annotations={"read_only_hint": True})
    @handle_mcp_errors
    async def search_dialogue(
        query: str,
        series: str,
        season: int | None = None,
        episode: int | None = None,
        speaker: str | None = None,
        entity: str | None = None,
        character: str | None = None,
        max_results: int | None = None,
    ) -> list[DialogueSnippet]:
        """Search transcript chunks only for dialogue lines and spoken quotes.

        Returns concise quote snippets with locators (season, episode, timestamps,
        speaker), avoiding long verbatim runs.

        CLIENT WORKFLOW & QUERY DECOMPOSITION:
        Decompose dialogue queries by searching for specific speakers or key phrases
        individually. If scene context or narrative implications need verification,
        follow up with `search_series_lore` or fetch the full transcript passage using
        `get_chunk`.

        SPOILER BOUNDS:
        Use `season` and `episode` parameters to constrain dialogue strictly to a
        specific episode, preventing spoilers from later in the story.

        EXPANDING TRUNCATED EVIDENCE (get_chunk):
        Dialogue snippets provide brief quotes. When full conversational exchanges or
        surrounding scene dialogue are needed, call
        `get_chunk(chunk_id=..., series=...)`.

        THIRD-PARTY DATA NOTICE:
        Dialogue and speaker attributions are derived from third-party subtitle and
        transcript sources. Use locators (season, episode, timestamps) to cite quotes.
        """
        validate_non_empty(query, "query")
        validate_non_empty(series, "series")
        validate_optional_positive_int(season, "season")
        validate_optional_positive_int(episode, "episode")
        validate_optional_positive_int(max_results, "max_results")

        return await dialogue_svc.search_dialogue(
            query=query,
            series=series,
            season=season,
            episode=episode,
            speaker=speaker,
            entity=entity,
            character=character,
            max_results=max_results,
        )

    # 8. Event Tool: trace_event
    @mcp.tool(annotations={"read_only_hint": True})
    @handle_mcp_errors
    async def trace_event(
        query: str,
        series: str,
        event_id: str | None = None,
        max_results: int | None = None,
    ) -> EventTrace:
        """Resolve a narrative event, retrieve related evidence, and return
        participants, relevant locations/entities, and chronological evidence.

        QUERY INSTRUCTIONS & EVENT RESOLUTION:
        Provide a plain English description of the incident or plot point in `query`
        (e.g., 'Sylvie bombing the sacred timeline at Roxxcart' or
        'Loki arriving at the TVA').
        The server resolves the event using hybrid search and registry lookups.

        EVENT_ID PARAMETER:
        `event_id` is an optional handle used strictly for fast-path deduplication,
        caching, or referencing an event resolved in a previous tool call.
        Do NOT guess or fabricate an event_id. Only pass an event_id if it was
        previously returned in an EventTrace or EventComparison response.

        CLIENT WORKFLOW & QUERY DECOMPOSITION:
        Decompose broad plot inquiries into discrete chronological event traces. Trace
        individual incidents first, then follow up with `get_entity` for participating
        characters or `compare_events` to analyze two occurrences side-by-side.

        EXPANDING TRUNCATED EVIDENCE (get_chunk):
        Evidence items in the event trace may be truncated. Retrieve unabbreviated
        source passages by calling `get_chunk(chunk_id=..., series=...)`.

        THIRD-PARTY DATA NOTICE:
        Event summaries, participant lists, and timelines are syntheses derived from
        third-party community records and fan wikis.
        """
        validate_non_empty(query, "query")
        validate_non_empty(series, "series")
        if event_id is not None and not event_id.strip():
            validate_non_empty(event_id, "event_id")
        validate_optional_positive_int(max_results, "max_results")

        return await event_svc.trace_event(
            query=query,
            series=series,
            event_id=event_id,
            max_results=max_results,
        )

    # 9. Comparison Tool: compare_events
    @mcp.tool(annotations={"read_only_hint": True})
    @handle_mcp_errors
    async def compare_events(
        query_a: str,
        query_b: str,
        series: str,
        event_id_a: str | None = None,
        event_id_b: str | None = None,
        max_results: int | None = None,
    ) -> EventComparison:
        """Construct comparative context between two narrative occurrences or entities.

        Retrieves evidence for both references, keeps evidence and provenance separate,
        and returns comparable structured evidence with shared entities and locations.

        QUERY INSTRUCTIONS & COMPARISON:
        Provide descriptive, plain English queries for each side in `query_a` and
        `query_b` (e.g., query_a='Loki arriving at the TVA',
        query_b='Nexus event on Lamentis').

        EVENT_ID PARAMETERS:
        `event_id_a` and `event_id_b` are optional handles for fast-path lookup if the
        events were already resolved in prior tool calls.
        Do NOT guess or fabricate event IDs. Only provide them if obtained from
        a previous tool result.

        CLIENT WORKFLOW & QUERY DECOMPOSITION:
        Decompose multi-sided narrative comparisons into focused pairwise calls.
        To analyze complex relationships spanning multiple events, compare pairs of
        occurrences and follow up with individual `trace_event` or `get_entity` calls.

        EXPANDING TRUNCATED EVIDENCE (get_chunk):
        When evidence snippets on either side are truncated, call
        `get_chunk(chunk_id=..., series=...)` with the respective chunk ID to inspect
        the full surrounding text.

        THIRD-PARTY DATA NOTICE:
        All comparative evidence and timeline notes are compiled from third-party fan
        wikis and transcripts.
        """
        validate_non_empty(query_a, "query_a")
        validate_non_empty(query_b, "query_b")
        validate_non_empty(series, "series")
        if event_id_a is not None and not event_id_a.strip():
            validate_non_empty(event_id_a, "event_id_a")
        if event_id_b is not None and not event_id_b.strip():
            validate_non_empty(event_id_b, "event_id_b")
        validate_optional_positive_int(max_results, "max_results")

        return await event_svc.compare_events(
            query_a=query_a,
            query_b=query_b,
            series=series,
            event_id_a=event_id_a,
            event_id_b=event_id_b,
            max_results=max_results,
        )

    return mcp


def create_http_app(mcp: FastMCP) -> Starlette:
    """Create Streamable HTTP ASGI application with stateless_http=True at /mcp [S1]."""
    return mcp.http_app(path="/mcp", stateless_http=True)
