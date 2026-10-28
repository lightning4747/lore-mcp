"""Application orchestrator for series lore retrieval."""

import hashlib
import json
import logging
import urllib.parse
from typing import Any

from lore_mcp.application.chronology import DefaultChronologyBuilder
from lore_mcp.application.classifier import QueryClassifier
from lore_mcp.application.errors import InvalidArgument
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.application.scope import ScopeResolver
from lore_mcp.config import RequestLimitsSettings, WebSearchSettings
from lore_mcp.domain.entities import SeriesManifest
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import (
    Evidence,
    EvidenceBundle,
    SearchHit,
    SearchQuery,
)
from lore_mcp.domain.types import ChunkId, ValidatedSeriesId
from lore_mcp.ports.budget import BudgetCounter
from lore_mcp.ports.cache import Cache
from lore_mcp.ports.chronology import ChronologyBuilder
from lore_mcp.ports.clock import Clock, SystemClock
from lore_mcp.ports.web_search import WebSearch

logger = logging.getLogger(__name__)


def compute_cache_key(
    series_id: str,
    query: str,
    season: int | None = None,
    episode: int | None = None,
    max_season: int | None = None,
    max_episode: int | None = None,
    include_unscoped: bool = False,
    entity: str | None = None,
    max_results: int = 8,
) -> str:
    """Compute a deterministic hash cache key for retrieval query parameters."""
    payload = {
        "series_id": series_id.lower().strip(),
        "query": query.lower().strip(),
        "season": season,
        "episode": episode,
        "max_season": max_season,
        "max_episode": max_episode,
        "include_unscoped": include_unscoped,
        "entity": entity.lower().strip() if entity else None,
        "max_results": max_results,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"lore_retrieval:{series_id}:{digest}"


def canonicalize_url(url: str) -> str:
    """Canonicalize a URL for deduplication."""
    parsed = urllib.parse.urlparse(url.strip())
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    if ":" in netloc:
        host, port = netloc.split(":", 1)
        if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
            netloc = host
    path = parsed.path.rstrip("/")
    canonical = f"{scheme}://{netloc}{path}"
    if parsed.query:
        canonical += f"?{parsed.query}"
    return canonical


def is_domain_allowed(url: str, allowlist: list[str]) -> bool:
    """Check if the URL hostname matches any allowed domain."""
    if not allowlist:
        return True
    parsed = urllib.parse.urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    for allowed in allowlist:
        allowed_clean = allowed.strip().lower()
        if not allowed_clean:
            continue
        if host == allowed_clean or host.endswith("." + allowed_clean):
            return True
    return False


def is_series_relevant(title: str, snippet: str, manifest: SeriesManifest) -> bool:
    """Check if the title or snippet mentions the series name or aliases."""
    content_lower = f"{title} {snippet}".lower()
    candidates: list[str] = [
        manifest.display_name.lower(),
        str(manifest.series_id).lower(),
    ]
    for alias in manifest.aliases:
        alias_clean = alias.strip().lower()
        if alias_clean:
            candidates.append(alias_clean)
    name_tokens = [
        w.strip("(),.:;\"'")
        for w in manifest.display_name.lower().split()
        if len(w.strip("(),.:;\"'")) >= 3
    ]
    candidates.extend(name_tokens)
    return any(c in content_lower for c in candidates if c)


class LoreRetrievalService:
    """Orchestrates end-to-end series lore retrieval.

    Workflow:
    1. Resolve & validate series scope.
    2. Check cache.
    3. Classify query intent and expand aliases.
    4. Perform hybrid dense+sparse retrieval and reciprocal rank fusion.
    5. Rerank candidates with heuristic boosts and tie-breaking.
    6. Check sufficiency and run external fallback if insufficient.
    7. Build chronological narrative context.
    8. Build verified evidence items with provenance and return EvidenceBundle.
    """

    def __init__(
        self,
        scope_resolver: ScopeResolver,
        retriever: HybridRetriever,
        cache: Cache | None = None,
        web_search: WebSearch | None = None,
        budget_counter: BudgetCounter | None = None,
        classifier: QueryClassifier | None = None,
        chronology_builder: ChronologyBuilder | None = None,
        clock: Clock | None = None,
        web_search_settings: WebSearchSettings | None = None,
        limits_settings: RequestLimitsSettings | None = None,
        sufficiency_threshold: float = 0.01,
    ) -> None:
        self._scope_resolver = scope_resolver
        self._retriever = retriever
        self._cache = cache
        self._web_search = web_search
        self._budget_counter = budget_counter
        self._classifier = classifier or QueryClassifier()
        self._chronology_builder: ChronologyBuilder = (
            chronology_builder or DefaultChronologyBuilder()
        )
        self._clock: Clock = clock or SystemClock()
        self._web_search_settings = web_search_settings or WebSearchSettings()
        self._limits_settings = limits_settings or RequestLimitsSettings()
        self._sufficiency_threshold = sufficiency_threshold

    async def search(
        self,
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
        """Execute end-to-end lore search and return verified EvidenceBundle."""
        # 0. Query validation
        cleaned_query = query.strip()
        if not cleaned_query:
            raise InvalidArgument(
                "Query text must not be empty or whitespace.",
                argument_name="query",
            )
        if len(cleaned_query) > self._limits_settings.max_query_length:
            raise InvalidArgument(
                f"Query exceeds maximum character length of "
                f"{self._limits_settings.max_query_length}.",
                argument_name="query",
            )

        limit = (
            max_results
            if max_results is not None
            else self._limits_settings.default_max_results
        )
        if limit <= 0:
            raise InvalidArgument(
                "max_results must be greater than 0.", argument_name="max_results"
            )
        if limit > self._limits_settings.hard_max_results:
            limit = self._limits_settings.hard_max_results

        # 1. Resolve Scope
        resolved_scope = await self._scope_resolver.resolve(series)
        series_id = resolved_scope.series_id
        manifest = resolved_scope.manifest

        # 2. Classify Query and Expand Aliases
        classified = self._classifier.expand_aliases(cleaned_query, manifest)

        # Source type preference based on intent
        source_type_filter: str | None = None
        if classified.intent == "dialogue":
            # Dialogue strongly prefers transcripts
            source_type_filter = "transcript"

        search_query = SearchQuery(
            series_id=series_id,
            query=classified.raw_query,
            sparse_query=classified.sparse_query,
            season=season,
            episode=episode,
            max_season=max_season,
            max_episode=max_episode,
            entity=entity,
            source_type=source_type_filter,
            max_results=limit,
            include_unscoped=include_unscoped,
        )

        # 4 & 5. Hybrid Retrieval, Fusion & Reranking
        hits: list[SearchHit] = await self._retriever.retrieve(search_query)

        # If dialogue intent with transcript filter returned no hits,
        # retry without source filter
        if not hits and source_type_filter is not None:
            fallback_query = SearchQuery(
                series_id=series_id,
                query=classified.raw_query,
                sparse_query=classified.sparse_query,
                season=season,
                episode=episode,
                max_season=max_season,
                max_episode=max_episode,
                entity=entity,
                source_type=None,
                max_results=limit,
                include_unscoped=include_unscoped,
            )
            hits = await self._retriever.retrieve(fallback_query)

        # 6. Check Sufficiency and Fallback
        is_sufficient = bool(hits) and any(
            hit.score >= self._sufficiency_threshold for hit in hits
        )
        status: Any = "ok"
        notices: list[str] = []

        if classified.query_class == "complex_multi_entity":
            notices.append(
                "Notice: Query detected as multi-entity/complex. For best accuracy, "
                "consider decomposing into focused independent calls for each entity."
            )

        if max_season is not None or max_episode is not None:
            notices.append(
                f"Spoiler filter active: (season<={max_season}, episode<={max_episode})"
            )

        evidence_list: list[Evidence] = []
        if is_sufficient:
            # Map hits to Evidence models
            evidence_list = [self._hit_to_evidence(hit, series_id) for hit in hits]
        else:
            # Check if external web search fallback is available and allowed
            status = "insufficient"
            local_evidence = [self._hit_to_evidence(hit, series_id) for hit in hits]
            if self._web_search is not None:
                fallback_evidence, fallback_notice = await self._execute_web_fallback(
                    cleaned_query, series_id, manifest
                )
                if fallback_evidence:
                    evidence_list = fallback_evidence + local_evidence
                    status = "fallback_used"
                    notices.append("Retrieved from external web search fallback")
                else:
                    evidence_list = local_evidence
                    if fallback_notice:
                        notices.append(fallback_notice)
            else:
                evidence_list = local_evidence

        evidence_list = evidence_list[:limit]

        # 7. Build Chronological Context
        temporal_ctx = self._chronology_builder.build_context(evidence_list)

        bundle = EvidenceBundle(
            query=cleaned_query,
            series_id=series_id,
            status=status,
            evidence=evidence_list,
            truncated=False,
            chronology=temporal_ctx.timeline_notes,
            temporal_context=temporal_ctx,
            notices=notices,
        )

        return bundle

    def _hit_to_evidence(
        self, hit: SearchHit, series_id: ValidatedSeriesId
    ) -> Evidence:
        primary_entity = hit.chunk.entity_ids[0] if hit.chunk.entity_ids else None
        return Evidence(
            chunk_id=hit.chunk.chunk_id,
            series_id=series_id,
            content=hit.chunk.text,
            entity_id=primary_entity,
            season=hit.chunk.season,
            episode=hit.chunk.episode,
            scene_index=hit.chunk.scene_index,
            narrative_order=hit.chunk.narrative_order,
            narrative_confidence=hit.chunk.narrative_confidence,
            score=hit.score,
            dense_score=hit.dense_score,
            sparse_score=hit.sparse_score,
            rerank_score=hit.rerank_score,
            provenance=hit.chunk.provenance,
        )

    async def _execute_web_fallback(
        self,
        query: str,
        series_id: ValidatedSeriesId,
        manifest: SeriesManifest,
    ) -> tuple[list[Evidence], str | None]:
        if self._web_search is None:
            return [], None

        # Check daily budget if counter is configured (fail closed)
        if self._budget_counter is not None:
            today_str = self._clock.now().strftime("%Y-%m-%d")
            budget_key = f"lore:v1:budget:web_search_daily:{today_str}"
            try:
                has_budget = await self._budget_counter.check_budget(
                    budget_key, self._web_search_settings.daily_budget
                )
                if not has_budget:
                    logger.info("Daily web search budget reached. Skipping fallback.")
                    return [], "Web search budget exhausted"
                await self._budget_counter.increment(budget_key, window_seconds=86400)
            except Exception as err:
                logger.warning(
                    "Budget counter check failed: %s. Failing closed.",
                    err,
                )
                return [], "Web search budget counter unavailable"

        web_query = f"{manifest.display_name} {query}".strip()
        try:
            results = await self._web_search.search(web_query, limit=5)
        except Exception as err:
            logger.warning(
                "External web search provider failed for query '%s': %s",
                web_query,
                err,
            )
            return [], f"Web search provider error: {err}"

        fallback_evidence: list[Evidence] = []
        seen_urls: set[str] = set()
        now = self._clock.now()

        for res in results:
            # 1. Apply series domain allowlist
            if not is_domain_allowed(res.url, manifest.web_fallback_domain_allowlist):
                continue

            # 2. Canonicalize and remove duplicate URLs
            canonical_url = canonicalize_url(res.url)
            if canonical_url in seen_urls:
                continue
            seen_urls.add(canonical_url)

            # 3. Validate series relevance
            if not is_series_relevant(res.title, res.snippet, manifest):
                continue

            # 4. Construct verified Evidence model
            cid_hash = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:12]
            cid = ChunkId(f"web-{cid_hash}")
            ev = Evidence(
                chunk_id=cid,
                series_id=series_id,
                content=f"{res.title}: {res.snippet}",
                entity_id=None,
                season=None,
                episode=None,
                scene_index=None,
                narrative_order=None,
                score=0.5,
                dense_score=None,
                sparse_score=None,
                rerank_score=0.5,
                provenance=Provenance(
                    source_type="web",
                    source_url=res.url,
                    license=None,
                    attribution_text=res.title,
                    retrieval_timestamp=res.retrieval_timestamp or now,
                    revision_id=None,
                ),
            )
            fallback_evidence.append(ev)

        return fallback_evidence, None

    def _build_chronology(self, evidence_list: list[Evidence]) -> list[str]:
        """Construct sequential chronological notes from retrieved evidence items."""
        return self._chronology_builder.build_context(evidence_list).timeline_notes
