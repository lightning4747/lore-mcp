"""Application service for resolving narrative events and tracing
chronological evidence.
"""

import logging

from lore_mcp.application.chronology import DefaultChronologyBuilder
from lore_mcp.application.errors import DataUnavailable, InvalidArgument
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.application.scope import ScopeResolver
from lore_mcp.config import RequestLimitsSettings
from lore_mcp.domain.entities import (
    Entity,
    Event,
    EventComparison,
    EventComparisonSide,
    EventTrace,
)
from lore_mcp.domain.identifiers import generate_event_id, slugify
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import Evidence, SearchHit, SearchQuery
from lore_mcp.domain.types import EntityId, ValidatedSeriesId
from lore_mcp.ports.chronology import ChronologyBuilder
from lore_mcp.ports.entity_registry import EntityRegistry
from lore_mcp.ports.event_registry import EventRegistry

logger = logging.getLogger(__name__)

_GENERIC_TERMS: frozenset[str] = frozenset(
    {
        "event",
        "series",
        "about",
        "scene",
        "episode",
        "season",
        "show",
        "lore",
    }
)


def _extract_significant_terms(text: str) -> list[str]:
    """Extract non-generic query terms of length > 2."""
    terms = [t for t in text.lower().split() if len(t) > 2 and t not in _GENERIC_TERMS]
    if not terms:
        return [t for t in text.lower().split() if len(t) > 2]
    return terms


class EventService:
    """Orchestrates event resolution, related evidence retrieval,
    participant/entity extraction, and chronological ordering.
    """

    def __init__(
        self,
        scope_resolver: ScopeResolver,
        retriever: HybridRetriever,
        event_registry: EventRegistry | None = None,
        entity_registry: EntityRegistry | None = None,
        chronology_builder: ChronologyBuilder | None = None,
        limits_settings: RequestLimitsSettings | None = None,
    ) -> None:
        self._scope_resolver = scope_resolver
        self._retriever = retriever
        self._event_registry = event_registry
        self._entity_registry = entity_registry
        self._chronology_builder: ChronologyBuilder = (
            chronology_builder or DefaultChronologyBuilder()
        )
        self._limits_settings = limits_settings or RequestLimitsSettings()

    async def trace_event(
        self,
        query: str,
        series: str,
        event_id: str | None = None,
        max_results: int | None = None,
    ) -> EventTrace:
        """Resolve a narrative event, retrieve related evidence, and return
        participants, relevant locations/entities, and chronological evidence.
        """
        # 0. Input validation
        clean_query = query.strip() if query else ""
        if not clean_query:
            raise InvalidArgument(
                "Event search query must not be empty.",
                argument_name="query",
            )

        if event_id is not None and not event_id.strip():
            raise InvalidArgument(
                "Event identifier must not be empty if provided.",
                argument_name="event_id",
            )
        clean_event_id = event_id.strip() if event_id else None

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

        # 1. Resolve series scope
        resolved_scope = await self._scope_resolver.resolve(series)
        series_id = resolved_scope.series_id

        # 2. Resolve canonical event
        canonical_event: Event | None = None
        hits: list[SearchHit] = []

        # Fast path: check event_id in registry first if provided
        if clean_event_id and self._event_registry is not None:
            canonical_event = await self._event_registry.resolve_event(
                series_id=series_id,
                name_or_id=clean_event_id,
            )

        # If not resolved via event_id, check registry by query
        if canonical_event is None and self._event_registry is not None:
            canonical_event = await self._event_registry.resolve_event(
                series_id=series_id,
                name_or_id=clean_query,
            )

        # Fallback resolution from indexed lore chunks if not in registry
        if canonical_event is None:
            candidate_query = SearchQuery(
                series_id=series_id,
                query=clean_query,
                max_results=limit,
            )
            raw_hits = await self._retriever.retrieve(candidate_query)
            target_terms = _extract_significant_terms(clean_query)
            hits = [
                h
                for h in raw_hits
                if any(t in h.chunk.text.lower() for t in target_terms)
            ]
            if hits:
                best_hit = hits[0]
                event_slug = slugify(clean_query) or f"event-{best_hit.chunk.chunk_id}"
                canonical_event = Event(
                    event_id=generate_event_id(
                        series_id=series_id,
                        episode=best_hit.chunk.episode,
                        event_key=event_slug,
                        season=best_hit.chunk.season,
                    ),
                    series_id=series_id,
                    name=clean_query.title(),
                    description=best_hit.chunk.text,
                    season=best_hit.chunk.season,
                    episode=best_hit.chunk.episode,
                    scene_index=best_hit.chunk.scene_index,
                    narrative_order=best_hit.chunk.narrative_order,
                    participants=[EntityId(e) for e in best_hit.chunk.entity_ids],
                    location=None,
                )

        if canonical_event is None:
            raise DataUnavailable(
                f"Event '{clean_query}' not found in series '{series_id}'.",
                resource_id=clean_query,
            )

        # 3. Retrieve event-associated evidence if not already fetched
        if not hits:
            # Query constrained by event season/episode if available
            scoped_query = SearchQuery(
                series_id=series_id,
                query=canonical_event.name,
                season=canonical_event.season,
                episode=canonical_event.episode,
                max_results=limit,
            )
            hits = await self._retriever.retrieve(scoped_query)

            # If scoped search yielded zero hits, fallback to broader query
            if not hits:
                desc_snippet = canonical_event.description[:80]
                broader_text = f"{canonical_event.name} {desc_snippet}".strip()
                broader_query = SearchQuery(
                    series_id=series_id,
                    query=broader_text,
                    max_results=limit,
                )
                raw_broader_hits = await self._retriever.retrieve(broader_query)
                broader_terms = _extract_significant_terms(broader_text)
                hits = [
                    bh
                    for bh in raw_broader_hits
                    if any(t in bh.chunk.text.lower() for t in broader_terms)
                ][:limit]

        # 4. Map hits to Evidence objects
        evidence_list: list[Evidence] = [
            Evidence(
                chunk_id=hit.chunk.chunk_id,
                series_id=series_id,
                content=hit.chunk.text,
                entity_id=hit.chunk.entity_ids[0] if hit.chunk.entity_ids else None,
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
            for hit in hits
        ]

        # 5. Order evidence chronologically
        evidence_list.sort(key=self._chronological_sort_key)

        # 6. Extract participants, relevant locations, and entities
        seen_participants: set[str] = set()
        participants: list[EntityId] = []
        for p in canonical_event.participants:
            p_str = str(p)
            if p_str not in seen_participants:
                seen_participants.add(p_str)
                participants.append(p)

        seen_locations: set[str] = set()
        locations: list[str] = []
        if canonical_event.location:
            seen_locations.add(canonical_event.location)
            locations.append(canonical_event.location)

        seen_entities: set[str] = set(seen_participants)
        entities: list[EntityId] = list(participants)
        for hit in hits:
            for eid in hit.chunk.entity_ids:
                if str(eid) not in seen_entities:
                    seen_entities.add(str(eid))
                    entities.append(eid)
        for ev in evidence_list:
            if ev.entity_id and str(ev.entity_id) not in seen_entities:
                seen_entities.add(str(ev.entity_id))
                entities.append(ev.entity_id)

        # 7. Build chronological context
        temporal_ctx = self._chronology_builder.build_context(evidence_list)

        # 8. Deduplicate provenance records
        seen_urls: set[str] = set()
        provenance_list: list[Provenance] = []
        for ev in evidence_list:
            if ev.provenance.source_url not in seen_urls:
                seen_urls.add(ev.provenance.source_url)
                provenance_list.append(ev.provenance)

        return EventTrace(
            event=canonical_event,
            participants=participants,
            locations=locations,
            entities=entities,
            evidence=evidence_list,
            chronology=temporal_ctx.timeline_notes,
            temporal_context=temporal_ctx,
            provenance=provenance_list,
            notices=[],
        )

    @staticmethod
    def _chronological_sort_key(ev: Evidence) -> tuple[int, int, int, int]:
        """Sort key for chronological narrative ordering."""
        # When all markers are present:
        # season -> episode -> narrative_order -> scene_index
        season_val = ev.season if ev.season is not None else 0
        episode_val = ev.episode if ev.episode is not None else 0
        order_val = ev.narrative_order if ev.narrative_order is not None else 0
        scene_val = ev.scene_index if ev.scene_index is not None else 0
        return (season_val, episode_val, order_val, scene_val)

    def _build_chronology(self, evidence_list: list[Evidence]) -> list[str]:
        """Construct chronological timeline markers from evidence."""
        return self._chronology_builder.build_context(evidence_list).timeline_notes

    async def compare_events(
        self,
        query_a: str,
        query_b: str,
        series: str,
        event_id_a: str | None = None,
        event_id_b: str | None = None,
        max_results: int | None = None,
    ) -> EventComparison:
        """Compare two event or entity references, retrieving evidence for both,
        keeping evidence and provenance separate, and returning comparable
        structured evidence.
        """
        # 0. Input validation
        clean_query_a = query_a.strip() if query_a else ""
        clean_query_b = query_b.strip() if query_b else ""
        if not clean_query_a:
            raise InvalidArgument(
                "First event query must not be empty.",
                argument_name="query_a",
            )
        if not clean_query_b:
            raise InvalidArgument(
                "Second event query must not be empty.",
                argument_name="query_b",
            )

        if event_id_a is not None and not event_id_a.strip():
            raise InvalidArgument(
                "First event identifier must not be empty if provided.",
                argument_name="event_id_a",
            )
        if event_id_b is not None and not event_id_b.strip():
            raise InvalidArgument(
                "Second event identifier must not be empty if provided.",
                argument_name="event_id_b",
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

        # 1. Resolve series scope
        resolved_scope = await self._scope_resolver.resolve(series)
        series_id = resolved_scope.series_id

        # 2. Resolve both sides independently
        side_a = await self._resolve_reference_side(
            series_id=series_id,
            query=clean_query_a,
            limit=limit,
            event_id=event_id_a.strip() if event_id_a else None,
        )
        side_b = await self._resolve_reference_side(
            series_id=series_id,
            query=clean_query_b,
            limit=limit,
            event_id=event_id_b.strip() if event_id_b else None,
        )

        # 3. Compute shared entities and locations
        side_b_entity_set = {str(e) for e in side_b.entities}
        shared_entities: list[EntityId] = [
            e for e in side_a.entities if str(e) in side_b_entity_set
        ]

        side_b_loc_set = {loc.lower() for loc in side_b.locations}
        shared_locations: list[str] = [
            loc for loc in side_a.locations if loc.lower() in side_b_loc_set
        ]

        return EventComparison(
            series_id=series_id,
            side_a=side_a,
            side_b=side_b,
            shared_entities=shared_entities,
            shared_locations=shared_locations,
            notices=[],
        )

    async def _resolve_reference_side(
        self,
        series_id: ValidatedSeriesId,
        query: str,
        limit: int,
        event_id: str | None = None,
    ) -> EventComparisonSide:
        """Resolve an event or entity reference and build its evidence side."""
        canonical_event: Event | None = None
        canonical_entity: Entity | None = None

        # 1. Fast path: check event_id in registries if provided
        if event_id and self._event_registry is not None:
            canonical_event = await self._event_registry.resolve_event(
                series_id=series_id,
                name_or_id=event_id,
            )
        if event_id and canonical_event is None and self._entity_registry is not None:
            canonical_entity = await self._entity_registry.resolve_entity(
                series_id=series_id,
                name_or_alias=event_id,
            )

        # 2. Check registries by natural language query
        if (
            canonical_event is None
            and canonical_entity is None
            and self._event_registry is not None
        ):
            canonical_event = await self._event_registry.resolve_event(
                series_id=series_id,
                name_or_id=query,
            )
        if (
            canonical_event is None
            and canonical_entity is None
            and self._entity_registry is not None
        ):
            canonical_entity = await self._entity_registry.resolve_entity(
                series_id=series_id,
                name_or_alias=query,
            )

        hits: list[SearchHit] = []

        # 3. Fallback resolution from indexed lore chunks if not found in registries
        if canonical_event is None and canonical_entity is None:
            candidate_query = SearchQuery(
                series_id=series_id,
                query=query,
                max_results=limit,
            )
            raw_hits = await self._retriever.retrieve(candidate_query)
            target_terms = _extract_significant_terms(query)
            hits = [
                h
                for h in raw_hits
                if any(t in h.chunk.text.lower() for t in target_terms)
            ]
            if hits:
                best_hit = hits[0]
                event_slug = slugify(query) or f"event-{best_hit.chunk.chunk_id}"
                canonical_event = Event(
                    event_id=generate_event_id(
                        series_id=series_id,
                        episode=best_hit.chunk.episode,
                        event_key=event_slug,
                        season=best_hit.chunk.season,
                    ),
                    series_id=series_id,
                    name=query.title(),
                    description=best_hit.chunk.text,
                    season=best_hit.chunk.season,
                    episode=best_hit.chunk.episode,
                    scene_index=best_hit.chunk.scene_index,
                    narrative_order=best_hit.chunk.narrative_order,
                    participants=[EntityId(e) for e in best_hit.chunk.entity_ids],
                    location=None,
                )

        if canonical_event is None and canonical_entity is None:
            raise DataUnavailable(
                f"Event or entity '{query}' not found in series '{series_id}'.",
                resource_id=query,
            )

        # 4. Retrieve evidence chunks
        if canonical_event is not None:
            if not hits:
                scoped_query = SearchQuery(
                    series_id=series_id,
                    query=canonical_event.name,
                    season=canonical_event.season,
                    episode=canonical_event.episode,
                    max_results=limit,
                )
                hits = await self._retriever.retrieve(scoped_query)
                if not hits:
                    desc_snippet = canonical_event.description[:80]
                    broader_text = f"{canonical_event.name} {desc_snippet}".strip()
                    broader_query = SearchQuery(
                        series_id=series_id,
                        query=broader_text,
                        max_results=limit,
                    )
                    raw_broader_hits = await self._retriever.retrieve(broader_query)
                    broader_terms = _extract_significant_terms(broader_text)
                    hits = [
                        bh
                        for bh in raw_broader_hits
                        if any(t in bh.chunk.text.lower() for t in broader_terms)
                    ][:limit]

            evidence_list: list[Evidence] = [
                Evidence(
                    chunk_id=hit.chunk.chunk_id,
                    series_id=series_id,
                    content=hit.chunk.text,
                    entity_id=hit.chunk.entity_ids[0] if hit.chunk.entity_ids else None,
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
                for hit in hits
            ]
            evidence_list.sort(key=self._chronological_sort_key)

            seen_participants: set[str] = set()
            participants: list[EntityId] = []
            for p in canonical_event.participants:
                p_str = str(p)
                if p_str not in seen_participants:
                    seen_participants.add(p_str)
                    participants.append(p)

            locations: list[str] = []
            if canonical_event.location:
                locations.append(canonical_event.location)

            seen_entities: set[str] = set(seen_participants)
            entities: list[EntityId] = list(participants)
            for hit in hits:
                for eid in hit.chunk.entity_ids:
                    if str(eid) not in seen_entities:
                        seen_entities.add(str(eid))
                        entities.append(eid)
            for ev in evidence_list:
                if ev.entity_id and str(ev.entity_id) not in seen_entities:
                    seen_entities.add(str(ev.entity_id))
                    entities.append(ev.entity_id)

            target_type = "event"

        else:
            assert canonical_entity is not None
            if not hits:
                entity_query = SearchQuery(
                    series_id=series_id,
                    query=canonical_entity.name,
                    entity=canonical_entity.entity_id,
                    max_results=limit,
                )
                hits = await self._retriever.retrieve(entity_query)
                if not hits:
                    terms = [canonical_entity.name, *canonical_entity.aliases]
                    fallback_text = " ".join(terms)
                    fallback_query = SearchQuery(
                        series_id=series_id,
                        query=fallback_text,
                        max_results=limit,
                    )
                    hits = await self._retriever.retrieve(fallback_query)

            evidence_list = [
                Evidence(
                    chunk_id=hit.chunk.chunk_id,
                    series_id=series_id,
                    content=hit.chunk.text,
                    entity_id=canonical_entity.entity_id,
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
                for hit in hits
            ]
            evidence_list.sort(key=self._chronological_sort_key)

            participants = [canonical_entity.entity_id]
            locations = (
                [canonical_entity.name]
                if canonical_entity.entity_type == "location"
                else []
            )

            seen_entities = {str(canonical_entity.entity_id)}
            entities = [canonical_entity.entity_id]
            for hit in hits:
                for eid in hit.chunk.entity_ids:
                    if str(eid) not in seen_entities:
                        seen_entities.add(str(eid))
                        entities.append(eid)
            for rel_id in canonical_entity.related_entities:
                if str(rel_id) not in seen_entities:
                    seen_entities.add(str(rel_id))
                    entities.append(rel_id)

            target_type = "entity"

        temporal_ctx = self._chronology_builder.build_context(evidence_list)

        seen_urls: set[str] = set()
        provenance_list: list[Provenance] = []
        for ev in evidence_list:
            if ev.provenance.source_url not in seen_urls:
                seen_urls.add(ev.provenance.source_url)
                provenance_list.append(ev.provenance)

        return EventComparisonSide(
            reference=event_id if event_id else query,
            target_type=target_type,
            event=canonical_event,
            entity=canonical_entity,
            participants=participants,
            locations=locations,
            entities=entities,
            evidence=evidence_list,
            chronology=temporal_ctx.timeline_notes,
            temporal_context=temporal_ctx,
            provenance=provenance_list,
        )
