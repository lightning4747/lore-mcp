"""Application service for resolving narrative events and tracing
chronological evidence.
"""

import logging

from lore_mcp.application.errors import DataUnavailable, InvalidArgument
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.application.scope import ScopeResolver
from lore_mcp.config import RequestLimitsSettings
from lore_mcp.domain.entities import Event, EventTrace
from lore_mcp.domain.identifiers import generate_event_id, slugify
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import Evidence, SearchHit, SearchQuery
from lore_mcp.domain.types import EntityId
from lore_mcp.ports.event_registry import EventRegistry

logger = logging.getLogger(__name__)


class EventService:
    """Orchestrates event resolution, related evidence retrieval,
    participant/entity extraction, and chronological ordering.
    """

    def __init__(
        self,
        scope_resolver: ScopeResolver,
        retriever: HybridRetriever,
        event_registry: EventRegistry | None = None,
        limits_settings: RequestLimitsSettings | None = None,
    ) -> None:
        self._scope_resolver = scope_resolver
        self._retriever = retriever
        self._event_registry = event_registry
        self._limits_settings = limits_settings or RequestLimitsSettings()

    async def trace_event(
        self,
        event_id: str,
        series: str,
        event: str | None = None,
        max_results: int | None = None,
    ) -> EventTrace:
        """Resolve a narrative event, retrieve related evidence, and return
        participants, relevant locations/entities, and chronological evidence.
        """
        # 0. Input validation
        target_event = (event or event_id).strip()
        if not target_event:
            raise InvalidArgument(
                "Event identifier or name must not be empty.",
                argument_name="event_id",
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

        # 2. Resolve canonical event
        canonical_event: Event | None = None
        hits: list[SearchHit] = []

        if self._event_registry is not None:
            canonical_event = await self._event_registry.resolve_event(
                series_id=series_id,
                name_or_id=target_event,
            )

        # Fallback resolution from indexed lore chunks if not in registry
        if canonical_event is None:
            candidate_query = SearchQuery(
                series_id=series_id,
                query=target_event,
                max_results=limit,
            )
            raw_hits = await self._retriever.retrieve(candidate_query)
            target_terms = [t for t in target_event.lower().split() if len(t) > 2]
            hits = [
                h
                for h in raw_hits
                if any(t in h.chunk.text.lower() for t in target_terms)
            ]
            if hits:
                best_hit = hits[0]
                event_slug = slugify(target_event) or f"event-{best_hit.chunk.chunk_id}"
                canonical_event = Event(
                    event_id=generate_event_id(
                        series_id=series_id,
                        episode=best_hit.chunk.episode,
                        event_key=event_slug,
                        season=best_hit.chunk.season,
                    ),
                    series_id=series_id,
                    name=target_event.title(),
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
                f"Event '{target_event}' not found in series '{series_id}'.",
                resource_id=target_event,
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

            # If scoped search yielded few hits, supplement with broader query
            if len(hits) < limit:
                desc_snippet = canonical_event.description[:80]
                broader_text = f"{canonical_event.name} {desc_snippet}".strip()
                broader_query = SearchQuery(
                    series_id=series_id,
                    query=broader_text,
                    max_results=limit,
                )
                broader_hits = await self._retriever.retrieve(broader_query)
                seen_chunk_ids = {h.chunk.chunk_id for h in hits}
                for bh in broader_hits:
                    if bh.chunk.chunk_id not in seen_chunk_ids:
                        hits.append(bh)
                        seen_chunk_ids.add(bh.chunk.chunk_id)
                hits = hits[:limit]

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

        # 7. Build chronological notes
        chronology_notes = self._build_chronology(evidence_list)

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
            chronology=chronology_notes,
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

    @staticmethod
    def _build_chronology(evidence_list: list[Evidence]) -> list[str]:
        """Construct chronological timeline markers from evidence."""
        notes: list[str] = []
        for ev in evidence_list:
            loc: list[str] = []
            if ev.season is not None:
                loc.append(f"S{ev.season}")
            if ev.episode is not None:
                loc.append(f"E{ev.episode}")
            if ev.scene_index is not None:
                loc.append(f"scene {ev.scene_index}")
            if ev.narrative_order is not None:
                loc.append(f"order {ev.narrative_order}")
            loc_str = " ".join(loc) if loc else "timeline"

            snippet = ev.content[:80] + ("..." if len(ev.content) > 80 else "")
            notes.append(f"[{loc_str}] {snippet}")
        return notes
