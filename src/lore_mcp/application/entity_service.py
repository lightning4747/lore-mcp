"""Application service for resolving canonical entities and associated evidence."""

import logging

from lore_mcp.application.errors import DataUnavailable, InvalidArgument
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.application.scope import ScopeResolver
from lore_mcp.config import RequestLimitsSettings
from lore_mcp.domain.entities import Entity, EntityDetails
from lore_mcp.domain.identifiers import slugify
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import Evidence, SearchHit, SearchQuery
from lore_mcp.domain.types import EntityId
from lore_mcp.ports.entity_registry import EntityRegistry

logger = logging.getLogger(__name__)


class EntityService:
    """Orchestrates canonical entity resolution, evidence retrieval, and provenance."""

    def __init__(
        self,
        scope_resolver: ScopeResolver,
        retriever: HybridRetriever,
        entity_registry: EntityRegistry | None = None,
        limits_settings: RequestLimitsSettings | None = None,
    ) -> None:
        self._scope_resolver = scope_resolver
        self._retriever = retriever
        self._entity_registry = entity_registry
        self._limits_settings = limits_settings or RequestLimitsSettings()

    async def get_entity(
        self,
        entity: str,
        series: str,
        max_results: int | None = None,
    ) -> EntityDetails:
        """Resolve a canonical entity by ID or alias, retrieve associated evidence,
        and return entity metadata with source provenance.
        """
        # 0. Input validation
        cleaned_entity = entity.strip()
        if not cleaned_entity:
            raise InvalidArgument(
                "Entity identifier or name must not be empty.",
                argument_name="entity",
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

        # 2. Resolve canonical entity
        canonical_entity: Entity | None = None
        hits: list[SearchHit] = []

        if self._entity_registry is not None:
            canonical_entity = await self._entity_registry.resolve_entity(
                series_id=series_id,
                name_or_alias=cleaned_entity,
            )

        # Fallback resolution from indexed chunks if not in registry
        if canonical_entity is None:
            query_slug = slugify(cleaned_entity)
            if query_slug:
                candidate_query = SearchQuery(
                    series_id=series_id,
                    query=cleaned_entity,
                    entity=query_slug,
                    max_results=limit,
                )
                hits = await self._retriever.retrieve(candidate_query)
                if hits:
                    canonical_entity = Entity(
                        entity_id=EntityId(query_slug),
                        series_id=series_id,
                        name=cleaned_entity.title(),
                        entity_type="character",
                        summary=f"Entity identified from {series_id} lore chunks.",
                    )

        if canonical_entity is None:
            raise DataUnavailable(
                f"Entity '{cleaned_entity}' not found in series '{series_id}'.",
                resource_id=cleaned_entity,
            )

        # 3. Retrieve entity-associated evidence if not already fetched
        if not hits:
            # First attempt: search constrained by entity_id
            search_query = SearchQuery(
                series_id=series_id,
                query=canonical_entity.name,
                entity=canonical_entity.entity_id,
                max_results=limit,
            )
            hits = await self._retriever.retrieve(search_query)

            # Second attempt: fallback to entity name and aliases if no hits
            if not hits:
                terms = [canonical_entity.name, *canonical_entity.aliases]
                fallback_text = " ".join(terms)

                fallback_query = SearchQuery(
                    series_id=series_id,
                    query=fallback_text,
                    max_results=limit,
                )
                hits = await self._retriever.retrieve(fallback_query)

        # 4. Map hits to Evidence objects
        evidence_list: list[Evidence] = [
            Evidence(
                chunk_id=hit.chunk.chunk_id,
                series_id=series_id,
                content=hit.chunk.text,
                entity_id=canonical_entity.entity_id,
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

        # 5. Aggregate deduplicated provenance records
        seen_urls: set[str] = set()
        provenance_list: list[Provenance] = []
        for ev in evidence_list:
            if ev.provenance.source_url not in seen_urls:
                seen_urls.add(ev.provenance.source_url)
                provenance_list.append(ev.provenance)

        return EntityDetails(
            entity=canonical_entity,
            evidence=evidence_list,
            provenance=provenance_list,
        )
