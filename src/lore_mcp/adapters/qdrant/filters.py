"""Qdrant metadata filter builder enforcing series scope and bounds [S10, S11]."""

from qdrant_client import models

from lore_mcp.domain.search import SearchQuery
from lore_mcp.domain.types import EntityId, SeriesId


def build_metadata_filter(
    series_id: SeriesId | str,
    *,
    season: int | None = None,
    episode: int | None = None,
    max_season: int | None = None,
    max_episode: int | None = None,
    entity: EntityId | str | None = None,
    entity_type: str | None = None,
    source_type: str | None = None,
    include_unscoped: bool = False,
) -> models.Filter:
    """Build a Qdrant metadata filter requiring series and supporting all constraints.

    Enforces:
    - series (mandatory): every filter requires series_id.
    - season and episode: equality checks or spoiler-guard "up to" range constraints:
      (season, episode) <= (max_season, max_episode).
    - unscoped chunks: when spoiler bounds are active, unscoped chunks (season is null)
      are excluded unless include_unscoped is True.
    - entity: matches entity_id list in payload.
    - entity_type: matches entity_type in payload.
    - source_type: matches source_type in payload.
    """
    if not series_id or not str(series_id).strip():
        raise ValueError("series_id is mandatory and cannot be empty")

    must_conditions: list[models.Condition] = [
        models.FieldCondition(
            key="series",
            match=models.MatchValue(value=str(series_id)),
        )
    ]

    if entity is not None:
        must_conditions.append(
            models.FieldCondition(
                key="entity_id",
                match=models.MatchValue(value=str(entity)),
            )
        )

    if entity_type is not None:
        must_conditions.append(
            models.FieldCondition(
                key="entity_type",
                match=models.MatchValue(value=str(entity_type)),
            )
        )

    if source_type is not None:
        must_conditions.append(
            models.FieldCondition(
                key="source_type",
                match=models.MatchValue(value=str(source_type)),
            )
        )

    # Exact equality filters
    if season is not None:
        must_conditions.append(
            models.FieldCondition(
                key="season",
                match=models.MatchValue(value=season),
            )
        )

    if episode is not None:
        must_conditions.append(
            models.FieldCondition(
                key="episode",
                match=models.MatchValue(value=episode),
            )
        )

    # "Up to" range bounds (Spoiler protection)
    should_conditions: list[models.Condition] | None = None

    has_spoiler_bound = max_season is not None or max_episode is not None
    if has_spoiler_bound:
        branches: list[models.Condition] = []

        if max_season is not None and max_episode is not None:
            # Earlier seasons strictly before max_season
            branches.append(
                models.FieldCondition(
                    key="season",
                    range=models.Range(lt=max_season),
                )
            )
            # Current season up to max_episode
            branches.append(
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="season",
                            match=models.MatchValue(value=max_season),
                        ),
                        models.FieldCondition(
                            key="episode",
                            range=models.Range(lte=max_episode),
                        ),
                    ]
                )
            )
        elif max_season is not None:
            # Up to max_season (all episodes)
            branches.append(
                models.FieldCondition(
                    key="season",
                    range=models.Range(lte=max_season),
                )
            )
        elif max_episode is not None:
            # Up to max_episode (any season)
            branches.append(
                models.FieldCondition(
                    key="episode",
                    range=models.Range(lte=max_episode),
                )
            )

        if include_unscoped:
            branches.append(
                models.IsEmptyCondition(is_empty=models.PayloadField(key="season"))
            )

        should_conditions = branches

    return models.Filter(
        must=must_conditions,
        should=should_conditions,
    )


def build_filter_from_query(query: SearchQuery) -> models.Filter:
    """Build a Qdrant metadata filter from a domain SearchQuery."""
    return build_metadata_filter(
        series_id=query.series_id,
        season=query.season,
        episode=query.episode,
        max_season=query.max_season,
        max_episode=query.max_episode,
        entity=query.entity,
        entity_type=query.entity_type,
        source_type=query.source_type,
        include_unscoped=query.include_unscoped,
    )
