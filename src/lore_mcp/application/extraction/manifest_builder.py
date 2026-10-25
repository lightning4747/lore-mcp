"""Builder for constructing updated SeriesManifest models (Task 4.4)."""

from collections.abc import Sequence

from lore_mcp.domain.entities import Entity, Event, SeriesManifest
from lore_mcp.domain.series_config import SeriesConfig


class ManifestBuilder:
    """Assembles a SeriesManifest from extracted entities, events, and versions."""

    def build(
        self,
        config: SeriesConfig,
        entities: Sequence[Entity],
        events: Sequence[Event] | None = None,
        previous_manifest: SeriesManifest | None = None,
        schema_version: int = 1,
    ) -> SeriesManifest:
        """Construct an updated SeriesManifest.

        Deduces seasons from extracted events, counts entities, preserves chunk counts,
        increments the ingest version, and applies the designated schema version.
        """
        all_events = events or []
        seasons = sorted(
            {e.season for e in all_events if e.season is not None and e.season > 0}
        )

        # If previous manifest had seasons and no events provided,
        # preserve previous seasons
        if not seasons and previous_manifest and previous_manifest.seasons:
            seasons = list(previous_manifest.seasons)

        entity_count = len(entities)
        chunk_count = previous_manifest.chunk_count if previous_manifest else 0
        ingest_version = (
            (previous_manifest.ingest_version + 1) if previous_manifest else 1
        )

        display_name = (
            config.display_names[0]
            if config.display_names
            else str(config.id).title()
        )

        entity_aliases: dict[str, str] = {}
        for entity in entities:
            for alias in entity.aliases:
                cleaned = alias.strip().lower()
                if cleaned and cleaned not in entity_aliases:
                    entity_aliases[cleaned] = entity.name

        return SeriesManifest(
            series_id=config.id,
            display_name=display_name,
            kind=config.kind,
            aliases=list(config.aliases),
            seasons=seasons,
            entity_count=entity_count,
            chunk_count=chunk_count,
            entity_aliases=entity_aliases,
            ingest_version=ingest_version,
            schema_version=schema_version,
        )
