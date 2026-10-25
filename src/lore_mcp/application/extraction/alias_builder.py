"""Builder for assembling per-series AliasTable indexes (Task 4.4)."""

import unicodedata
from collections.abc import Sequence

from lore_mcp.domain.entities import AliasEntry, AliasTable, Entity
from lore_mcp.domain.types import SeriesId, ValidatedSeriesId


def normalize_lookup_key(text: str) -> str:
    """Normalize text into an NFC-normalized, whitespace-collapsed, casefolded key."""
    cleaned = unicodedata.normalize("NFC", text.strip().casefold())
    # Collapse multiple whitespaces
    return " ".join(cleaned.split())


class AliasTableBuilder:
    """Constructs an AliasTable from a collection of canonical Entity records."""

    def build(
        self,
        series_id: SeriesId | ValidatedSeriesId,
        entities: Sequence[Entity],
    ) -> AliasTable:
        """Build an AliasTable for the given series and entities.

        Canonical names are indexed first, followed by each entity's aliases.
        If duplicate keys are encountered, the first-registered entry takes precedence.
        """
        seen_keys: set[str] = set()
        entries: list[AliasEntry] = []

        # Pass 1: Register canonical names
        for entity in entities:
            name_key = normalize_lookup_key(entity.name)
            if name_key and name_key not in seen_keys:
                seen_keys.add(name_key)
                entries.append(
                    AliasEntry(
                        lookup_key=name_key,
                        entity_id=entity.entity_id,
                        canonical_name=entity.name,
                        entity_type=entity.entity_type,
                    )
                )

        # Pass 2: Register aliases
        for entity in entities:
            for alias in entity.aliases:
                alias_key = normalize_lookup_key(alias)
                if alias_key and alias_key not in seen_keys:
                    seen_keys.add(alias_key)
                    entries.append(
                        AliasEntry(
                            lookup_key=alias_key,
                            entity_id=entity.entity_id,
                            canonical_name=entity.name,
                            entity_type=entity.entity_type,
                        )
                    )

        return AliasTable(
            series_id=series_id,
            entries=entries,
        )
