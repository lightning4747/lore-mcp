"""Event extractor implementing Task 4.3."""

import re

from lore_mcp.domain.documents import NormalizedDocument
from lore_mcp.domain.entities import AliasTable, Event
from lore_mcp.domain.identifiers import generate_event_id, slugify
from lore_mcp.domain.series_config import SeriesConfig
from lore_mcp.domain.types import EntityId
from lore_mcp.ports.extraction import EventExtractor

PLOT_HEADING_PATTERN = re.compile(
    r"^(plot|synopsis|story|events?|summary|recap|narrative|episode\s+\w+)$",
    re.IGNORECASE,
)

WIKILINK_PATTERN = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
SEASON_EPISODE_PATTERN = re.compile(r"S(\d{1,2})E(\d{1,2})", re.IGNORECASE)


def _extract_int_meta(metadata: dict[str, str], key: str) -> int | None:
    val = metadata.get(key)
    if val and val.isdigit():
        return int(val)
    return None


class DefaultEventExtractor(EventExtractor):
    """Rule-based extractor for Event records from episode plot sections."""

    def extract_events(
        self,
        doc: NormalizedDocument,
        config: SeriesConfig,
        entity_alias_table: AliasTable | None = None,
        curated_narrative_orders: dict[str, int] | None = None,
    ) -> list[Event]:
        """Extract zero or more Event records from a single normalized document."""
        # 1. Parse season and episode from metadata or title
        season = _extract_int_meta(doc.metadata, "season")
        episode = _extract_int_meta(doc.metadata, "episode")

        if season is None or episode is None:
            match = SEASON_EPISODE_PATTERN.search(doc.title)
            if match:
                if season is None:
                    season = int(match.group(1))
                if episode is None:
                    episode = int(match.group(2))

        # 2. Identify plot sections
        plot_sections = [
            sec
            for sec in doc.sections
            if PLOT_HEADING_PATTERN.match(sec.heading.strip())
        ]

        if not plot_sections:
            return []

        events: list[Event] = []
        scene_counter = 1

        # Pre-index locations from alias table if available
        location_keys: dict[str, str] = {}
        if entity_alias_table:
            for entry in entity_alias_table.entries:
                if entry.entity_type == "location":
                    location_keys[entry.lookup_key] = entry.canonical_name

        for sec in plot_sections:
            paragraphs = sec.content.split("\n\n")
            for para in paragraphs:
                cleaned_para = para.strip()
                if not cleaned_para or cleaned_para.startswith("#"):
                    continue

                # 3. Extract participants and location
                participants_set: set[EntityId] = set()
                matched_location: str | None = None

                # (a) Check inline wikilinks
                wikilinks = WIKILINK_PATTERN.findall(cleaned_para)
                for link in wikilinks:
                    if entity_alias_table:
                        target_id = entity_alias_table.lookup(link)
                        if target_id:
                            participants_set.add(target_id)
                            link_key = " ".join(link.strip().casefold().split())
                            if not matched_location and link_key in location_keys:
                                matched_location = location_keys[link_key]
                    else:
                        participants_set.add(EntityId(slugify(link)))

                # (b) Text matching against alias table for known entities
                if entity_alias_table:
                    para_lower = f" {cleaned_para.casefold()} "
                    for entry in entity_alias_table.entries:
                        # Only match key phrases of length >= 4 or capitalized terms
                        if (
                            len(entry.lookup_key) >= 4
                            and f" {entry.lookup_key} " in para_lower
                        ):
                            participants_set.add(entry.entity_id)
                            if not matched_location and entry.entity_type == "location":
                                matched_location = entry.canonical_name

                # Clean paragraph body by stripping any leftover wikilink brackets
                clean_description = WIKILINK_PATTERN.sub(r"\1", cleaned_para)

                # Derive event name from first sentence
                first_sentence = clean_description.split(".")[0].strip()
                if len(first_sentence) > 80:
                    event_name = first_sentence[:77] + "..."
                elif first_sentence:
                    event_name = first_sentence
                else:
                    event_name = f"Scene {scene_counter}"

                # 4. Generate deterministic event_id
                event_key = slugify(" ".join(clean_description.split()[:8]))
                event_id = generate_event_id(
                    series_id=config.id,
                    episode=episode,
                    event_key=event_key,
                    season=season,
                )

                # 5. Narrative order lookup
                narrative_order: int | None = None
                if curated_narrative_orders is not None:
                    narrative_order = curated_narrative_orders.get(str(event_id))

                events.append(
                    Event(
                        event_id=event_id,
                        series_id=config.id,
                        name=event_name,
                        description=clean_description,
                        season=season,
                        episode=episode,
                        scene_index=scene_counter,
                        narrative_order=narrative_order,
                        participants=sorted(participants_set),
                        location=matched_location,
                        consequences=[],
                    )
                )
                scene_counter += 1

        return events
