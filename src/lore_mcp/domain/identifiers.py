"""Deterministic, stable identifiers for entities, chunks, and events."""

import re
import uuid

from lore_mcp.domain.types import ChunkId, EntityId, EventId, SeriesId

# Namespace UUID for deterministic chunk ID generation
LORE_CHUNK_NAMESPACE: uuid.UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "lore-mcp.org")


def slugify(text: str) -> str:
    """Convert text to a clean, lowercase URL/identifier-safe slug."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned)
    return cleaned.strip("-_")


def generate_entity_id(canonical_title: str) -> EntityId:
    """Generate a stable entity ID from its canonical title within a series.

    The entity ID is the slug of the canonical title within the series.
    """
    slug = slugify(canonical_title)
    if not slug:
        raise ValueError(f"canonical_title {canonical_title!r} produced an empty slug")
    return EntityId(slug)


def generate_chunk_id(
    series_id: SeriesId | str,
    source_locator: str,
    section_path: str,
    split_index: int,
) -> ChunkId:
    """Generate a deterministic UUID5 chunk ID based on structural coordinates.

    Calculated over (series, source locator, section path, split index), not over
    content, so content edits overwrite existing chunks instead of creating duplicates.
    """
    if split_index < 0:
        raise ValueError(f"split_index must be non-negative, got {split_index}")
    key = f"{series_id}:{source_locator}:{section_path}:{split_index}"
    return ChunkId(str(uuid.uuid5(LORE_CHUNK_NAMESPACE, key)))


def generate_event_id(
    series_id: SeriesId | str,
    episode: int | None,
    event_key: str,
    season: int | None = None,
) -> EventId:
    """Generate a deterministic event ID from series, episode, and event key."""
    slug = slugify(event_key)
    if not slug:
        raise ValueError(f"event_key {event_key!r} produced an empty slug")

    prefix = f"{series_id}"
    if season is not None and episode is not None:
        return EventId(f"{prefix}:s{season}e{episode}:{slug}")
    if episode is not None:
        return EventId(f"{prefix}:e{episode}:{slug}")
    return EventId(f"{prefix}:{slug}")
