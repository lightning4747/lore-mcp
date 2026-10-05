"""Domain nominal identifiers and slug validation."""

import re
from typing import Annotated, NewType

from pydantic import BeforeValidator

SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*$")

SeriesId = NewType("SeriesId", str)
EntityId = NewType("EntityId", str)
ChunkId = NewType("ChunkId", str)
EventId = NewType("EventId", str)


def validate_series_id(value: object) -> SeriesId:
    """Validate and normalize a SeriesId slug."""
    if not isinstance(value, str):
        raise TypeError(f"SeriesId must be a string, got {type(value).__name__}")
    slug = value.strip().lower()
    if not slug or not SLUG_PATTERN.match(slug):
        raise ValueError(
            f"Invalid SeriesId slug: {value!r}. Must contain only lowercase "
            f"alphanumerics, underscores, and hyphens matching {SLUG_PATTERN.pattern}"
        )
    return SeriesId(slug)


ValidatedSeriesId = Annotated[SeriesId, BeforeValidator(validate_series_id)]
