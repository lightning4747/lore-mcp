"""Domain models for series configuration schemas."""

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lore_mcp.domain.types import ValidatedSeriesId


class WikiEndpointConfig(BaseModel):
    """Configuration for an external wiki source endpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_url: str = Field(
        description="Action API endpoint URL (e.g. 'https://marvel.fandom.com/api.php')"
    )
    wiki_type: str = Field(
        default="fandom",
        description="Wiki engine type ('fandom', 'wikipedia', 'mediawiki')",
    )
    license: str = Field(
        description="Verified content license for attribution (e.g. 'CC-BY-SA-3.0')"
    )
    attribution: str | None = Field(
        default=None,
        description="Mandatory attribution or contributor notice if required",
    )

    @field_validator("api_url", "license")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field must not be empty or whitespace")
        return cleaned


class SeriesConfig(BaseModel):
    """Declarative specification defining a series, sources, and extraction rules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: ValidatedSeriesId = Field(
        description="Canonical series slug identifier matching filename"
    )
    display_names: list[str] = Field(
        description="Canonical and localized display names for the series"
    )
    kind: Literal["show", "anime"] = Field(
        default="show",
        description=(
            "Format category ('show' for live-action/animation, 'anime' for anime)"
        ),
    )
    aliases: list[str] = Field(
        default_factory=list[str],
        description="Alternative titles, abbreviations, and informal references",
    )
    wiki_endpoints: list[WikiEndpointConfig] = Field(
        description="Primary depth wiki Action API endpoints and verified licenses"
    )
    category_to_entity_type: dict[str, str] = Field(
        description="Mapping from wiki categories to canonical entity types"
    )
    web_fallback_domain_allowlist: list[str] = Field(
        default_factory=list[str],
        description="Domain allowlist permitted for dynamic web fallback search",
    )
    curated_timeline_path: str | None = Field(
        default=None,
        description="Optional relative path to curated chronology timeline data file",
    )
    transcripts_dir: str | None = Field(
        default=None,
        description=(
            "Optional relative path to local episode dialogue transcripts directory"
        ),
    )

    @field_validator("display_names")
    @classmethod
    def validate_display_names(cls, values: list[str]) -> list[str]:
        cleaned = [name.strip() for name in values if name.strip()]
        if not cleaned:
            raise ValueError("display_names must contain at least one non-empty name")
        return cleaned

    @field_validator("wiki_endpoints")
    @classmethod
    def validate_wiki_endpoints(
        cls, values: list[WikiEndpointConfig]
    ) -> list[WikiEndpointConfig]:
        if not values:
            raise ValueError("wiki_endpoints must contain at least one endpoint")
        return values

    @classmethod
    def from_yaml(cls, yaml_text: str) -> "SeriesConfig":
        """Parse and validate SeriesConfig from a YAML string."""
        parsed: object = yaml.safe_load(yaml_text)

        if not isinstance(parsed, dict):
            raise ValueError(
                f"YAML content must deserialize to a dictionary mapping, "
                f"got {type(parsed).__name__}"
            )

        return cls.model_validate(parsed)
