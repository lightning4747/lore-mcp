"""Domain provenance model tracking origin and licensing of retrieved knowledge."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Provenance(BaseModel):
    """Source provenance tracking origin, license, and timestamp."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_type: str = Field(
        description=(
            "Type of source material (e.g., 'fandom', 'transcript', 'wikipedia', 'web')"
        )
    )
    source_url: str = Field(
        description="Canonical URL or locator of the original source"
    )
    license: str | None = Field(
        default=None,
        description="License identifier or declaration (e.g. 'CC-BY-SA-3.0')",
    )
    attribution_text: str | None = Field(
        default=None,
        description="Attribution or copyright statement required by the source",
    )
    retrieval_timestamp: datetime = Field(
        description="Timestamp when the source material was fetched"
    )
    revision_id: str | None = Field(
        default=None,
        description="Version, revision, or commit identifier of the source document",
    )
