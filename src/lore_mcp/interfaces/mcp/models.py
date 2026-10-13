"""Data models for MCP tool inputs and outputs."""

from pydantic import BaseModel, ConfigDict, Field

from lore_mcp.domain.entities import EntityDetails, EventDetails, EventTrace
from lore_mcp.domain.search import DialogueSnippet


class SeriesSummary(BaseModel):
    """Metadata summary of a registered series for client discovery."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: str = Field(description="Unique canonical series identifier slug")
    display_name: str = Field(description="Human-readable title of the series")
    aliases: list[str] = Field(
        default_factory=list[str],
        description="Alternative names, titles, and shorthand references",
    )
    kind: str = Field(
        default="show",
        description="Series category/kind (e.g. 'show', 'anime', 'movie')",
    )


__all__ = [
    "DialogueSnippet",
    "EntityDetails",
    "EventDetails",
    "EventTrace",
    "SeriesSummary",
]
