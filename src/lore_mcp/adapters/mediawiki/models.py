"""Data models for MediaWiki adapter."""

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lore_mcp.domain.documents import RawDocument
from lore_mcp.domain.types import SeriesId, ValidatedSeriesId


class MediaWikiPage(BaseModel):
    """Structured page fetched from MediaWiki Action API."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = Field(description="Canonical page title")
    page_id: int = Field(description="Unique MediaWiki page ID")
    source_url: str = Field(description="Canonical or full article URL")
    content: str = Field(description="Page text content (extract or parsed body)")
    revision_id: str | None = Field(
        default=None,
        description="Revision ID of the fetched page",
    )
    retrieval_timestamp: datetime = Field(
        description="Timestamp when page was retrieved",
    )
    license: str = Field(description="Verified wiki content license")
    categories: list[str] = Field(
        default_factory=list,
        description="Categories associated with the page",
    )
    redirects: list[str] = Field(
        default_factory=list,
        description="Redirect titles pointing to this page",
    )
    links: list[str] = Field(
        default_factory=list,
        description="Internal article links found within the page",
    )

    def to_raw_document(
        self,
        series_id: SeriesId | ValidatedSeriesId,
        source_type: str = "fandom",
        extra_metadata: dict[str, Any] | None = None,
    ) -> RawDocument:
        """Convert this MediaWikiPage into a domain RawDocument."""
        meta: dict[str, str] = {
            "title": self.title,
            "page_id": str(self.page_id),
            "revision_id": self.revision_id or "",
            "license": self.license,
            "categories": json.dumps(self.categories),
            "redirects": json.dumps(self.redirects),
            "links": json.dumps(self.links),
            "source_url": self.source_url,
        }
        if extra_metadata:
            for k, v in extra_metadata.items():
                meta[k] = str(v)

        return RawDocument(
            series_id=SeriesId(str(series_id)),
            source_url=self.source_url,
            source_type=source_type,
            content=self.content,
            retrieval_timestamp=self.retrieval_timestamp,
            metadata=meta,
        )
