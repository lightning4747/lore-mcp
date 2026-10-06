"""MediaWiki source adapter implementing [S12] and [S13]."""

from lore_mcp.adapters.mediawiki.adapter import MediaWikiDocumentSource
from lore_mcp.adapters.mediawiki.client import DEFAULT_USER_AGENT, MediaWikiClient
from lore_mcp.adapters.mediawiki.errors import (
    MediaWikiAPIError,
    MediaWikiError,
    MediaWikiHTTPError,
    MediaWikiMaxlagError,
    MediaWikiNetworkError,
)
from lore_mcp.adapters.mediawiki.models import MediaWikiPage

__all__ = [
    "DEFAULT_USER_AGENT",
    "MediaWikiAPIError",
    "MediaWikiClient",
    "MediaWikiDocumentSource",
    "MediaWikiError",
    "MediaWikiHTTPError",
    "MediaWikiMaxlagError",
    "MediaWikiNetworkError",
    "MediaWikiPage",
]
