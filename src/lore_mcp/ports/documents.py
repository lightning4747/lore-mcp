"""DocumentSource and DocumentStore port interfaces."""

from typing import Protocol, runtime_checkable

from lore_mcp.domain.documents import NormalizedDocument, RawDocument
from lore_mcp.domain.types import SeriesId


@runtime_checkable
class DocumentSource(Protocol):
    """Port for fetching raw source documents from external origins."""

    async def fetch_document(self, series_id: SeriesId, locator: str) -> RawDocument:
        """Fetch raw document content by its source locator."""
        ...

    async def list_locators(self, series_id: SeriesId) -> list[str]:
        """List available document locators for a series."""
        ...


@runtime_checkable
class DocumentStore(Protocol):
    """Port for persisting and retrieving raw and normalized documents."""

    async def put_raw(self, doc: RawDocument) -> None:
        """Store a raw source document."""
        ...

    async def get_raw(self, series_id: SeriesId, source_url: str) -> RawDocument | None:
        """Retrieve a raw document by series and URL."""
        ...

    async def put_normalized(self, doc: NormalizedDocument) -> None:
        """Store a normalized document."""
        ...

    async def get_normalized(
        self, series_id: SeriesId, source_url: str
    ) -> NormalizedDocument | None:
        """Retrieve a normalized document by series and URL."""
        ...

    async def list_urls(self, series_id: SeriesId) -> list[str]:
        """List all stored document URLs for a series."""
        ...
