"""MediaWiki implementation of the DocumentSource port interface."""

import logging
from collections.abc import AsyncIterator

from lore_mcp.adapters.mediawiki.client import MediaWikiClient
from lore_mcp.adapters.mediawiki.errors import MediaWikiAPIError, MediaWikiError
from lore_mcp.application.errors import DataUnavailable
from lore_mcp.domain.documents import RawDocument
from lore_mcp.domain.series_config import SeriesConfig, WikiEndpointConfig
from lore_mcp.domain.types import SeriesId
from lore_mcp.ports.documents import DocumentSource

logger = logging.getLogger(__name__)


class MediaWikiDocumentSource(DocumentSource):
    """MediaWiki DocumentSource adapter for series wiki endpoints."""

    def __init__(
        self,
        client: MediaWikiClient | None = None,
        endpoints: dict[SeriesId, list[WikiEndpointConfig]] | None = None,
    ) -> None:
        self._client = client or MediaWikiClient()
        self._endpoints: dict[SeriesId, list[WikiEndpointConfig]] = (
            dict(endpoints) if endpoints else {}
        )

    def register_endpoints(
        self,
        series_id: SeriesId | str,
        endpoints: list[WikiEndpointConfig],
    ) -> None:
        """Register wiki endpoints for a series."""
        sid = SeriesId(str(series_id))
        self._endpoints[sid] = list(endpoints)

    def register_series_config(self, config: SeriesConfig) -> None:
        """Register wiki endpoints directly from a SeriesConfig model."""
        self.register_endpoints(config.id, config.wiki_endpoints)

    def get_endpoints(self, series_id: SeriesId | str) -> list[WikiEndpointConfig]:
        """Get configured wiki endpoints for a series."""
        sid = SeriesId(str(series_id))
        return self._endpoints.get(sid, [])

    async def list_locators(self, series_id: SeriesId) -> list[str]:
        """List all article page titles (locators) for a series across its endpoints."""
        endpoints = self.get_endpoints(series_id)
        if not endpoints:
            return []

        locators: list[str] = []
        for ep in endpoints:
            params = {
                "action": "query",
                "generator": "allpages",
                "gaplimit": "500",
                "gapnamespace": "0",
                "prop": "info",
            }
            continue_params: dict[str, str] = {}
            while True:
                data = await self._client.request(
                    ep.api_url, {**params, **continue_params}
                )
                query_obj = data.get("query", {})
                pages_raw = query_obj.get("pages", [])
                page_dicts = self._client._extract_page_models(pages_raw)
                for p in page_dicts:
                    title = p.get("title")
                    if title and not p.get("missing", False):
                        locators.append(str(title))

                if "continue" in data and isinstance(data["continue"], dict):
                    continue_params = data["continue"]
                else:
                    break

        return locators

    async def fetch_document(self, series_id: SeriesId, locator: str) -> RawDocument:
        """Fetch raw document for a given locator (page title) from series endpoints."""
        endpoints = self.get_endpoints(series_id)
        if not endpoints:
            raise DataUnavailable(
                f"No wiki endpoints configured for series '{series_id}'",
                resource_id=locator,
            )

        last_error: Exception | None = None
        for ep in endpoints:
            try:
                page = await self._client.fetch_page(
                    endpoint_url=ep.api_url,
                    title=locator,
                    wiki_license=ep.license,
                )
                return page.to_raw_document(
                    series_id=SeriesId(series_id),
                    source_type=ep.wiki_type,
                    extra_metadata={"endpoint": ep.api_url},
                )
            except MediaWikiAPIError as err:
                if err.code == "missingtitle":
                    continue
                last_error = err
            except MediaWikiError as err:
                last_error = err

        msg = (
            f"Document '{locator}' not found on configured endpoints "
            f"for series '{series_id}'"
        )
        if last_error:
            msg += f" (last error: {last_error})"
        raise DataUnavailable(msg, resource_id=locator)

    async def stream_documents(
        self,
        series_id: SeriesId,
        limit_per_endpoint: int | None = None,
    ) -> AsyncIterator[RawDocument]:
        """Stream all raw documents across all wiki endpoints for a series."""
        endpoints = self.get_endpoints(series_id)
        if not endpoints:
            return

        for ep in endpoints:
            async for page in self._client.stream_pages(
                endpoint_url=ep.api_url,
                wiki_license=ep.license,
                limit=limit_per_endpoint,
            ):
                yield page.to_raw_document(
                    series_id=SeriesId(series_id),
                    source_type=ep.wiki_type,
                    extra_metadata={"endpoint": ep.api_url},
                )

    async def fetch_all_documents(
        self,
        series_id: SeriesId,
        limit_per_endpoint: int | None = None,
    ) -> list[RawDocument]:
        """Fetch and return all documents for a series into a list."""
        docs: list[RawDocument] = []
        async for doc in self.stream_documents(
            series_id, limit_per_endpoint=limit_per_endpoint
        ):
            docs.append(doc)
        return docs
