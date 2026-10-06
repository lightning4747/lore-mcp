"""MediaWiki Action API client implementing [S12] and [S13] conventions."""

import asyncio
import logging
import time
import urllib.parse
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import httpx

from lore_mcp.adapters.mediawiki.errors import (
    MediaWikiAPIError,
    MediaWikiHTTPError,
    MediaWikiMaxlagError,
    MediaWikiNetworkError,
)
from lore_mcp.adapters.mediawiki.models import MediaWikiPage

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "LoreMCP/0.1.0 (https://github.com/lightning4747/lore-mcp; lore-mcp@example.com)"
)


class MediaWikiClient:
    """Async MediaWiki Action API client.

    Adheres to:
    - Descriptive User-Agent [S13]
    - maxlag=5 throttling and retry [S13]
    - Generator queries and continue tokens [S12]
    - HTTP 200 API error payload detection
    - Bounded retries with exponential backoff
    - Content extraction (extracts or parsed text fallback)
    """

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        maxlag: int = 5,
        min_request_interval: float = 0.5,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        timeout: float = 15.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.maxlag = maxlag
        self.min_request_interval = min_request_interval
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout

        self._throttle_lock = asyncio.Lock()
        self._last_request_time: float = 0.0

        if http_client is not None:
            self._client = http_client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(
                timeout=timeout,
                headers={"User-Agent": self.user_agent},
            )
            self._owns_client = True

    async def close(self) -> None:
        """Close the underlying HTTP client if owned."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "MediaWikiClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.close()

    async def _throttle(self) -> None:
        """Throttle requests to respect server etiquette [S13]."""
        if self.min_request_interval <= 0:
            return

        async with self._throttle_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - elapsed)
            self._last_request_time = time.monotonic()

    def _calculate_backoff(
        self,
        attempt: int,
        retry_after: float | None = None,
    ) -> float:
        """Calculate backoff delay with Retry-After and exponential backoff."""
        if retry_after is not None and retry_after > 0:
            return min(retry_after, self.max_delay)
        delay = self.base_delay * (2**attempt)
        return min(delay, self.max_delay)

    async def request(
        self,
        endpoint_url: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute an Action API request with retries, maxlag, and error inspection."""
        query_params = {
            "format": "json",
            "formatversion": "2",
            "maxlag": str(self.maxlag),
            **params,
        }

        headers = {"User-Agent": self.user_agent}

        for attempt in range(self.max_retries + 1):
            await self._throttle()
            try:
                response = await self._client.get(
                    endpoint_url,
                    params=query_params,
                    headers=headers,
                )
            except (httpx.TimeoutException, httpx.NetworkError) as err:
                if attempt < self.max_retries:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        "Network error on %s (attempt %d/%d): %s. Backing off %.2fs",
                        endpoint_url,
                        attempt + 1,
                        self.max_retries,
                        err,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise MediaWikiNetworkError(str(err)) from err

            # Inspect HTTP status codes
            if response.status_code == 429 or 500 <= response.status_code < 600:
                retry_after_hdr = response.headers.get("Retry-After")
                retry_after: float | None = None
                if retry_after_hdr:
                    try:
                        retry_after = float(retry_after_hdr)
                    except ValueError:
                        pass

                if attempt < self.max_retries:
                    delay = self._calculate_backoff(attempt, retry_after)
                    logger.warning(
                        "HTTP %d from %s (attempt %d/%d). Backing off %.2fs",
                        response.status_code,
                        endpoint_url,
                        attempt + 1,
                        self.max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                raise MediaWikiHTTPError(
                    response.status_code,
                    (
                        f"Exhausted retries ({self.max_retries}) for {endpoint_url}: "
                        f"{response.text[:200]}"
                    ),
                )

            if response.status_code != 200:
                raise MediaWikiHTTPError(
                    response.status_code,
                    (
                        f"Unexpected status code {response.status_code}: "
                        f"{response.text[:200]}"
                    ),
                )

            # Response is HTTP 200. Parse JSON and inspect for API errors.
            try:
                data: dict[str, Any] = response.json()
            except Exception as err:
                raise MediaWikiAPIError(
                    code="invalidjson",
                    info=f"Failed to decode response as JSON: {response.text[:200]}",
                ) from err

            # Check single "error" payload
            if "error" in data and isinstance(data["error"], dict):
                error_obj = data["error"]
                code = str(error_obj.get("code", "unknown"))
                info = str(error_obj.get("info", "Unknown API error"))

                if code == "maxlag":
                    lag_val = error_obj.get("lag")
                    lag_float = float(lag_val) if lag_val is not None else None
                    retry_hdr = response.headers.get("Retry-After")
                    retry_after_float: float | None = None
                    if retry_hdr:
                        try:
                            retry_after_float = float(retry_hdr)
                        except ValueError:
                            pass
                    delay = self._calculate_backoff(
                        attempt,
                        retry_after_float or lag_float,
                    )

                    if attempt < self.max_retries:
                        logger.warning(
                            "MediaWiki maxlag on %s (lag: %s). Backing off %.2fs",
                            endpoint_url,
                            lag_val,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    raise MediaWikiMaxlagError(
                        info=info,
                        lag=lag_float,
                        retry_after=retry_after_float,
                        details=error_obj,
                    )

                # Non-maxlag API error treated as failure
                raise MediaWikiAPIError(code=code, info=info, details=error_obj)

            # Check multiple "errors" payload
            if "errors" in data and isinstance(data["errors"], list) and data["errors"]:
                first_err = data["errors"][0]
                code = str(first_err.get("code", "unknown"))
                info = str(
                    first_err.get("text", first_err.get("info", "Unknown API error"))
                )
                raise MediaWikiAPIError(
                    code=code,
                    info=info,
                    details={"errors": data["errors"]},
                )

            return data

        raise MediaWikiNetworkError(
            f"Exhausted retries ({self.max_retries}) querying {endpoint_url}"
        )

    async def fetch_parsed_content(
        self,
        endpoint_url: str,
        page_id: int | None = None,
        title: str | None = None,
    ) -> str:
        """Fetch parsed page HTML/wikitext when extracts extension is unavailable."""
        params: dict[str, Any] = {
            "action": "parse",
            "prop": "text",
        }
        if page_id is not None:
            params["pageid"] = page_id
        elif title is not None:
            params["page"] = title
        else:
            return ""

        try:
            data = await self.request(endpoint_url, params)
            parse_obj = data.get("parse", {})
            text_val = parse_obj.get("text", "")
            if isinstance(text_val, dict):
                return str(text_val.get("*", ""))
            return str(text_val)
        except Exception as err:
            logger.debug(
                "Failed to fetch parsed content for page_id=%s, title=%s: %s",
                page_id,
                title,
                err,
            )
            return ""

    def _extract_page_models(
        self,
        pages_raw: Any,
    ) -> list[dict[str, Any]]:
        """Normalize pages list whether returned as list (formatversion=2) or dict."""
        if isinstance(pages_raw, list):
            return [p for p in pages_raw if isinstance(p, dict)]
        if isinstance(pages_raw, dict):
            return [p for p in pages_raw.values() if isinstance(p, dict)]
        return []

    async def _build_page_model(
        self,
        endpoint_url: str,
        page_dict: dict[str, Any],
        wiki_license: str,
    ) -> MediaWikiPage:
        """Convert a raw MediaWiki page JSON payload into a structured MediaWikiPage."""
        title = str(page_dict.get("title", ""))
        page_id = int(page_dict.get("pageid", 0))

        # 1. Text: extracts where available, otherwise parsed content
        content = page_dict.get("extract", "")
        if not content:
            content = await self.fetch_parsed_content(
                endpoint_url,
                page_id=page_id if page_id > 0 else None,
                title=title if not page_id else None,
            )

        # 2. Source URL
        source_url = page_dict.get("canonicalurl") or page_dict.get("fullurl")
        if not source_url:
            base_site = endpoint_url.rsplit("/api.php", 1)[0]
            quoted_title = urllib.parse.quote(title.replace(" ", "_"))
            source_url = f"{base_site}/wiki/{quoted_title}"

        # 3. Revision ID
        revisions = page_dict.get("revisions")
        revision_id: str | None = None
        if isinstance(revisions, list) and revisions:
            first_rev = revisions[0]
            if isinstance(first_rev, dict) and "revid" in first_rev:
                revision_id = str(first_rev["revid"])

        # 4. Categories
        categories_raw = page_dict.get("categories", [])
        categories: list[str] = []
        if isinstance(categories_raw, list):
            for cat in categories_raw:
                if isinstance(cat, dict) and "title" in cat:
                    categories.append(str(cat["title"]))

        # 5. Redirects
        redirects_raw = page_dict.get("redirects", [])
        redirects: list[str] = []
        if isinstance(redirects_raw, list):
            for r in redirects_raw:
                if isinstance(r, dict) and "title" in r:
                    redirects.append(str(r["title"]))

        # 6. Links
        links_raw = page_dict.get("links", [])
        links: list[str] = []
        if isinstance(links_raw, list):
            for link_item in links_raw:
                if isinstance(link_item, dict) and "title" in link_item:
                    links.append(str(link_item["title"]))

        return MediaWikiPage(
            title=title,
            page_id=page_id,
            source_url=str(source_url),
            content=str(content),
            revision_id=revision_id,
            retrieval_timestamp=datetime.now(UTC),
            license=wiki_license,
            categories=categories,
            redirects=redirects,
            links=links,
        )

    async def stream_pages(
        self,
        endpoint_url: str,
        wiki_license: str,
        generator: str = "allpages",
        namespace: int = 0,
        batch_size: int = 50,
        limit: int | None = None,
    ) -> AsyncIterator[MediaWikiPage]:
        """Enumerate pages using generator queries and follow continue tokens [S12]."""
        base_params: dict[str, Any] = {
            "action": "query",
            "generator": generator,
            "prop": "extracts|categories|links|redirects|info|revisions",
            "explaintext": "1",
            "exintro": "0",
            "inprop": "url",
            "rvprop": "ids|timestamp",
            "cllimit": "max",
            "pllimit": "max",
            "rdlimit": "max",
        }

        # Generator parameters (allpages uses gap*, etc.)
        if generator == "allpages":
            base_params["gaplimit"] = batch_size
            base_params["gapnamespace"] = namespace
        else:
            base_params["glimit"] = batch_size
            base_params["gnamespace"] = namespace

        continue_params: dict[str, Any] = {}
        yielded_count = 0

        while True:
            params = {**base_params, **continue_params}
            data = await self.request(endpoint_url, params)
            query_obj = data.get("query", {})
            pages_raw = query_obj.get("pages", [])
            page_dicts = self._extract_page_models(pages_raw)

            for p in page_dicts:
                if p.get("missing", False):
                    continue

                page = await self._build_page_model(endpoint_url, p, wiki_license)
                yield page
                yielded_count += 1
                if limit is not None and yielded_count >= limit:
                    return

            # Follow continue tokens
            if "continue" in data and isinstance(data["continue"], dict):
                continue_params = data["continue"]
            else:
                break

    async def fetch_page(
        self,
        endpoint_url: str,
        title: str,
        wiki_license: str,
    ) -> MediaWikiPage:
        """Fetch single page by title preserving metadata and content."""
        params: dict[str, Any] = {
            "action": "query",
            "titles": title,
            "prop": "extracts|categories|links|redirects|info|revisions",
            "explaintext": "1",
            "exintro": "0",
            "inprop": "url",
            "rvprop": "ids|timestamp",
            "cllimit": "max",
            "pllimit": "max",
            "rdlimit": "max",
        }

        data = await self.request(endpoint_url, params)
        query_obj = data.get("query", {})
        pages_raw = query_obj.get("pages", [])
        page_dicts = self._extract_page_models(pages_raw)

        if not page_dicts or page_dicts[0].get("missing", False):
            raise MediaWikiAPIError(
                code="missingtitle",
                info=f"Page '{title}' not found on {endpoint_url}",
            )

        return await self._build_page_model(endpoint_url, page_dicts[0], wiki_license)
