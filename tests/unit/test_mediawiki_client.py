"""Unit tests for MediaWiki Action API client and DocumentSource adapter."""

import asyncio
import json
import urllib.parse
from datetime import UTC, datetime

import httpx
import pytest

from lore_mcp.adapters.mediawiki import (
    DEFAULT_USER_AGENT,
    MediaWikiAPIError,
    MediaWikiClient,
    MediaWikiDocumentSource,
    MediaWikiHTTPError,
    MediaWikiMaxlagError,
    MediaWikiNetworkError,
    MediaWikiPage,
)
from lore_mcp.application.errors import DataUnavailable
from lore_mcp.domain.series_config import WikiEndpointConfig
from lore_mcp.domain.types import SeriesId

TEST_ENDPOINT = "https://marvelcinematicuniverse.fandom.com/api.php"
TEST_LICENSE = "CC BY-SA 3.0"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_user_agent_maxlag_and_format_sent_by_default() -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            status_code=200,
            json={"batchcomplete": True, "query": {"pages": []}},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = MediaWikiClient(
            http_client=http_client,
            maxlag=5,
            min_request_interval=0.0,
        )
        await client.request(TEST_ENDPOINT, {"action": "query"})

    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req.headers["User-Agent"] == DEFAULT_USER_AGENT

    query_params = urllib.parse.parse_qs(req.url.query.decode("utf-8"))
    assert query_params["maxlag"] == ["5"]
    assert query_params["format"] == ["json"]
    assert query_params["formatversion"] == ["2"]
    assert query_params["action"] == ["query"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_request_throttling() -> None:
    call_times: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        call_times.append(asyncio.get_event_loop().time())
        return httpx.Response(
            status_code=200,
            json={"batchcomplete": True, "query": {"pages": []}},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = MediaWikiClient(
            http_client=http_client,
            min_request_interval=0.05,
        )
        await client.request(TEST_ENDPOINT, {"action": "query"})
        await client.request(TEST_ENDPOINT, {"action": "query"})

    assert len(call_times) == 2
    elapsed = call_times[1] - call_times[0]
    assert elapsed >= 0.045


@pytest.mark.unit
@pytest.mark.asyncio
async def test_http_200_with_api_error_payload_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "error": {
                    "code": "badvalue",
                    "info": "Unrecognized value for parameter 'action': invalid.",
                }
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = MediaWikiClient(
            http_client=http_client,
            min_request_interval=0.0,
        )
        with pytest.raises(MediaWikiAPIError) as exc_info:
            await client.request(TEST_ENDPOINT, {"action": "invalid"})

    assert exc_info.value.code == "badvalue"
    assert "Unrecognized value" in exc_info.value.info


@pytest.mark.unit
@pytest.mark.asyncio
async def test_http_200_with_errors_array_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "errors": [
                    {"code": "permissiondenied", "text": "Permission denied by wiki."}
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = MediaWikiClient(
            http_client=http_client,
            min_request_interval=0.0,
        )
        with pytest.raises(MediaWikiAPIError) as exc_info:
            await client.request(TEST_ENDPOINT, {"action": "query"})

    assert exc_info.value.code == "permissiondenied"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_maxlag_retry_and_backoff_success() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                status_code=200,
                headers={"Retry-After": "0.01"},
                json={
                    "error": {
                        "code": "maxlag",
                        "info": "Waiting for replica: 6.2 seconds lagged.",
                        "lag": 6.2,
                    }
                },
            )
        return httpx.Response(
            status_code=200,
            json={"batchcomplete": True, "query": {"pages": [{"title": "Loki"}]}},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = MediaWikiClient(
            http_client=http_client,
            max_retries=2,
            base_delay=0.01,
            min_request_interval=0.0,
        )
        data = await client.request(TEST_ENDPOINT, {"action": "query"})

    assert attempts == 2
    assert data["query"]["pages"][0]["title"] == "Loki"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_maxlag_exhaustion_raises_maxlag_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "error": {
                    "code": "maxlag",
                    "info": "Waiting for replica: 10 seconds lagged.",
                    "lag": 10.0,
                }
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = MediaWikiClient(
            http_client=http_client,
            max_retries=2,
            base_delay=0.01,
            min_request_interval=0.0,
        )
        with pytest.raises(MediaWikiMaxlagError) as exc_info:
            await client.request(TEST_ENDPOINT, {"action": "query"})

    assert exc_info.value.code == "maxlag"
    assert exc_info.value.lag == 10.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_http_500_retries_and_recovers() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                status_code=503,
                headers={"Retry-After": "0.01"},
                content="Service Unavailable",
            )
        return httpx.Response(
            status_code=200,
            json={"batchcomplete": True, "query": {"pages": []}},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = MediaWikiClient(
            http_client=http_client,
            max_retries=2,
            base_delay=0.01,
            min_request_interval=0.0,
        )
        data = await client.request(TEST_ENDPOINT, {"action": "query"})

    assert attempts == 2
    assert "query" in data


@pytest.mark.unit
@pytest.mark.asyncio
async def test_http_500_exhaustion_raises_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=500, content="Internal Error")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = MediaWikiClient(
            http_client=http_client,
            max_retries=1,
            base_delay=0.01,
            min_request_interval=0.0,
        )
        with pytest.raises(MediaWikiHTTPError) as exc_info:
            await client.request(TEST_ENDPOINT, {"action": "query"})

    assert exc_info.value.status_code == 500


@pytest.mark.unit
@pytest.mark.asyncio
async def test_network_timeout_retries_and_exhaustion() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("Connection timed out")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = MediaWikiClient(
            http_client=http_client,
            max_retries=1,
            base_delay=0.01,
            min_request_interval=0.0,
        )
        with pytest.raises(MediaWikiNetworkError) as exc_info:
            await client.request(TEST_ENDPOINT, {"action": "query"})

    assert "Connection timed out" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generator_pagination_and_continue_tokens() -> None:
    page1 = {
        "pageid": 101,
        "title": "Loki",
        "extract": "Loki is the God of Mischief.",
        "canonicalurl": "https://marvelcinematicuniverse.fandom.com/wiki/Loki",
        "revisions": [{"revid": 1001, "timestamp": "2024-01-01T00:00:00Z"}],
        "categories": [{"ns": 14, "title": "Category:Characters"}],
        "redirects": [{"title": "Loki Laufeyson"}],
        "links": [{"title": "Thor"}],
    }
    page2 = {
        "pageid": 102,
        "title": "Mobius M. Mobius",
        "extract": "Mobius is a TVA analyst.",
        "canonicalurl": (
            "https://marvelcinematicuniverse.fandom.com/wiki/Mobius_M._Mobius"
        ),
        "revisions": [{"revid": 1002, "timestamp": "2024-01-01T00:00:00Z"}],
        "categories": [{"ns": 14, "title": "Category:TVA"}],
        "redirects": [],
        "links": [{"title": "Loki"}],
    }

    requests_made: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_made.append(request)
        query_params = urllib.parse.parse_qs(request.url.query.decode("utf-8"))
        if "gapcontinue" not in query_params:
            return httpx.Response(
                status_code=200,
                json={
                    "continue": {
                        "gapcontinue": "Mobius M. Mobius",
                        "continue": "gapcontinue||",
                    },
                    "query": {"pages": [page1]},
                },
            )
        return httpx.Response(
            status_code=200,
            json={
                "batchcomplete": True,
                "query": {"pages": [page2]},
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = MediaWikiClient(
            http_client=http_client,
            min_request_interval=0.0,
        )
        pages: list[MediaWikiPage] = []
        async for page in client.stream_pages(
            TEST_ENDPOINT,
            wiki_license=TEST_LICENSE,
            batch_size=1,
        ):
            pages.append(page)

    assert len(requests_made) == 2
    assert len(pages) == 2

    # Verify first page
    p1 = pages[0]
    assert p1.title == "Loki"
    assert p1.page_id == 101
    assert p1.content == "Loki is the God of Mischief."
    assert p1.revision_id == "1001"
    assert p1.source_url == "https://marvelcinematicuniverse.fandom.com/wiki/Loki"
    assert p1.license == TEST_LICENSE
    assert p1.categories == ["Category:Characters"]
    assert p1.redirects == ["Loki Laufeyson"]
    assert p1.links == ["Thor"]

    # Verify second page
    p2 = pages[1]
    assert p2.title == "Mobius M. Mobius"
    assert p2.page_id == 102
    assert p2.content == "Mobius is a TVA analyst."
    assert p2.revision_id == "1002"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_missing_falls_back_to_parsed_content() -> None:
    page_without_extract = {
        "pageid": 200,
        "title": "TVA",
        "extract": "",  # Extracts extension missing or empty
        "canonicalurl": "https://marvelcinematicuniverse.fandom.com/wiki/TVA",
        "revisions": [{"revid": 2001}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        query_params = urllib.parse.parse_qs(request.url.query.decode("utf-8"))
        action = query_params.get("action", [""])[0]

        if action == "query":
            return httpx.Response(
                status_code=200,
                json={"query": {"pages": [page_without_extract]}},
            )
        if action == "parse":
            return httpx.Response(
                status_code=200,
                json={
                    "parse": {
                        "title": "TVA",
                        "pageid": 200,
                        "text": "<div>Time Variance Authority body content</div>",
                    }
                },
            )
        return httpx.Response(status_code=400, content="Bad request")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = MediaWikiClient(
            http_client=http_client,
            min_request_interval=0.0,
        )
        page = await client.fetch_page(
            TEST_ENDPOINT, title="TVA", wiki_license=TEST_LICENSE
        )

    assert page.title == "TVA"
    assert "Time Variance Authority body content" in page.content
    assert page.page_id == 200


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetch_page_missing_raises_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "query": {
                    "pages": [{"ns": 0, "title": "NonExistentPage", "missing": True}]
                }
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = MediaWikiClient(
            http_client=http_client,
            min_request_interval=0.0,
        )
        with pytest.raises(MediaWikiAPIError) as exc_info:
            await client.fetch_page(
                TEST_ENDPOINT,
                title="NonExistentPage",
                wiki_license=TEST_LICENSE,
            )

    assert exc_info.value.code == "missingtitle"


@pytest.mark.unit
def test_page_to_raw_document_preserves_provenance_and_metadata() -> None:
    now = datetime.now(UTC)
    page = MediaWikiPage(
        title="Sylvie",
        page_id=300,
        source_url="https://marvelcinematicuniverse.fandom.com/wiki/Sylvie",
        content="Sylvie Laufeydottir is a variant of Loki.",
        revision_id="3001",
        retrieval_timestamp=now,
        license="CC BY-SA 3.0",
        categories=["Category:Variants", "Category:Characters"],
        redirects=["Lady Loki"],
        links=["Loki", "Alioth"],
    )

    raw_doc = page.to_raw_document(
        series_id=SeriesId("loki"),
        source_type="fandom",
        extra_metadata={"custom_key": "custom_val"},
    )

    assert raw_doc.series_id == "loki"
    assert (
        raw_doc.source_url == "https://marvelcinematicuniverse.fandom.com/wiki/Sylvie"
    )
    assert raw_doc.source_type == "fandom"
    assert raw_doc.content == "Sylvie Laufeydottir is a variant of Loki."
    assert raw_doc.retrieval_timestamp == now

    # Verify metadata fields
    meta = raw_doc.metadata
    assert meta["title"] == "Sylvie"
    assert meta["page_id"] == "300"
    assert meta["revision_id"] == "3001"
    assert meta["license"] == "CC BY-SA 3.0"
    assert meta["custom_key"] == "custom_val"
    assert json.loads(meta["categories"]) == [
        "Category:Variants",
        "Category:Characters",
    ]
    assert json.loads(meta["redirects"]) == ["Lady Loki"]
    assert json.loads(meta["links"]) == ["Loki", "Alioth"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mediawiki_document_source_adapter() -> None:
    page_data = {
        "pageid": 401,
        "title": "He Who Remains",
        "extract": "Creator of the Time Variance Authority.",
        "canonicalurl": (
            "https://marvelcinematicuniverse.fandom.com/wiki/He_Who_Remains"
        ),
        "revisions": [{"revid": 4001, "timestamp": "2024-01-01T00:00:00Z"}],
        "categories": [{"title": "Category:Variants"}],
        "redirects": [],
        "links": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        query_params = urllib.parse.parse_qs(request.url.query.decode("utf-8"))
        if "generator" in query_params:
            return httpx.Response(
                status_code=200,
                json={"query": {"pages": [page_data]}},
            )
        titles = query_params.get("titles", [""])[0]
        if titles == "He Who Remains":
            return httpx.Response(
                status_code=200,
                json={"query": {"pages": [page_data]}},
            )
        return httpx.Response(
            status_code=200,
            json={"query": {"pages": [{"title": titles, "missing": True}]}},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = MediaWikiClient(
            http_client=http_client,
            min_request_interval=0.0,
        )
        adapter = MediaWikiDocumentSource(client=client)

        adapter.register_endpoints(
            SeriesId("loki"),
            [
                WikiEndpointConfig(
                    api_url=TEST_ENDPOINT,
                    wiki_type="fandom",
                    license=TEST_LICENSE,
                )
            ],
        )

        # 1. list_locators
        locators = await adapter.list_locators(SeriesId("loki"))
        assert locators == ["He Who Remains"]

        # 2. fetch_document
        raw_doc = await adapter.fetch_document(SeriesId("loki"), "He Who Remains")
        assert raw_doc.series_id == "loki"
        assert (
            raw_doc.source_url
            == "https://marvelcinematicuniverse.fandom.com/wiki/He_Who_Remains"
        )
        assert raw_doc.source_type == "fandom"
        assert raw_doc.metadata["license"] == TEST_LICENSE
        assert raw_doc.metadata["revision_id"] == "4001"

        # 3. fetch_document missing -> raises DataUnavailable
        with pytest.raises(DataUnavailable) as exc_info:
            await adapter.fetch_document(SeriesId("loki"), "MissingCharacter")
        assert exc_info.value.resource_id == "MissingCharacter"

        # 4. fetch_all_documents
        all_docs = await adapter.fetch_all_documents(SeriesId("loki"))
        assert len(all_docs) == 1
        assert all_docs[0].metadata["title"] == "He Who Remains"

        # 5. unknown series returns empty locators or raises on fetch
        empty_locs = await adapter.list_locators(SeriesId("unknown"))
        assert empty_locs == []
        with pytest.raises(DataUnavailable):
            await adapter.fetch_document(SeriesId("unknown"), "He Who Remains")
