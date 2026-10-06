"""Contract tests verifying MediaWiki and SRT adapters satisfy DocumentSource port."""

import urllib.parse
from pathlib import Path

import httpx
import pytest

from lore_mcp.adapters.mediawiki import MediaWikiClient, MediaWikiDocumentSource
from lore_mcp.adapters.srt import SrtTranscriptSource
from lore_mcp.domain.series_config import WikiEndpointConfig
from lore_mcp.domain.types import SeriesId
from lore_mcp.testing import DocumentSourceContractSuite

TEST_ENDPOINT = "https://marvelcinematicuniverse.fandom.com/api.php"


@pytest.mark.contract
class TestMediaWikiDocumentSourceContract(DocumentSourceContractSuite):
    @pytest.fixture
    def source(self) -> MediaWikiDocumentSource:
        page_payload = {
            "pageid": 100,
            "title": "Loki",
            "extract": "Loki is the God of Mischief.",
            "canonicalurl": "https://marvelcinematicuniverse.fandom.com/wiki/Loki",
            "revisions": [{"revid": 101, "timestamp": "2024-01-01T00:00:00Z"}],
            "categories": [{"title": "Category:Characters"}],
            "redirects": [],
            "links": [],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            query_params = urllib.parse.parse_qs(request.url.query.decode("utf-8"))
            if "generator" in query_params:
                return httpx.Response(
                    status_code=200,
                    json={"query": {"pages": [page_payload]}},
                )
            titles = query_params.get("titles", [""])[0]
            if titles == "Loki":
                return httpx.Response(
                    status_code=200,
                    json={"query": {"pages": [page_payload]}},
                )
            return httpx.Response(
                status_code=200,
                json={"query": {"pages": [{"title": titles, "missing": True}]}},
            )

        transport = httpx.MockTransport(handler)
        http_client = httpx.AsyncClient(transport=transport)
        mw_client = MediaWikiClient(http_client=http_client, min_request_interval=0.0)

        adapter = MediaWikiDocumentSource(client=mw_client)
        adapter.register_endpoints(
            SeriesId("loki"),
            [
                WikiEndpointConfig(
                    api_url=TEST_ENDPOINT,
                    wiki_type="fandom",
                    license="CC BY-SA 3.0",
                )
            ],
        )
        return adapter


@pytest.mark.contract
class TestSrtDocumentSourceContract(DocumentSourceContractSuite):
    @pytest.fixture
    def source(self, tmp_path: Path) -> SrtTranscriptSource:
        sub_file = tmp_path / "Loki.S01E01.srt"
        sub_file.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\n"
            "LOKI: I am burdened with glorious purpose.\n",
            encoding="utf-8",
        )
        adapter = SrtTranscriptSource()
        adapter.register_transcripts_dir(SeriesId("loki"), tmp_path)
        return adapter
