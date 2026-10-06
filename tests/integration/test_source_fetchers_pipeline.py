"""Integration tests for source fetching pipeline and offline disaster recovery."""

import shutil
import urllib.parse
from pathlib import Path

import httpx
import pytest

from lore_mcp.adapters.docstore import FilesystemDocumentStore
from lore_mcp.adapters.mediawiki import MediaWikiClient, MediaWikiDocumentSource
from lore_mcp.adapters.srt import SrtTranscriptSource
from lore_mcp.domain.series_config import WikiEndpointConfig
from lore_mcp.domain.types import SeriesId

TEST_ENDPOINT = "https://marvelcinematicuniverse.fandom.com/api.php"
TEST_LICENSE = "CC BY-SA 3.0"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_source_fetching_and_offline_rebuild_pipeline(
    tmp_path: Path,
) -> None:
    # 1. Setup Mock MediaWiki with pagination and metadata
    page1 = {
        "pageid": 101,
        "title": "Loki",
        "extract": "Loki is the God of Mischief.",
        "canonicalurl": "https://marvelcinematicuniverse.fandom.com/wiki/Loki",
        "revisions": [{"revid": 1001, "timestamp": "2024-01-01T00:00:00Z"}],
        "categories": [
            {"title": "Category:Asgardians"},
            {"title": "Category:Characters"},
        ],
        "redirects": [{"title": "Loki Laufeyson"}],
        "links": [{"title": "Thor"}, {"title": "Sylvie"}],
    }
    page2 = {
        "pageid": 102,
        "title": "Sylvie",
        "extract": "Sylvie is a Loki variant.",
        "canonicalurl": "https://marvelcinematicuniverse.fandom.com/wiki/Sylvie",
        "revisions": [{"revid": 1002, "timestamp": "2024-01-01T00:00:00Z"}],
        "categories": [{"title": "Category:Characters"}],
        "redirects": [],
        "links": [{"title": "Loki"}],
    }

    def mw_handler(request: httpx.Request) -> httpx.Response:
        query_params = urllib.parse.parse_qs(request.url.query.decode("utf-8"))
        if "generator" in query_params:
            if "gapcontinue" not in query_params:
                return httpx.Response(
                    status_code=200,
                    json={
                        "continue": {
                            "gapcontinue": "Sylvie",
                            "continue": "gapcontinue||",
                        },
                        "query": {"pages": [page1]},
                    },
                )
            return httpx.Response(
                status_code=200,
                json={"batchcomplete": True, "query": {"pages": [page2]}},
            )
        titles = query_params.get("titles", [""])[0]
        if titles == "Loki":
            return httpx.Response(status_code=200, json={"query": {"pages": [page1]}})
        if titles == "Sylvie":
            return httpx.Response(status_code=200, json={"query": {"pages": [page2]}})
        return httpx.Response(
            status_code=200,
            json={"query": {"pages": [{"title": titles, "missing": True}]}},
        )

    transport = httpx.MockTransport(mw_handler)
    http_client = httpx.AsyncClient(transport=transport)
    mw_client = MediaWikiClient(http_client=http_client, min_request_interval=0.0)
    mw_source = MediaWikiDocumentSource(client=mw_client)
    mw_source.register_endpoints(
        SeriesId("loki"),
        [
            WikiEndpointConfig(
                api_url=TEST_ENDPOINT,
                wiki_type="fandom",
                license=TEST_LICENSE,
            )
        ],
    )

    # 2. Setup Local SRT Transcripts
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    srt_file = transcripts_dir / "Loki.S01E06.srt"
    srt_file.write_text(
        """1
00:00:01,000 --> 00:00:03,500
<i>HE WHO REMAINS:</i> Welcome to the Citadel at the End of Time.

2
00:00:04,000 --> 00:00:06,000
LOKI: Who are you?
""",
        encoding="utf-8",
    )
    srt_source = SrtTranscriptSource()
    srt_source.register_transcripts_dir(SeriesId("loki"), transcripts_dir)

    # 3. Setup Filesystem DocumentStore
    store_dir = tmp_path / "store"
    doc_store = FilesystemDocumentStore(base_dir=store_dir)

    # 4. Ingest MediaWiki documents into DocumentStore
    mw_docs = await mw_source.fetch_all_documents(SeriesId("loki"))
    assert len(mw_docs) == 2
    for doc in mw_docs:
        await doc_store.put_raw(doc)

    # 5. Ingest SRT transcript into DocumentStore
    srt_locators = await srt_source.list_locators(SeriesId("loki"))
    assert srt_locators == ["Loki.S01E06.srt"]
    srt_doc = await srt_source.fetch_document(SeriesId("loki"), srt_locators[0])
    assert srt_doc.metadata["season"] == "1"
    assert srt_doc.metadata["episode"] == "6"
    await doc_store.put_raw(srt_doc)

    # 6. Verify Store contents and series iteration
    stored_urls = await doc_store.list_urls(SeriesId("loki"))
    assert len(stored_urls) == 3
    assert "https://marvelcinematicuniverse.fandom.com/wiki/Loki" in stored_urls
    assert "https://marvelcinematicuniverse.fandom.com/wiki/Sylvie" in stored_urls
    assert f"file://{srt_file}" in stored_urls

    stored_docs = [d async for d in doc_store.iter_raw(SeriesId("loki"))]
    assert len(stored_docs) == 3

    # 7. Simulate disaster recovery: delete URL index directories
    shutil.rmtree(store_dir / "loki" / "raw" / "by_url")
    assert await doc_store.get_raw(SeriesId("loki"), mw_docs[0].source_url) is None

    # 8. Rebuild index offline from content-addressed objects without network calls
    rebuilt = await doc_store.rebuild_url_index(SeriesId("loki"))
    assert rebuilt == 3

    # 9. Verify documents are fully restored after offline rebuild
    restored_loki = await doc_store.get_raw(
        SeriesId("loki"), "https://marvelcinematicuniverse.fandom.com/wiki/Loki"
    )
    assert restored_loki is not None
    assert restored_loki.content == "Loki is the God of Mischief."
    assert restored_loki.metadata["license"] == TEST_LICENSE
    assert restored_loki.metadata["revision_id"] == "1001"

    restored_srt = await doc_store.get_raw(SeriesId("loki"), f"file://{srt_file}")
    assert restored_srt is not None
    assert "Citadel at the End of Time" in restored_srt.content
    assert restored_srt.metadata["season"] == "1"
    assert restored_srt.metadata["episode"] == "6"
