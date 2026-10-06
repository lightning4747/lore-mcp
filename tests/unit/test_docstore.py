"""Unit tests for FilesystemDocumentStore content-addressing and offline rebuild."""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lore_mcp.adapters.docstore import (
    DocumentCorruptedError,
    FilesystemDocumentStore,
    compute_sha256,
)
from lore_mcp.domain.documents import DocumentSection, NormalizedDocument, RawDocument
from lore_mcp.domain.types import SeriesId


@pytest.mark.unit
@pytest.mark.asyncio
async def test_content_addressed_storage(tmp_path: Path) -> None:
    store = FilesystemDocumentStore(base_dir=tmp_path)
    content = "Loki variant timeline data."
    expected_hash = compute_sha256(content)

    doc = RawDocument(
        series_id=SeriesId("loki"),
        source_url="https://marvel.fandom.com/wiki/Loki",
        source_type="fandom",
        content=content,
        retrieval_timestamp=datetime.now(UTC),
    )
    await store.put_raw(doc)

    # Verify content-addressed file location
    expected_obj_path = tmp_path / "loki" / "raw" / "objects" / f"{expected_hash}.json"
    assert expected_obj_path.is_file()

    # Re-fetch via URL
    fetched = await store.get_raw(
        SeriesId("loki"), "https://marvel.fandom.com/wiki/Loki"
    )
    assert fetched is not None
    assert fetched.content == content
    assert fetched.source_url == "https://marvel.fandom.com/wiki/Loki"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deduplication_on_identical_content(tmp_path: Path) -> None:
    store = FilesystemDocumentStore(base_dir=tmp_path)
    content = "Identical content across two different URLs."

    doc1 = RawDocument(
        series_id=SeriesId("loki"),
        source_url="https://marvel.fandom.com/wiki/PageA",
        source_type="fandom",
        content=content,
        retrieval_timestamp=datetime.now(UTC),
    )
    doc2 = RawDocument(
        series_id=SeriesId("loki"),
        source_url="https://marvel.fandom.com/wiki/PageB",
        source_type="fandom",
        content=content,
        retrieval_timestamp=datetime.now(UTC),
    )
    await store.put_raw(doc1)
    await store.put_raw(doc2)

    # Only one object file is stored
    objs_dir = tmp_path / "loki" / "raw" / "objects"
    obj_files = list(objs_dir.glob("*.json"))
    assert len(obj_files) == 1

    # But both URLs resolve to it
    res1 = await store.get_raw(SeriesId("loki"), "https://marvel.fandom.com/wiki/PageA")
    res2 = await store.get_raw(SeriesId("loki"), "https://marvel.fandom.com/wiki/PageB")
    assert res1 is not None and res2 is not None
    assert res1.content == content
    assert res2.content == content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_series_iteration_and_listing(tmp_path: Path) -> None:
    store = FilesystemDocumentStore(base_dir=tmp_path)

    doc_loki = RawDocument(
        series_id=SeriesId("loki"),
        source_url="https://marvel.fandom.com/wiki/Loki",
        source_type="fandom",
        content="Loki content",
        retrieval_timestamp=datetime.now(UTC),
    )
    doc_andalor = RawDocument(
        series_id=SeriesId("andalor"),
        source_url="https://starwars.fandom.com/wiki/Cassian_Andor",
        source_type="fandom",
        content="Andor content",
        retrieval_timestamp=datetime.now(UTC),
    )
    await store.put_raw(doc_loki)
    await store.put_raw(doc_andalor)

    # 1. list_series
    all_series = await store.list_series()
    assert all_series == ["andalor", "loki"]

    # 2. iter_raw for loki
    loki_docs: list[RawDocument] = []
    async for d in store.iter_raw(SeriesId("loki")):
        loki_docs.append(d)
    assert len(loki_docs) == 1
    assert loki_docs[0].content == "Loki content"

    # 3. iter_normalized
    norm_doc = NormalizedDocument(
        series_id=SeriesId("loki"),
        source_url="https://marvel.fandom.com/wiki/Loki",
        source_type="fandom",
        title="Loki",
        text="Normalized Loki",
        sections=[DocumentSection(heading="Origin", content="Frost Giant")],
        content_hash="hash_norm_1",
        retrieval_timestamp=datetime.now(UTC),
    )
    await store.put_normalized(norm_doc)

    norm_docs: list[NormalizedDocument] = []
    async for nd in store.iter_normalized(SeriesId("loki")):
        norm_docs.append(nd)
    assert len(norm_docs) == 1
    assert norm_docs[0].title == "Loki"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_offline_rebuild_from_objects_without_network(tmp_path: Path) -> None:
    store = FilesystemDocumentStore(base_dir=tmp_path)

    raw_doc = RawDocument(
        series_id=SeriesId("loki"),
        source_url="https://marvel.fandom.com/wiki/Loki",
        source_type="fandom",
        content="Offline content for cluster rebuild",
        retrieval_timestamp=datetime.now(UTC),
    )
    await store.put_raw(raw_doc)

    norm_doc = NormalizedDocument(
        series_id=SeriesId("loki"),
        source_url="https://marvel.fandom.com/wiki/Loki",
        source_type="fandom",
        title="Loki Page",
        text="Normalized text",
        sections=[],
        content_hash="h_loki",
        retrieval_timestamp=datetime.now(UTC),
    )
    await store.put_normalized(norm_doc)

    # Wipe the by_url index directories completely to simulate index loss
    shutil.rmtree(tmp_path / "loki" / "raw" / "by_url")
    shutil.rmtree(tmp_path / "loki" / "normalized" / "by_url")

    # Verification: lookups fail because URL pointers were deleted
    loki_url = "https://marvel.fandom.com/wiki/Loki"
    assert await store.get_raw(SeriesId("loki"), loki_url) is None
    assert await store.get_normalized(SeriesId("loki"), loki_url) is None

    # Rebuild indexes 100% offline from the content-addressed objects
    rebuilt_count = await store.rebuild_url_index(SeriesId("loki"))
    assert rebuilt_count == 2

    # Verify both documents are reachable again
    restored_raw = await store.get_raw(
        SeriesId("loki"), "https://marvel.fandom.com/wiki/Loki"
    )
    assert restored_raw is not None
    assert restored_raw.content == "Offline content for cluster rebuild"

    restored_norm = await store.get_normalized(
        SeriesId("loki"), "https://marvel.fandom.com/wiki/Loki"
    )
    assert restored_norm is not None
    assert restored_norm.title == "Loki Page"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_corruption_detection(tmp_path: Path) -> None:
    store = FilesystemDocumentStore(base_dir=tmp_path)
    doc = RawDocument(
        series_id=SeriesId("loki"),
        source_url="https://marvel.fandom.com/wiki/Loki",
        source_type="fandom",
        content="Tampered content",
        retrieval_timestamp=datetime.now(UTC),
    )
    await store.put_raw(doc)

    # Tamper with file content
    h = compute_sha256("Tampered content")
    obj_file = tmp_path / "loki" / "raw" / "objects" / f"{h}.json"
    tampered_doc = RawDocument(
        series_id=SeriesId("loki"),
        source_url="https://marvel.fandom.com/wiki/Loki",
        source_type="fandom",
        content="Changed content that doesn't match filename hash",
        retrieval_timestamp=datetime.now(UTC),
    )
    obj_file.write_text(tampered_doc.model_dump_json(), encoding="utf-8")

    with pytest.raises(DocumentCorruptedError) as exc_info:
        async for _ in store.iter_raw(SeriesId("loki")):
            pass

    assert "Document corrupted" in str(exc_info.value)
