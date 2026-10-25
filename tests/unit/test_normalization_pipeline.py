"""Unit tests for the Normalization Pipeline coordinator (Phase 3)."""

from datetime import UTC, datetime

import pytest

from lore_mcp.application.normalization.pipeline import (
    DefaultNormalizationPipeline,
)
from lore_mcp.domain.documents import RawDocument
from lore_mcp.domain.types import SeriesId
from lore_mcp.ports.normalization import NormalizationPipeline
from lore_mcp.testing.fakes import FakeDocumentStore


def _create_raw_doc(
    url: str,
    content: str,
    source_type: str = "fandom",
    series_id: str = "loki",
    metadata: dict[str, str] | None = None,
) -> RawDocument:
    return RawDocument(
        series_id=SeriesId(series_id),
        source_url=url,
        source_type=source_type,
        content=content,
        retrieval_timestamp=datetime.now(UTC),
        metadata=metadata or {},
    )


@pytest.mark.unit
def test_pipeline_protocol_compliance() -> None:
    pipeline = DefaultNormalizationPipeline()
    assert isinstance(pipeline, NormalizationPipeline)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_normalize_single_document_routing() -> None:
    pipeline = DefaultNormalizationPipeline()

    # 1. Wiki Document
    wiki_raw = _create_raw_doc(
        url="https://marvel.fandom.com/wiki/Mobius",
        content="""
        <div class="navbox">Navigation noise</div>
        == Biography ==
        Mobius M. Mobius is an agent of the Time Variance Authority.
        """,
        source_type="fandom",
        metadata={"title": "Mobius M. Mobius"},
    )
    wiki_norm = await pipeline.normalize_document(wiki_raw)
    assert wiki_norm is not None
    assert wiki_norm.source_type == "fandom"
    assert "Navigation noise" not in wiki_norm.text
    assert "Mobius M. Mobius is an agent" in wiki_norm.text
    assert any(s.heading == "Biography" for s in wiki_norm.sections)

    # 2. Transcript Document
    transcript_raw = _create_raw_doc(
        url="file:///episodes/loki_s01e01.srt",
        content="""1
00:01:00,000 --> 00:01:03,000
<i>LOKI:</i> I am Loki of Asgard.

2
00:01:03,500 --> 00:01:05,000
[MUSIC PLAYING]
And I am burdened with glorious purpose.
""",
        source_type="transcript",
        metadata={"season": "1", "episode": "1"},
    )
    transcript_norm = await pipeline.normalize_document(transcript_raw)
    assert transcript_norm is not None
    assert transcript_norm.source_type == "transcript"
    assert "MUSIC PLAYING" not in transcript_norm.text
    assert "<i>" not in transcript_norm.text
    assert "Loki: I am Loki of Asgard." in transcript_norm.text
    assert "dialogue_segments" in transcript_norm.metadata

    # 3. Empty content returns None
    empty_raw = _create_raw_doc(
        url="https://marvel.fandom.com/wiki/Empty",
        content="   \n\t  ",
        source_type="fandom",
    )
    assert await pipeline.normalize_document(empty_raw) is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_batch_deduplication_and_persistence() -> None:
    store = FakeDocumentStore()
    pipeline = DefaultNormalizationPipeline(store=store)

    doc_target = _create_raw_doc(
        url="https://marvel.fandom.com/wiki/Sylvie",
        content="== Overview ==\nSylvie Laufeydottir is a variant of Loki.",
        source_type="fandom",
        metadata={"title": "Sylvie (Earth-TRN732)"},
    )
    doc_redirect = _create_raw_doc(
        url="https://marvel.fandom.com/wiki/Sylvie_Lushton",
        content="#REDIRECT [[Sylvie (Earth-TRN732)]]",
        source_type="fandom",
        metadata={"title": "Sylvie Lushton"},
    )
    doc_dup_1 = _create_raw_doc(
        url="https://marvel.fandom.com/wiki/TVA_Mirror1",
        content="The Time Variance Authority monitors timelines.",
        source_type="fandom",
        metadata={"title": "TVA Mirror 1"},
    )
    doc_dup_2 = _create_raw_doc(
        url="https://marvel.fandom.com/wiki/TVA_Mirror2",
        content="  the  time variance  authority monitors  timelines. \n",
        source_type="fandom",
        metadata={"title": "TVA Mirror 2"},
    )

    batch = [doc_target, doc_redirect, doc_dup_1, doc_dup_2]
    unique_docs, report = await pipeline.process_batch(batch)

    # 4 raw docs: 1 redirect collapsed, 1 exact duplicate dropped -> 2 unique docs
    assert len(unique_docs) == 2
    assert report.exact_duplicates_dropped == 1
    assert report.redirects_collapsed == 1

    target_norm = next(d for d in unique_docs if "Sylvie" in d.title)
    assert "Sylvie Lushton" in target_norm.metadata["aliases"]

    # Verify storage in FakeDocumentStore
    stored_target = await store.get_normalized(SeriesId("loki"), doc_target.source_url)
    assert stored_target is not None
    assert "Sylvie Lushton" in stored_target.metadata["aliases"]

    stored_dup1 = await store.get_normalized(SeriesId("loki"), doc_dup_1.source_url)
    assert stored_dup1 is not None

    stored_dup2 = await store.get_normalized(SeriesId("loki"), doc_dup_2.source_url)
    assert stored_dup2 is None

    stored_redirect = await store.get_normalized(
        SeriesId("loki"), doc_redirect.source_url
    )
    assert stored_redirect is None
