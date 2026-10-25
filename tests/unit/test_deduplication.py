"""Unit tests for content deduplication and redirect collapsing (Task 3.3 & 3.4)."""

import json
from datetime import UTC, datetime

import pytest

from lore_mcp.application.normalization.deduplicator import (
    ContentDeduplicator,
    compute_normalized_hash,
)
from lore_mcp.domain.documents import DocumentSection, NormalizedDocument
from lore_mcp.domain.types import SeriesId


def _create_norm_doc(
    title: str,
    text: str,
    source_url: str = "https://marvel.fandom.com/wiki/Test",
    source_type: str = "fandom",
    metadata: dict[str, str] | None = None,
) -> NormalizedDocument:
    meta = metadata or {}
    meta["title"] = title
    return NormalizedDocument(
        series_id=SeriesId("loki"),
        source_url=source_url,
        source_type=source_type,
        title=title,
        text=text,
        sections=[DocumentSection(heading=title, content=text)],
        content_hash=compute_normalized_hash(text),
        retrieval_timestamp=datetime.now(UTC),
        metadata=meta,
    )


@pytest.mark.unit
def test_deterministic_fingerprint() -> None:
    text_1 = "I am Loki of Asgard, burdened with glorious purpose."
    text_2 = "  i am loki of asgard,   burdened with glorious purpose.  \n"

    h1 = compute_normalized_hash(text_1)
    h2 = compute_normalized_hash(text_2)

    assert h1 == h2

    h3 = compute_normalized_hash("Different text altogether")
    assert h1 != h3


@pytest.mark.unit
def test_exact_duplicate_dropping() -> None:
    dedup = ContentDeduplicator()

    doc_a = _create_norm_doc("Loki Page 1", "Loki is the God of Mischief.")
    # Duplicate content with different title
    doc_b = _create_norm_doc("Loki Page 2", "Loki is the God of Mischief.")

    unique_docs, report = dedup.filter_duplicates([doc_a, doc_b])

    assert len(unique_docs) == 1
    assert unique_docs[0].title == "Loki Page 1"
    assert report.total_documents_processed == 2
    assert report.unique_documents_output == 1
    assert report.exact_duplicates_dropped == 1


@pytest.mark.unit
def test_redirect_collapsing() -> None:
    dedup = ContentDeduplicator()

    canonical_doc = _create_norm_doc(
        "Sylvie",
        "Sylvie is a female variant of Loki who bombed the Sacred Timeline.",
    )
    redirect_doc_1 = _create_norm_doc(
        "Lady Loki",
        "#REDIRECT [[Sylvie]]",
        source_url="https://marvel.fandom.com/wiki/Lady_Loki",
    )
    redirect_doc_2 = _create_norm_doc(
        "Sylvie Laufeydottir",
        "#REDIRECT [[Sylvie]]",
        source_url="https://marvel.fandom.com/wiki/Sylvie_Laufeydottir",
    )

    unique_docs, report = dedup.filter_duplicates(
        [canonical_doc, redirect_doc_1, redirect_doc_2]
    )

    assert len(unique_docs) == 1
    sylvie = unique_docs[0]
    assert sylvie.title == "Sylvie"

    aliases = json.loads(sylvie.metadata["aliases"])
    assert "Lady Loki" in aliases
    assert "Sylvie Laufeydottir" in aliases
    assert report.redirects_collapsed >= 2
    assert report.unique_documents_output == 1


@pytest.mark.unit
def test_repeated_transcript_detection() -> None:
    dedup = ContentDeduplicator()

    dialogue_text = (
        "Loki: I am burdened with glorious purpose.\n"
        "Mobius: Yeah, about that, we have paperwork.\n"
        "Hunter B-15: Reset the timeline."
    )

    meta_rel_1 = {"season": "1", "episode": "1", "source": "web-dl"}
    meta_rel_2 = {"season": "1", "episode": "1", "source": "bluray"}

    doc_rel_1 = _create_norm_doc(
        "Episode 1 Dialogue (WEB)",
        dialogue_text,
        source_url="file://s01e01_web.srt",
        source_type="transcript",
        metadata=meta_rel_1,
    )
    doc_rel_2 = _create_norm_doc(
        "Episode 1 Dialogue (BluRay)",
        dialogue_text + "\n",  # identical dialog
        source_url="file://s01e01_bluray.srt",
        source_type="transcript",
        metadata=meta_rel_2,
    )

    unique_docs, report = dedup.filter_duplicates([doc_rel_1, doc_rel_2])

    assert len(unique_docs) == 1
    assert unique_docs[0].source_url == "file://s01e01_web.srt"
    assert report.exact_duplicates_dropped == 1
