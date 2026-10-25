"""Unit tests for source document normalization (Task 3.1 & 3.4)."""

import unicodedata
from datetime import UTC, datetime

import pytest

from lore_mcp.application.normalization.document_normalizer import (
    DefaultDocumentNormalizer,
    compute_sha256,
)
from lore_mcp.domain.documents import RawDocument
from lore_mcp.domain.types import SeriesId


def _create_raw_doc(
    content: str, metadata: dict[str, str] | None = None
) -> RawDocument:
    meta = metadata or {"title": "Loki (Character)", "license": "CC-BY-SA 3.0"}
    return RawDocument(
        series_id=SeriesId("loki"),
        source_url="https://marvel.fandom.com/wiki/Loki",
        source_type="fandom",
        content=content,
        retrieval_timestamp=datetime.now(UTC),
        metadata=meta,
    )


@pytest.mark.unit
def test_html_noise_removal() -> None:
    normalizer = DefaultDocumentNormalizer()
    raw_content = """
    <html>
      <head>
        <script>alert("evil");</script>
        <style>body { color: red; }</style>
      </head>
      <body>
        <nav>Navigation menu</nav>
        <div class="navbox">Franchise Navbox</div>
        <div class="toc">Table of Contents</div>
        <div class="mw-jump-link"><a href="#content">Jump to content</a></div>
        <h1>Loki Overview</h1>
        <p>Loki is the God of Mischief.<span class="mw-editsection">[edit]</span></p>
        <div class="reflist"><p>References list</p></div>
      </body>
    </html>
    """
    doc = _create_raw_doc(raw_content)
    norm = normalizer.normalize(doc)

    assert "Navigation menu" not in norm.text
    assert "Franchise Navbox" not in norm.text
    assert "Table of Contents" not in norm.text
    assert "alert" not in norm.text
    assert "[edit]" not in norm.text
    assert "Jump to content" not in norm.text
    assert "Loki is the God of Mischief." in norm.text
    assert "# Loki Overview" in norm.text


@pytest.mark.unit
def test_infobox_stripping() -> None:
    normalizer = DefaultDocumentNormalizer()
    raw_content = """
    <div class="portable-infobox">
      <div class="pi-title">Loki Laufeyson</div>
      <div>Species: Frost Giant</div>
    </div>
    <p>Loki was adopted by Odin.</p>
    """
    doc = _create_raw_doc(raw_content)
    norm = normalizer.normalize(doc)

    assert "portable-infobox" not in norm.text
    assert "Species: Frost Giant" not in norm.text
    assert "Loki was adopted by Odin." in norm.text


@pytest.mark.unit
def test_wikitext_and_ref_removal() -> None:
    normalizer = DefaultDocumentNormalizer()
    raw_content = """
    {{CharacterInfobox|name=Sylvie}}
    <!-- Secret comment -->
    Sylvie is a [[Variant (TVA)|Variant]] of Loki<ref>Episode 2</ref>.
    She escaped the [[TVA]] using a [[TemPad]].
    {{Navbox Marvel}}
    """
    doc = _create_raw_doc(raw_content)
    norm = normalizer.normalize(doc)

    assert "CharacterInfobox" not in norm.text
    assert "Secret comment" not in norm.text
    assert "<ref>" not in norm.text
    assert "Episode 2" not in norm.text
    assert "Navbox Marvel" not in norm.text
    assert "Sylvie is a Variant of Loki." in norm.text
    assert "She escaped the TVA using a TemPad." in norm.text


@pytest.mark.unit
def test_unicode_nfc_normalization() -> None:
    normalizer = DefaultDocumentNormalizer()
    # Decomposed: 'e' + combining acute accent (\u0301)
    decomposed = "Sylvie Laufeydo\u0301ttir"
    assert unicodedata.is_normalized("NFD", decomposed)

    doc = _create_raw_doc(f"<p>{decomposed}</p>")
    norm = normalizer.normalize(doc)

    assert unicodedata.is_normalized("NFC", norm.text)
    assert norm.text == "Sylvie Laufeydóttir"


@pytest.mark.unit
def test_whitespace_and_paragraph_preservation() -> None:
    normalizer = DefaultDocumentNormalizer()
    raw_content = (
        "Paragraph 1 with \t extra   spaces and non-breaking\u00a0space.\n\n\n\n\n"
        "Paragraph 2 after excessive line breaks.\r\n\r\n"
        "Paragraph 3 with trailing spaces.   "
    )
    doc = _create_raw_doc(raw_content)
    norm = normalizer.normalize(doc)

    assert "Paragraph 1 with extra spaces and non-breaking space." in norm.text
    assert "Paragraph 2 after excessive line breaks." in norm.text
    assert "Paragraph 3 with trailing spaces." in norm.text

    # Paragraphs separated by \n\n
    paragraphs = norm.text.split("\n\n")
    assert len(paragraphs) == 3


@pytest.mark.unit
def test_heading_section_preservation() -> None:
    normalizer = DefaultDocumentNormalizer()
    raw_content = """
    Loki is a prince of Asgard.

    == Early Life ==
    Raised in Asgard alongside Thor.

    === The Frost Giant Revelation ===
    Discovered his true heritage in Jotunheim.

    == The Time Variance Authority ==
    Captured by Minutemen in the Gobi Desert.
    """
    doc = _create_raw_doc(raw_content)
    norm = normalizer.normalize(doc)

    assert len(norm.sections) == 4
    assert norm.sections[0].heading == "Loki (Character)"
    assert "Loki is a prince of Asgard." in norm.sections[0].content

    assert norm.sections[1].heading == "Early Life"
    assert "Raised in Asgard alongside Thor." in norm.sections[1].content

    assert norm.sections[2].heading == "The Frost Giant Revelation"
    assert "Discovered his true heritage in Jotunheim." in norm.sections[2].content

    assert norm.sections[3].heading == "The Time Variance Authority"
    assert "Captured by Minutemen in the Gobi Desert." in norm.sections[3].content


@pytest.mark.unit
def test_metadata_and_license_preservation() -> None:
    normalizer = DefaultDocumentNormalizer()
    raw_content = "<p>Loki lore content.</p>"
    meta = {
        "title": "Loki",
        "license": "CC-BY-SA 3.0",
        "revision_id": "rev-9999",
        "categories": '["Characters", "Asgardians"]',
    }
    doc = _create_raw_doc(raw_content, metadata=meta)
    norm = normalizer.normalize(doc)

    assert norm.title == "Loki"
    assert norm.metadata["license"] == "CC-BY-SA 3.0"
    assert norm.metadata["revision_id"] == "rev-9999"
    assert norm.metadata["categories"] == '["Characters", "Asgardians"]'
    assert norm.content_hash == compute_sha256(norm.text)
