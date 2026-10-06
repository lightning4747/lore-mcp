"""Unit tests for SRT transcript fetcher, parser, and DocumentSource adapter."""

from pathlib import Path

import pytest

from lore_mcp.adapters.srt import (
    SrtParseError,
    SrtParser,
    SrtTranscriptSource,
    extract_season_episode,
    strip_subtitle_markup,
)
from lore_mcp.application.errors import DataUnavailable, InvalidArgument
from lore_mcp.domain.types import SeriesId


@pytest.mark.unit
def test_extract_season_episode_formats() -> None:
    # Standard S02E06
    assert extract_season_episode("Loki.S02E06.srt") == (2, 6)
    assert extract_season_episode("loki.s01e01.srt") == (1, 1)
    assert extract_season_episode("LOKI_S2E6.srt") == (2, 6)
    assert extract_season_episode("Loki.S02.E06.srt") == (2, 6)
    assert extract_season_episode("Loki-S02-E06.srt") == (2, 6)

    # 2x06 format
    assert extract_season_episode("Loki.2x06.srt") == (2, 6)
    assert extract_season_episode("01x03.srt") == (1, 3)

    # Full words format
    assert extract_season_episode("Season 2 Episode 6.srt") == (2, 6)
    assert extract_season_episode("Season 01 Episode 04.srt") == (1, 4)
    assert extract_season_episode("Season 2 - Ep 05.srt") == (2, 5)

    # No season/episode in filename
    assert extract_season_episode("movie.srt") == (None, None)
    assert extract_season_episode("transcript_special.srt") == (None, None)


@pytest.mark.unit
def test_strip_subtitle_markup() -> None:
    # HTML tags
    html_input = '<i>It is</i> <b>glorious</b> <font color="#ff0000">purpose</font>.'
    assert strip_subtitle_markup(html_input) == "It is glorious purpose."

    # SSA/ASS tags
    ssa_input = "{\\an8}Look at the top! {\\pos(100,200)}Centered text."
    assert strip_subtitle_markup(ssa_input) == "Look at the top! Centered text."

    # HTML entities and whitespace
    entities_input = "Loki &amp; Sylvie &nbsp; escaped &lt;TVA&gt;."
    assert strip_subtitle_markup(entities_input) == "Loki & Sylvie escaped <TVA>."


@pytest.mark.unit
def test_srt_parsing_with_speaker_and_timestamps() -> None:
    srt_text = """1
00:00:01,000 --> 00:00:03,500
LOKI: I am burdened with glorious purpose.

2
00:00:04,100 --> 00:00:06,800
MOBIUS: We've got a lot of questions for you.

3
00:00:07,000 --> 00:00:09,000
<i>[dramatic orchestral music playing]</i>
"""
    doc = SrtParser.parse_text(
        series_id=SeriesId("loki"),
        content=srt_text,
        source_path="Loki.S01E01.srt",
    )

    assert doc.series_id == "loki"
    assert doc.season == 1
    assert doc.episode == 1
    assert doc.scene_index is None
    assert len(doc.cues) == 3

    # Cue 1: speaker extracted
    c1 = doc.cues[0]
    assert c1.speaker == "LOKI"
    assert c1.text == "I am burdened with glorious purpose."
    assert c1.start_ms == 1000
    assert c1.end_ms == 3500
    assert c1.start_time == "00:00:01,000"
    assert c1.end_time == "00:00:03,500"

    # Cue 2: speaker extracted
    c2 = doc.cues[1]
    assert c2.speaker == "MOBIUS"
    assert c2.text == "We've got a lot of questions for you."
    assert c2.start_ms == 4100
    assert c2.end_ms == 6800

    # Cue 3: no speaker prefix, markup stripped
    c3 = doc.cues[2]
    assert c3.speaker is None
    assert c3.text == "[dramatic orchestral music playing]"


@pytest.mark.unit
def test_dialogue_ordering_preserved() -> None:
    # Cues presented out of chronological order in file
    srt_out_of_order = """2
00:00:05,000 --> 00:00:08,000
Later dialogue line.

1
00:00:01,000 --> 00:00:04,000
Earlier dialogue line.
"""
    doc = SrtParser.parse_text(
        series_id=SeriesId("loki"),
        content=srt_out_of_order,
        source_path="S01E02.srt",
    )

    assert len(doc.cues) == 2
    assert doc.cues[0].text == "Earlier dialogue line."
    assert doc.cues[0].start_ms == 1000
    assert doc.cues[1].text == "Later dialogue line."
    assert doc.cues[1].start_ms == 5000


@pytest.mark.unit
def test_multiple_file_encodings(tmp_path: Path) -> None:
    content = "1\n00:00:01,000 --> 00:00:02,000\nCafé résumé naïve façade.\n"

    # 1. UTF-8 with BOM
    utf8_bom_file = tmp_path / "utf8_bom.srt"
    utf8_bom_file.write_bytes(content.encode("utf-8-sig"))
    doc1 = SrtParser.parse_file(SeriesId("loki"), utf8_bom_file)
    assert "Café résumé naïve façade." in doc1.cues[0].text

    # 2. Windows-1252 / CP1252
    cp1252_file = tmp_path / "cp1252.srt"
    cp1252_file.write_bytes(content.encode("cp1252"))
    doc2 = SrtParser.parse_file(SeriesId("loki"), cp1252_file)
    assert "Café résumé naïve façade." in doc2.cues[0].text

    # 3. ISO-8859-1
    iso_file = tmp_path / "iso.srt"
    iso_file.write_bytes(content.encode("iso-8859-1"))
    doc3 = SrtParser.parse_file(SeriesId("loki"), iso_file)
    assert "Café résumé naïve façade." in doc3.cues[0].text


@pytest.mark.unit
def test_malformed_srt_file_handling(tmp_path: Path) -> None:
    # Missing .srt extension
    txt_file = tmp_path / "sub.txt"
    txt_file.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
    with pytest.raises(SrtParseError) as exc_info:
        SrtParser.parse_file(SeriesId("loki"), txt_file)
    assert "File must have .srt extension" in str(exc_info.value)

    # Empty blocks or broken timestamps are skipped gracefully
    messy_srt = """1
broken timestamp line
some text

2
00:00:01,000 --> 00:00:03,000
Valid line.
"""
    doc = SrtParser.parse_text(SeriesId("loki"), messy_srt)
    assert len(doc.cues) == 1
    assert doc.cues[0].text == "Valid line."


@pytest.mark.unit
@pytest.mark.asyncio
async def test_srt_transcript_source_adapter(tmp_path: Path) -> None:
    ep1_file = tmp_path / "Loki.S02E01.srt"
    ep1_file.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nLOKI: Ouroboros!\n",
        encoding="utf-8",
    )

    ep2_file = tmp_path / "Loki.S02E02.srt"
    ep2_file.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nMOBIUS: Key lime pie.\n",
        encoding="utf-8",
    )

    source = SrtTranscriptSource()
    source.register_transcripts_dir(SeriesId("loki"), tmp_path)

    # 1. list_locators
    locators = await source.list_locators(SeriesId("loki"))
    assert locators == ["Loki.S02E01.srt", "Loki.S02E02.srt"]

    # 2. fetch_document
    raw_doc = await source.fetch_document(SeriesId("loki"), "Loki.S02E01.srt")
    assert raw_doc.series_id == "loki"
    assert raw_doc.source_type == "transcript"
    assert "file://" in raw_doc.source_url
    assert "LOKI: Ouroboros!" in raw_doc.content
    assert raw_doc.metadata["season"] == "2"
    assert raw_doc.metadata["episode"] == "1"
    assert raw_doc.metadata["scene_index"] == ""

    # 3. Remote URLs are rejected
    with pytest.raises(InvalidArgument) as inv_err:
        await source.fetch_document(
            SeriesId("loki"), "https://example.com/Loki.S02E01.srt"
        )
    assert "Remote transcripts are not supported" in str(inv_err.value)

    # 4. Nonexistent file raises DataUnavailable
    with pytest.raises(DataUnavailable) as miss_err:
        await source.fetch_document(SeriesId("loki"), "Loki.S02E03.srt")
    assert miss_err.value.resource_id == "Loki.S02E03.srt"

    # 5. Unknown series returns empty locators
    empty_locs = await source.list_locators(SeriesId("unknown"))
    assert empty_locs == []


@pytest.mark.unit
def test_gitignore_contains_srt_rule() -> None:
    gitignore_path = Path(".gitignore")
    assert gitignore_path.is_file()
    content = gitignore_path.read_text(encoding="utf-8")
    assert "*.srt" in content
