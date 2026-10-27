"""Unit tests for dialogue normalization and scene segmentation (Task 3.2 & 3.4)."""

import json
from datetime import UTC, datetime

import pytest

from lore_mcp.application.normalization.dialogue_normalizer import (
    DefaultDialogueNormalizer,
    format_timestamp_ms,
    parse_timestamp_ms,
)
from lore_mcp.domain.documents import DialogueCue, RawDocument
from lore_mcp.domain.normalization import DialogueNormalizationOptions
from lore_mcp.domain.types import SeriesId


def _create_transcript_raw_doc(
    cues_data: list[dict[str, object]],
    season: int = 1,
    episode: int = 2,
) -> RawDocument:
    meta = {
        "source_path": f"transcripts/loki/s{season:02d}e{episode:02d}.srt",
        "season": str(season),
        "episode": str(episode),
        "cues_json": json.dumps(cues_data),
    }
    return RawDocument(
        series_id=SeriesId("loki"),
        source_url=f"file://transcripts/loki/s{season:02d}e{episode:02d}.srt",
        source_type="transcript",
        content="",
        retrieval_timestamp=datetime.now(UTC),
        metadata=meta,
    )


@pytest.mark.unit
def test_timestamp_parsing_and_formatting() -> None:
    ts = "01:23:45,678"
    ms = parse_timestamp_ms(ts)
    assert ms == (1 * 3600 + 23 * 60 + 45) * 1000 + 678
    formatted = format_timestamp_ms(ms)
    assert formatted == ts


@pytest.mark.unit
def test_subtitle_markup_and_audio_noise_stripping() -> None:
    normalizer = DefaultDialogueNormalizer()
    opts = DialogueNormalizationOptions()

    # 1. HTML and SSA tags
    text, spk = normalizer.clean_cue_text(
        "<i>{\\an8}I am Loki of Asgard.</i>", "LOKI", opts
    )
    assert text == "I am Loki of Asgard."
    assert spk == "Loki"

    # 2. Standalone music and sound effects dropped
    text_music, _ = normalizer.clean_cue_text("♪ [DRAMATIC THEME MUSIC] ♪", None, opts)
    assert text_music == ""

    text_noise, _ = normalizer.clean_cue_text("(CHUCKLES)", None, opts)
    assert text_noise == ""

    # 3. Inline sound effect stripped while retaining dialogue
    text_inline, spk_inline = normalizer.clean_cue_text(
        "LOKI: I am... (laughs) burdened with glorious purpose.", None, opts
    )
    assert text_inline == "I am... burdened with glorious purpose."
    assert spk_inline == "Loki"


@pytest.mark.unit
def test_speaker_normalization() -> None:
    normalizer = DefaultDialogueNormalizer()
    opts = DialogueNormalizationOptions()

    text, spk = normalizer.clean_cue_text(
        "HUNTER B-15: Reset the timeline.", None, opts
    )
    assert spk == "Hunter B-15"
    assert text == "Reset the timeline."

    text2, spk2 = normalizer.clean_cue_text("[MOBIUS] You like jet skis?", None, opts)
    assert spk2 == "Mobius"
    assert text2 == "You like jet skis?"


@pytest.mark.unit
def test_time_gap_scene_segmentation() -> None:
    normalizer = DefaultDialogueNormalizer()
    # Pause threshold: 4000ms (4 seconds)
    cues = [
        # Scene 1: cues close together (gap: 500ms)
        DialogueCue(
            start_ms=1000,
            end_ms=3000,
            start_time="00:00:01,000",
            end_time="00:00:03,000",
            speaker="Loki",
            text="I am burdened with glorious purpose.",
        ),
        DialogueCue(
            start_ms=3500,
            end_ms=6000,
            start_time="00:00:03,500",
            end_time="00:00:06,000",
            speaker="Mobius",
            text="Yeah, about that...",
        ),
        # Scene break: gap = 15000 - 6000 = 9000ms >= 4000ms
        DialogueCue(
            start_ms=15000,
            end_ms=18000,
            start_time="00:00:15,000",
            end_time="00:00:18,000",
            speaker="Sylvie",
            text="Do you really think this is about you?",
        ),
        # Scene 2 continued (gap: 1000ms)
        DialogueCue(
            start_ms=19000,
            end_ms=22000,
            start_time="00:00:19,000",
            end_time="00:00:22,000",
            speaker="Loki",
            text="This is about us.",
        ),
    ]

    segments = normalizer.segment_cues(cues)
    assert len(segments) == 2

    seg1 = segments[0]
    assert seg1.scene_index == 1
    assert seg1.start_ms == 1000
    assert seg1.end_ms == 6000
    assert len(seg1.cues) == 2
    assert "Loki: I am burdened with glorious purpose." in seg1.text
    assert "Mobius: Yeah, about that..." in seg1.text

    seg2 = segments[1]
    assert seg2.scene_index == 2
    assert seg2.start_ms == 15000
    assert seg2.end_ms == 22000
    assert len(seg2.cues) == 2
    assert "Sylvie: Do you really think this is about you?" in seg2.text
    assert "Loki: This is about us." in seg2.text


@pytest.mark.unit
def test_normalize_transcript_document() -> None:
    normalizer = DefaultDialogueNormalizer()
    cues_data = [
        {
            "start_ms": 1000,
            "end_ms": 3000,
            "start_time": "00:00:01,000",
            "end_time": "00:00:03,000",
            "speaker": "LOKI",
            "text": "<i>Burdened with purpose.</i>",
        },
        {
            "start_ms": 3500,
            "end_ms": 4000,
            "start_time": "00:00:03,500",
            "end_time": "00:00:04,000",
            "speaker": None,
            "text": "[CHUCKLES]",
        },
        # Gap > 4000ms
        {
            "start_ms": 10000,
            "end_ms": 13000,
            "start_time": "00:00:10,000",
            "end_time": "00:00:13,000",
            "speaker": "MOBIUS",
            "text": "Let's take a walk.",
        },
    ]

    raw_doc = _create_transcript_raw_doc(cues_data, season=1, episode=2)
    norm = normalizer.normalize_transcript(raw_doc)

    assert norm.source_type == "transcript"
    assert norm.title == "Episode S01E02 Dialogue"
    assert norm.metadata["season"] == "1"
    assert norm.metadata["episode"] == "2"
    assert norm.metadata["scene_count"] == "2"
    assert norm.metadata["total_cues"] == "2"  # [CHUCKLES] was filtered out

    assert len(norm.sections) == 2
    assert norm.sections[0].heading == "Scene 1 [00:00:01,000 --> 00:00:03,000]"
    assert "Loki: Burdened with purpose." in norm.sections[0].content
    assert norm.sections[1].heading == "Scene 2 [00:00:10,000 --> 00:00:13,000]"
    assert "Mobius: Let's take a walk." in norm.sections[1].content
