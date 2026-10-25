"""Unit tests for DefaultDialogueChunker (Task 5.3)."""

import json
from datetime import UTC, datetime

import pytest

from lore_mcp.application.chunking.dialogue_chunker import DefaultDialogueChunker
from lore_mcp.application.chunking.splitter import ChunkSplitter
from lore_mcp.domain.documents import (
    DialogueCue,
    DialogueSegment,
    NormalizedDocument,
)
from lore_mcp.domain.types import SeriesId


@pytest.mark.unit
class TestDialogueChunker:
    """Verifies dialogue chunking by scene windows with episode metadata."""

    @pytest.fixture
    def dialogue_document(self) -> NormalizedDocument:
        cues_1 = [
            DialogueCue(
                start_ms=1000,
                end_ms=3000,
                start_time="00:00:01,000",
                end_time="00:00:03,000",
                speaker="LOKI",
                text="I am Loki of Asgard.",
            )
        ]
        cues_2 = [
            DialogueCue(
                start_ms=15000,
                end_ms=18000,
                start_time="00:00:15,000",
                end_time="00:00:18,000",
                speaker="MOBIUS",
                text="You have a lot of explaining to do.",
            )
        ]
        segments = [
            DialogueSegment(
                scene_index=1,
                start_ms=1000,
                end_ms=3000,
                start_time="00:00:01,000",
                end_time="00:00:03,000",
                cues=cues_1,
                text="[00:00:01,000 --> 00:00:03,000] LOKI: I am Loki of Asgard.",
            ),
            DialogueSegment(
                scene_index=2,
                start_ms=15000,
                end_ms=18000,
                start_time="00:00:15,000",
                end_time="00:00:18,000",
                cues=cues_2,
                text=(
                    "[00:00:15,000 --> 00:00:18,000] "
                    "MOBIUS: You have a lot of explaining to do."
                ),
            ),
        ]
        raw_segments_json = json.dumps([s.model_dump() for s in segments])

        return NormalizedDocument(
            series_id=SeriesId("loki"),
            source_url="transcripts/loki_s1e1.srt",
            source_type="transcript",
            title="Loki - S01E01 - Glorious Purpose",
            text="Dialogue text",
            content_hash="diag-hash",
            retrieval_timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            metadata={
                "season": "1",
                "episode": "1",
                "dialogue_segments": raw_segments_json,
                "license": "Fair Use / Fan Transcript",
            },
        )

    def test_dialogue_chunking_preserves_scenes_and_metadata(
        self, dialogue_document: NormalizedDocument
    ) -> None:
        chunker = DefaultDialogueChunker()
        chunks = chunker.chunk_dialogue(dialogue_document)

        assert len(chunks) == 2

        c1 = chunks[0]
        assert c1.series_id == "loki"
        assert c1.season == 1
        assert c1.episode == 1
        assert c1.scene_index == 1
        assert c1.entity_type == "dialogue"
        assert c1.primary_concept == "Loki - S01E01 - Glorious Purpose"
        assert "LOKI: I am Loki of Asgard." in c1.text
        assert c1.provenance.license == "Fair Use / Fan Transcript"

        c2 = chunks[1]
        assert c2.season == 1
        assert c2.episode == 1
        assert c2.scene_index == 2
        assert "MOBIUS: You have a lot of explaining to do." in c2.text

    def test_missing_dialogue_segments_metadata_raises(self) -> None:
        doc = NormalizedDocument(
            series_id=SeriesId("loki"),
            source_url="transcripts/loki_s1e1.srt",
            source_type="transcript",
            title="Loki - S01E01",
            text="Dialogue text",
            content_hash="diag-hash",
            retrieval_timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            metadata={"season": "1", "episode": "1"},
        )
        chunker = DefaultDialogueChunker()
        with pytest.raises(ValueError, match="missing 'dialogue_segments' metadata"):
            chunker.chunk_dialogue(doc)

    def test_empty_dialogue_segments_returns_empty_list(self) -> None:
        doc = NormalizedDocument(
            series_id=SeriesId("loki"),
            source_url="transcripts/loki_s1e1.srt",
            source_type="transcript",
            title="Loki - S01E01",
            text="",
            content_hash="diag-hash",
            retrieval_timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            metadata={"season": "1", "episode": "1", "dialogue_segments": "[]"},
        )
        chunker = DefaultDialogueChunker()
        assert chunker.chunk_dialogue(doc) == []

    def test_oversized_scene_splits_while_preserving_scene_index(self) -> None:
        cues = [
            DialogueCue(
                start_ms=i * 1000,
                end_ms=(i + 1) * 1000,
                start_time=f"00:00:{i:02d},000",
                end_time=f"00:00:{i+1:02d},000",
                speaker="SPEAKER",
                text=(
                    f"Line {i} of scene dialogue that continues for a long duration."
                ),
            )
            for i in range(15)
        ]
        text_lines = [
            f"[{c.start_time} --> {c.end_time}] {c.speaker}: {c.text}"
            for c in cues
        ]
        seg = DialogueSegment(
            scene_index=4,
            start_ms=0,
            end_ms=15000,
            start_time="00:00:00,000",
            end_time="00:00:15,000",
            cues=cues,
            text="\n".join(text_lines),
        )
        doc = NormalizedDocument(
            series_id=SeriesId("loki"),
            source_url="transcripts/loki_s1e1.srt",
            source_type="transcript",
            title="Loki - S01E01",
            text="Dialogue",
            content_hash="diag-hash",
            retrieval_timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            metadata={
                "season": "1",
                "episode": "1",
                "dialogue_segments": json.dumps([seg.model_dump()]),
            },
        )
        small_splitter = ChunkSplitter(max_tokens=50)
        chunker = DefaultDialogueChunker(splitter=small_splitter)
        chunks = chunker.chunk_dialogue(doc)

        assert len(chunks) >= 2
        for c in chunks:
            assert c.scene_index == 4
            assert c.season == 1
            assert c.episode == 1
