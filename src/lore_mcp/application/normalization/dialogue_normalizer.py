"""Normalizer for dialogue transcripts and subtitle streams (Task 3.2)."""

import hashlib
import html
import re
from collections.abc import Sequence

from pydantic import TypeAdapter

from lore_mcp.domain.documents import (
    DialogueCue,
    DialogueSegment,
    DocumentSection,
    NormalizedDocument,
    RawDocument,
)
from lore_mcp.domain.normalization import DialogueNormalizationOptions
from lore_mcp.ports.normalization import DialogueNormalizer

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
SSA_TAG_PATTERN = re.compile(r"\{[^\}]+\}")
MUSIC_NOTES_PATTERN = re.compile(r"[♪♫#]+")

# Standalone bracketed/parenthesized noise (e.g. "[MUSIC PLAYING]", "(CHUCKLES)")
STANDALONE_NOISE_PATTERN = re.compile(
    r"^\s*[\[\(][A-Za-z0-9\s,.'\-_!?♪♫#]+[\]\)]\s*$"
)

# Inline sound effect tags within spoken text (e.g. "I am... (laughs) burdened")
INLINE_SOUND_PATTERN = re.compile(
    r"[\[\(]"
    r"(?:chuckles|sighs|groans|gasps|laughs|screams|screaming|sobs|crying|"
    r"applause|whispers|grunts|exclaims|snickers|gasping|coughing|theme music|"
    r"music|music playing|rock music|dramatic music|eerie music|tense music)"
    r"[\]\)]",
    re.IGNORECASE,
)

# Speaker prefix patterns: "LOKI: ..." or "[Loki] ..."
SPEAKER_PREFIX_COLON = re.compile(r"^([A-Z0-9][A-Za-z0-9\s'\.-]{1,35}):\s*(.*)$")
SPEAKER_PREFIX_BRACKET = re.compile(r"^\[([A-Z0-9][A-Za-z0-9\s'\.-]{1,35})\]\s*(.*)$")

# Fallback timestamp pattern: 00:01:23,456 --> 00:01:25,789
FALLBACK_TIMESTAMP_LINE = re.compile(
    r"^\[?(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\]?\s*(.*)$"
)


def compute_sha256(text: str) -> str:
    """Compute deterministic SHA-256 hex digest of string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_timestamp_ms(ts: str) -> int:
    """Convert 'HH:MM:SS,mmm' or 'HH:MM:SS.mmm' into milliseconds."""
    clean = ts.strip("[] ").replace(".", ",")
    parts = clean.split(":")
    if len(parts) != 3:
        return 0
    hours = int(parts[0])
    minutes = int(parts[1])
    sec_parts = parts[2].split(",")
    seconds = int(sec_parts[0])
    millis = int(sec_parts[1]) if len(sec_parts) > 1 else 0
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + millis


def format_timestamp_ms(ms: int) -> str:
    """Format milliseconds into 'HH:MM:SS,mmm'."""
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def normalize_speaker_name(raw_name: str) -> str:
    """Normalize speaker name to title case if ALL CAPS, preserving designations."""
    name = raw_name.strip()
    if name.isupper():
        # Title case standard names (LOKI -> Loki), preserve designations like B-15
        words = name.split()
        return " ".join(
            w.capitalize() if not re.match(r"^[A-Z]-\d+$", w) else w
            for w in words
        )
    return name


class DefaultDialogueNormalizer(DialogueNormalizer):
    """Normalizes subtitle dialogue transcripts and segments them into scene windows."""

    def __init__(
        self, default_options: DialogueNormalizationOptions | None = None
    ) -> None:
        self._default_options = default_options or DialogueNormalizationOptions()

    def clean_cue_text(
        self,
        raw_text: str,
        existing_speaker: str | None,
        options: DialogueNormalizationOptions,
    ) -> tuple[str, str | None]:
        """Strip formatting, remove noise, and extract clean speaker & dialogue."""
        # 1. Strip HTML tags
        text = HTML_TAG_PATTERN.sub("", raw_text)

        # 2. Strip SSA style tags
        text = SSA_TAG_PATTERN.sub("", text)

        # 3. Strip music note symbols
        text = MUSIC_NOTES_PATTERN.sub("", text)

        # 4. Unescape HTML entities
        text = html.unescape(text).strip()

        # 5. Filter standalone noise descriptions
        if options.strip_sound_effects and STANDALONE_NOISE_PATTERN.match(text):
            return "", None

        # 6. Extract speaker if embedded in text
        speaker = existing_speaker
        match_colon = SPEAKER_PREFIX_COLON.match(text)
        if match_colon:
            speaker = normalize_speaker_name(match_colon.group(1))
            text = match_colon.group(2).strip()
        else:
            match_bracket = SPEAKER_PREFIX_BRACKET.match(text)
            if match_bracket:
                speaker = normalize_speaker_name(match_bracket.group(1))
                text = match_bracket.group(2).strip()

        if speaker:
            speaker = normalize_speaker_name(speaker)

        # 7. Strip inline sound effects if enabled
        if options.strip_sound_effects:
            text = INLINE_SOUND_PATTERN.sub("", text)

        # 8. Clean whitespace
        text = re.sub(r"[ \t]+", " ", text).strip()
        return text, speaker

    def parse_raw_cues(
        self, doc: RawDocument, options: DialogueNormalizationOptions
    ) -> list[DialogueCue]:
        """Parse, clean, and order all dialogue cues from RawDocument."""
        raw_cues_data: list[dict[str, object]] = []

        # Check metadata for cues_json (from SrtTranscriptSource)
        cues_json = doc.metadata.get("cues_json")
        if cues_json:
            try:
                adapter = TypeAdapter(list[dict[str, object]])
                raw_cues_data = adapter.validate_json(cues_json)
            except Exception:
                raw_cues_data = []

        cleaned_cues: list[DialogueCue] = []

        if raw_cues_data:
            for item in raw_cues_data:
                raw_text = str(item.get("text", ""))
                raw_speaker = (
                    str(item.get("speaker")) if item.get("speaker") else None
                )
                start_val = item.get("start_ms", 0)
                end_val = item.get("end_ms", 0)
                start_ms = (
                    int(start_val)
                    if isinstance(start_val, (int, float, str))
                    else 0
                )
                end_ms = (
                    int(end_val)
                    if isinstance(end_val, (int, float, str))
                    else 0
                )
                start_time = str(
                    item.get("start_time") or format_timestamp_ms(start_ms)
                )
                end_time = str(
                    item.get("end_time") or format_timestamp_ms(end_ms)
                )

                clean_text, speaker = self.clean_cue_text(
                    raw_text, raw_speaker, options
                )
                if clean_text:
                    cleaned_cues.append(
                        DialogueCue(
                            start_ms=start_ms,
                            end_ms=end_ms,
                            start_time=start_time,
                            end_time=end_time,
                            speaker=speaker,
                            text=clean_text,
                        )
                    )
        else:
            # Fallback line-by-line parsing from content
            for line in doc.content.split("\n"):
                line_str = line.strip()
                if not line_str:
                    continue
                ts_match = FALLBACK_TIMESTAMP_LINE.match(line_str)
                if ts_match:
                    start_time = ts_match.group(1)
                    end_time = ts_match.group(2)
                    rest = ts_match.group(3)
                    start_ms = parse_timestamp_ms(start_time)
                    end_ms = parse_timestamp_ms(end_time)
                    clean_text, speaker = self.clean_cue_text(
                        rest, None, options
                    )
                    if clean_text:
                        cleaned_cues.append(
                            DialogueCue(
                                start_ms=start_ms,
                                end_ms=end_ms,
                                start_time=start_time,
                                end_time=end_time,
                                speaker=speaker,
                                text=clean_text,
                            )
                        )

        # Sort strictly by start_ms
        cleaned_cues.sort(key=lambda c: (c.start_ms, c.end_ms))
        return cleaned_cues

    def segment_cues(
        self,
        cues: Sequence[DialogueCue],
        options: DialogueNormalizationOptions | None = None,
    ) -> list[DialogueSegment]:
        """Cluster dialogue cues into scene-like temporal windows using time gaps."""
        if not cues:
            return []

        opts = options or self._default_options
        segments: list[DialogueSegment] = []

        current_scene_index = 1
        current_cues: list[DialogueCue] = []

        for cue in cues:
            if not current_cues:
                current_cues.append(cue)
                continue

            last_cue = current_cues[-1]
            time_gap = cue.start_ms - last_cue.end_ms
            window_duration = cue.end_ms - current_cues[0].start_ms
            cue_count = len(current_cues)

            is_scene_break = (
                time_gap >= opts.pause_threshold_ms
                or window_duration >= opts.max_segment_duration_ms
                or cue_count >= opts.max_cues_per_segment
            )

            if is_scene_break:
                # Close current segment
                seg_text = "\n".join(
                    f"{c.speaker}: {c.text}" if c.speaker else c.text
                    for c in current_cues
                )
                segments.append(
                    DialogueSegment(
                        scene_index=current_scene_index,
                        start_ms=current_cues[0].start_ms,
                        end_ms=current_cues[-1].end_ms,
                        start_time=current_cues[0].start_time,
                        end_time=current_cues[-1].end_time,
                        cues=list(current_cues),
                        text=seg_text,
                    )
                )
                current_scene_index += 1
                current_cues = [cue]
            else:
                current_cues.append(cue)

        if current_cues:
            seg_text = "\n".join(
                f"{c.speaker}: {c.text}" if c.speaker else c.text
                for c in current_cues
            )
            segments.append(
                DialogueSegment(
                    scene_index=current_scene_index,
                    start_ms=current_cues[0].start_ms,
                    end_ms=current_cues[-1].end_ms,
                    start_time=current_cues[0].start_time,
                    end_time=current_cues[-1].end_time,
                    cues=list(current_cues),
                    text=seg_text,
                )
            )

        return segments

    def normalize_transcript(
        self,
        doc: RawDocument,
        options: DialogueNormalizationOptions | None = None,
    ) -> NormalizedDocument:
        """Clean subtitle cues, strip audio artifacts, and segment into scenes."""
        opts = options or self._default_options

        # 1. Clean cues and sort chronologically
        cleaned_cues = self.parse_raw_cues(doc, opts)

        # 2. Segment into scene windows
        segments = self.segment_cues(cleaned_cues, opts)

        # 3. Build DocumentSections
        sections: list[DocumentSection] = []
        for seg in segments:
            heading = f"Scene {seg.scene_index} [{seg.start_time} --> {seg.end_time}]"
            sections.append(
                DocumentSection(
                    heading=heading,
                    content=seg.text,
                )
            )

        # 4. Assembled full text
        full_text_blocks = [
            f"## {s.heading}\n\n{s.content}" for s in sections
        ]
        full_text = "\n\n".join(full_text_blocks)
        content_hash = compute_sha256(full_text)

        # 5. Extract episode metadata
        season = doc.metadata.get("season")
        episode = doc.metadata.get("episode")

        if season and episode:
            title = f"Episode S{int(season):02d}E{int(episode):02d} Dialogue"
        else:
            title = "Episode Dialogue"

        cleaned_meta: dict[str, str] = dict(doc.metadata)
        cleaned_meta["title"] = title
        cleaned_meta["content_hash"] = content_hash
        cleaned_meta["scene_count"] = str(len(segments))
        cleaned_meta["total_cues"] = str(len(cleaned_cues))

        return NormalizedDocument(
            series_id=doc.series_id,
            source_url=doc.source_url,
            source_type="transcript",
            title=title,
            text=full_text,
            sections=sections,
            content_hash=content_hash,
            retrieval_timestamp=doc.retrieval_timestamp,
            metadata=cleaned_meta,
        )
