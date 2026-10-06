"""Parser for SubRip (.srt) subtitle files with timestamp and encoding support."""

import html
import re
from datetime import UTC, datetime
from pathlib import Path

from lore_mcp.adapters.srt.errors import SrtEncodingError, SrtParseError
from lore_mcp.adapters.srt.models import TranscriptCue, TranscriptDocument
from lore_mcp.domain.types import SeriesId, ValidatedSeriesId

# Regex patterns for season/episode extraction from filenames
SEASON_EPISODE_PATTERNS = [
    # S02E06, s02e06, S2E6, S02.E06, S02_E06, S02-E06, LOKI_S2E6
    re.compile(r"(?i)(?:^|[\W_])[sS](\d{1,4})[._\s-]*[eE](\d{1,4})(?:$|[\W_])"),
    # 2x06, 02x06
    re.compile(r"(?i)(?:^|[\W_])(\d{1,2})x(\d{1,4})(?:$|[\W_])"),
    # Season 2 Episode 6, season 02 episode 06
    re.compile(
        r"(?i)(?:^|[\W_])season[._\s-]*(\d{1,4})[._\s-]*episode[._\s-]*(\d{1,4})(?:$|[\W_])"
    ),
    # Season 2 - Ep 06
    re.compile(
        r"(?i)(?:^|[\W_])season[._\s-]*(\d{1,4})[._\s-]*ep[._\s-]*(\d{1,4})(?:$|[\W_])"
    ),
]

# Regex for SRT timestamp line: 00:01:23,456 --> 00:01:25,789
TIMESTAMP_PATTERN = re.compile(
    r"^(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})"
)

# Regex for subtitle formatting markup
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
SSA_TAG_PATTERN = re.compile(r"\{[^\}]+\}")

# Regex for speaker prefixes like "LOKI: ..." or "MOBIUS M. MOBIUS: ..."
SPEAKER_PREFIX_PATTERN = re.compile(
    r"^([A-Z0-9][A-Z0-9\s'\.-]{1,35}):\s+(.*)$",
    re.DOTALL,
)

ENCODING_CANDIDATES = [
    "utf-8-sig",
    "utf-8",
    "cp1252",
    "iso-8859-1",
    "utf-16",
]


def extract_season_episode(filename: str) -> tuple[int | None, int | None]:
    """Extract season and episode numbers from filename (e.g. S02E06)."""
    base_stem = Path(filename).stem
    for pattern in SEASON_EPISODE_PATTERNS:
        match = pattern.search(base_stem)
        if match:
            season = int(match.group(1))
            episode = int(match.group(2))
            return season, episode
    return None, None


def strip_subtitle_markup(text: str) -> str:
    """Strip HTML formatting, SSA style tags, and unescape entities."""
    # 1. Strip HTML tags (<i>, <b>, <font...>, etc.) before unescaping entities
    no_html = HTML_TAG_PATTERN.sub("", text)
    # 2. Strip SSA/ASS tags ({\an8}, etc.)
    no_ssa = SSA_TAG_PATTERN.sub("", no_html)
    # 3. Unescape HTML entities (&amp;, &lt;, etc.)
    unescaped = html.unescape(no_ssa)
    # 4. Replace non-breaking spaces and collapse whitespace
    normalized_space = unescaped.replace("\xa0", " ")
    cleaned = re.sub(r"\s+", " ", normalized_space)
    return cleaned.strip()


def parse_timestamp_to_ms(ts_str: str) -> int:
    """Convert SRT timestamp 'HH:MM:SS,mmm' or 'HH:MM:SS.mmm' to milliseconds."""
    clean_ts = ts_str.replace(".", ",")
    parts = clean_ts.split(",")
    hms = parts[0].split(":")
    hours = int(hms[0])
    minutes = int(hms[1])
    seconds = int(hms[2])
    millis = int(parts[1].ljust(3, "0")[:3])
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + millis


def format_ms_to_timestamp(ms: int) -> str:
    """Format milliseconds back to canonical SRT timestamp 'HH:MM:SS,mmm'."""
    hours = ms // 3600000
    remainder = ms % 3600000
    minutes = remainder // 60000
    remainder = remainder % 60000
    seconds = remainder // 1000
    millis = remainder % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def read_file_with_encodings(path: Path | str) -> str:
    """Read file attempting common subtitle encodings in priority order."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"SRT file not found: {path}")

    raw_bytes = file_path.read_bytes()
    for enc in ENCODING_CANDIDATES:
        try:
            return raw_bytes.decode(enc)
        except (UnicodeDecodeError, ValueError):
            continue

    raise SrtEncodingError(str(path), ENCODING_CANDIDATES)


class SrtParser:
    """Parser for SubRip (.srt) subtitle files."""

    @classmethod
    def parse_text(
        cls,
        series_id: SeriesId | ValidatedSeriesId,
        content: str,
        source_path: str = "transcript.srt",
    ) -> TranscriptDocument:
        """Parse raw SRT content string into a structured TranscriptDocument."""
        season, episode = extract_season_episode(source_path)

        # Normalize line endings
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        # Split into double-newline separated cue blocks
        blocks = re.split(r"\n\s*\n", normalized)

        cues: list[TranscriptCue] = []
        cue_counter = 1

        for block in blocks:
            lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
            if not lines:
                continue

            # Identify timestamp line
            ts_line_idx = -1
            match = None
            for idx, line in enumerate(lines):
                match = TIMESTAMP_PATTERN.search(line)
                if match:
                    ts_line_idx = idx
                    break

            if ts_line_idx == -1 or match is None:
                continue

            raw_start = match.group(1)
            raw_end = match.group(2)
            start_ms = parse_timestamp_to_ms(raw_start)
            end_ms = parse_timestamp_to_ms(raw_end)
            start_time = format_ms_to_timestamp(start_ms)
            end_time = format_ms_to_timestamp(end_ms)

            # Subtitle text consists of lines after the timestamp line
            text_lines = lines[ts_line_idx + 1 :]
            raw_text = "\n".join(text_lines)
            cleaned_text = strip_subtitle_markup(raw_text)

            if not cleaned_text:
                continue

            # Detect speaker prefix if available
            speaker: str | None = None
            sp_match = SPEAKER_PREFIX_PATTERN.match(cleaned_text)
            if sp_match:
                speaker = sp_match.group(1).strip()
                cleaned_text = sp_match.group(2).strip()

            cues.append(
                TranscriptCue(
                    index=cue_counter,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    start_time=start_time,
                    end_time=end_time,
                    text=cleaned_text,
                    speaker=speaker,
                )
            )
            cue_counter += 1

        # Preserve dialogue ordering by start timestamp
        cues.sort(key=lambda c: (c.start_ms, c.index))

        return TranscriptDocument(
            series_id=SeriesId(str(series_id)),
            source_path=source_path,
            season=season,
            episode=episode,
            scene_index=None,
            cues=cues,
            retrieval_timestamp=datetime.now(UTC),
        )

    @classmethod
    def parse_file(
        cls,
        series_id: SeriesId | ValidatedSeriesId,
        path: Path | str,
    ) -> TranscriptDocument:
        """Read and parse an SRT file from the local filesystem."""
        file_path = Path(path)
        if not file_path.suffix.lower() == ".srt":
            raise SrtParseError(str(path), "File must have .srt extension")

        content = read_file_with_encodings(file_path)
        return cls.parse_text(
            series_id=series_id,
            content=content,
            source_path=str(file_path),
        )
