"""DocumentSource adapter for local SRT subtitle transcripts."""

import logging
from pathlib import Path

from lore_mcp.adapters.srt.errors import SrtError
from lore_mcp.adapters.srt.models import TranscriptDocument
from lore_mcp.adapters.srt.parser import SrtParser
from lore_mcp.application.errors import DataUnavailable, InvalidArgument
from lore_mcp.domain.documents import RawDocument
from lore_mcp.domain.series_config import SeriesConfig
from lore_mcp.domain.types import SeriesId
from lore_mcp.ports.documents import DocumentSource

logger = logging.getLogger(__name__)


class SrtTranscriptSource(DocumentSource):
    """DocumentSource adapter that reads local .srt transcript files."""

    def __init__(
        self,
        transcripts_dirs: dict[SeriesId, Path] | None = None,
    ) -> None:
        self._transcripts_dirs: dict[SeriesId, Path] = (
            dict(transcripts_dirs) if transcripts_dirs else {}
        )

    def register_transcripts_dir(
        self,
        series_id: SeriesId | str,
        directory: Path | str,
    ) -> None:
        """Register the local directory containing SRT transcripts for a series."""
        sid = SeriesId(str(series_id))
        self._transcripts_dirs[sid] = Path(directory)

    def register_series_config(
        self,
        config: SeriesConfig,
        base_dir: Path | str = ".",
    ) -> None:
        """Register transcripts directory from a SeriesConfig if specified."""
        if config.transcripts_dir:
            full_path = Path(base_dir) / config.transcripts_dir
            self.register_transcripts_dir(config.id, full_path)

    def get_transcripts_dir(self, series_id: SeriesId | str) -> Path | None:
        """Get the configured transcripts directory for a series."""
        sid = SeriesId(str(series_id))
        return self._transcripts_dirs.get(sid)

    def _resolve_srt_path(self, series_id: SeriesId, locator: str) -> Path:
        """Resolve locator to a validated local .srt path. Never allow remote URLs."""
        if locator.startswith(("http://", "https://", "ftp://")):
            raise InvalidArgument(
                f"Remote transcripts are not supported: '{locator}'. "
                f"Local .srt files only.",
                argument_name="locator",
            )

        loc_path = Path(locator)
        # Direct local file check
        if loc_path.is_file() and loc_path.suffix.lower() == ".srt":
            return loc_path

        # Resolve against configured directory
        dir_path = self.get_transcripts_dir(series_id)
        if dir_path and dir_path.is_dir():
            candidate = dir_path / locator
            if candidate.is_file() and candidate.suffix.lower() == ".srt":
                return candidate
            # Try appending .srt if omitted
            if not locator.endswith(".srt"):
                candidate_with_ext = dir_path / f"{locator}.srt"
                if candidate_with_ext.is_file():
                    return candidate_with_ext

        raise DataUnavailable(
            f"Transcript file '{locator}' not found for series '{series_id}'",
            resource_id=locator,
        )

    async def list_locators(self, series_id: SeriesId) -> list[str]:
        """List all available .srt transcript files for a series."""
        dir_path = self.get_transcripts_dir(series_id)
        if not dir_path or not dir_path.is_dir():
            return []

        locators: list[str] = []
        for file in sorted(dir_path.glob("*.srt")):
            if file.is_file():
                locators.append(file.name)
        return locators

    async def fetch_transcript(
        self,
        series_id: SeriesId,
        locator: str,
    ) -> TranscriptDocument:
        """Parse and return a structured TranscriptDocument."""
        srt_path = self._resolve_srt_path(series_id, locator)
        try:
            return SrtParser.parse_file(series_id, srt_path)
        except SrtError as err:
            raise DataUnavailable(
                f"Failed to read transcript '{locator}': {err}",
                resource_id=locator,
            ) from err

    async def fetch_document(self, series_id: SeriesId, locator: str) -> RawDocument:
        """Fetch raw document for a given SRT locator satisfying DocumentSource."""
        transcript = await self.fetch_transcript(series_id, locator)
        return transcript.to_raw_document()
