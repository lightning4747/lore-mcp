"""Data models for SRT transcript documents and dialogue cues."""

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lore_mcp.domain.documents import RawDocument
from lore_mcp.domain.types import SeriesId, ValidatedSeriesId


class TranscriptCue(BaseModel):
    """A discrete subtitle or dialogue cue with timing and text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(description="Sequential cue index in the SRT file")
    start_ms: int = Field(description="Start time in milliseconds")
    end_ms: int = Field(description="End time in milliseconds")
    start_time: str = Field(description="Formatted start timestamp (HH:MM:SS,mmm)")
    end_time: str = Field(description="Formatted end timestamp (HH:MM:SS,mmm)")
    text: str = Field(description="Cleaned dialogue text with markup stripped")
    speaker: str | None = Field(
        default=None,
        description="Identified speaker name, or null if unspecified",
    )


class TranscriptDocument(BaseModel):
    """Complete episode dialogue transcript parsed from an SRT file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: ValidatedSeriesId = Field(description="Owning series identifier")
    source_path: str = Field(description="Local file path of the SRT file")
    season: int | None = Field(
        default=None,
        description="Season number mapped from filename, or null",
    )
    episode: int | None = Field(
        default=None,
        description="Episode number mapped from filename, or null",
    )
    scene_index: int | None = Field(
        default=None,
        description="Scene index if available, defaults to null",
    )
    cues: list[TranscriptCue] = Field(
        default_factory=list,
        description="Ordered list of dialogue cues",
    )
    retrieval_timestamp: datetime = Field(
        description="Timestamp when transcript was loaded",
    )

    def to_raw_document(
        self,
        extra_metadata: dict[str, Any] | None = None,
    ) -> RawDocument:
        """Convert this TranscriptDocument into a domain RawDocument."""
        lines: list[str] = []
        for cue in self.cues:
            prefix = f"[{cue.start_time} --> {cue.end_time}] "
            if cue.speaker:
                lines.append(f"{prefix}{cue.speaker}: {cue.text}")
            else:
                lines.append(f"{prefix}{cue.text}")

        full_content = "\n".join(lines)

        meta: dict[str, str] = {
            "source_path": self.source_path,
            "season": str(self.season) if self.season is not None else "",
            "episode": str(self.episode) if self.episode is not None else "",
            "scene_index": str(self.scene_index)
            if self.scene_index is not None
            else "",
            "cue_count": str(len(self.cues)),
            "cues_json": json.dumps([c.model_dump() for c in self.cues]),
        }
        if extra_metadata:
            for k, v in extra_metadata.items():
                meta[k] = str(v)

        return RawDocument(
            series_id=SeriesId(str(self.series_id)),
            source_url=f"file://{self.source_path}",
            source_type="transcript",
            content=full_content,
            retrieval_timestamp=self.retrieval_timestamp,
            metadata=meta,
        )
