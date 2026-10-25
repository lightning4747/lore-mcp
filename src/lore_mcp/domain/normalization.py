"""Domain models for content normalization, dialogue windowing, and deduplication."""

from pydantic import BaseModel, ConfigDict, Field


class DocumentNormalizationOptions(BaseModel):
    """Configuration options for general document and wiki normalization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strip_infoboxes: bool = Field(
        default=True, description="Whether to strip infobox template boilerplate"
    )
    strip_navboxes: bool = Field(
        default=True, description="Whether to strip navigation boxes and menus"
    )
    preserve_headings: bool = Field(
        default=True, description="Whether to structure text into heading sections"
    )
    nfc_normalize: bool = Field(
        default=True,
        description="Whether to perform Unicode normalization to NFC form",
    )


class DialogueNormalizationOptions(BaseModel):
    """Configuration options for dialogue transcript cleanup and scene windowing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pause_threshold_ms: int = Field(
        default=4000,
        description="Silence duration in milliseconds demarcating scene transitions",
    )
    max_segment_duration_ms: int = Field(
        default=180000,
        description="Maximum segment window duration in milliseconds (default 3 mins)",
    )
    max_cues_per_segment: int = Field(
        default=40,
        description="Maximum cue count per segment before forcing a window break",
    )
    strip_sound_effects: bool = Field(
        default=True,
        description="Whether to strip non-speech auditory annotations [MUSIC, etc.]",
    )


class DeduplicationReport(BaseModel):
    """Summary of documents processed and deduplicated."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_documents_processed: int = Field(default=0)
    unique_documents_output: int = Field(default=0)
    exact_duplicates_dropped: int = Field(default=0)
    redirects_collapsed: int = Field(default=0)
    recap_cues_skipped: int = Field(default=0)
