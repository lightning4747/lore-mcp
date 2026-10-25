"""Normalization and deduplication port interfaces."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from lore_mcp.domain.documents import (
    DialogueCue,
    DialogueSegment,
    NormalizedDocument,
    RawDocument,
)
from lore_mcp.domain.normalization import (
    DeduplicationReport,
    DialogueNormalizationOptions,
    DocumentNormalizationOptions,
)


@runtime_checkable
class DocumentNormalizer(Protocol):
    """Port for normalizing general wiki, html, and text source documents."""

    def normalize(
        self,
        doc: RawDocument,
        options: DocumentNormalizationOptions | None = None,
    ) -> NormalizedDocument:
        """Clean raw source text, strip noise, and extract sections."""
        ...


@runtime_checkable
class DialogueNormalizer(Protocol):
    """Port for normalizing dialogue transcripts and segmenting into scene windows."""

    def normalize_transcript(
        self,
        doc: RawDocument,
        options: DialogueNormalizationOptions | None = None,
    ) -> NormalizedDocument:
        """Clean subtitle cues, strip audio artifacts, and segment into scenes."""
        ...

    def segment_cues(
        self,
        cues: Sequence[DialogueCue],
        options: DialogueNormalizationOptions | None = None,
    ) -> list[DialogueSegment]:
        """Cluster dialogue cues into scene-like temporal windows using time gaps."""
        ...


@runtime_checkable
class Deduplicator(Protocol):
    """Port for deterministic document fingerprinting and deduplication."""

    def compute_fingerprint(self, content: str) -> str:
        """Compute a deterministic hash fingerprint for content comparison."""
        ...

    def filter_duplicates(
        self,
        docs: Sequence[NormalizedDocument],
    ) -> tuple[list[NormalizedDocument], DeduplicationReport]:
        """Eliminate redundant documents and collapse redirects into target docs."""
        ...


@runtime_checkable
class NormalizationPipeline(Protocol):
    """Port for orchestrating document normalization, deduplication, and persistence."""

    async def normalize_document(
        self, doc: RawDocument
    ) -> NormalizedDocument | None:
        """Normalize a single raw document according to its source type."""
        ...

    async def process_batch(
        self, docs: Sequence[RawDocument]
    ) -> tuple[list[NormalizedDocument], DeduplicationReport]:
        """Normalize a batch of raw documents, deduplicate, and persist to store."""
        ...
