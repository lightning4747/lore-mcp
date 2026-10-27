"""Normalization pipeline orchestrator implementation (Phase 3)."""

from collections.abc import Sequence

from lore_mcp.application.normalization.deduplicator import ContentDeduplicator
from lore_mcp.application.normalization.dialogue_normalizer import (
    DefaultDialogueNormalizer,
)
from lore_mcp.application.normalization.document_normalizer import (
    DefaultDocumentNormalizer,
)
from lore_mcp.domain.documents import NormalizedDocument, RawDocument
from lore_mcp.domain.normalization import (
    DeduplicationReport,
    DialogueNormalizationOptions,
    DocumentNormalizationOptions,
)
from lore_mcp.ports.documents import DocumentStore
from lore_mcp.ports.normalization import (
    Deduplicator,
    DialogueNormalizer,
    DocumentNormalizer,
    NormalizationPipeline,
)


class DefaultNormalizationPipeline(NormalizationPipeline):
    """Pipeline coordinator for normalization, deduplication, and persistence."""

    def __init__(
        self,
        document_normalizer: DocumentNormalizer | None = None,
        dialogue_normalizer: DialogueNormalizer | None = None,
        deduplicator: Deduplicator | None = None,
        store: DocumentStore | None = None,
        doc_options: DocumentNormalizationOptions | None = None,
        dialogue_options: DialogueNormalizationOptions | None = None,
    ) -> None:
        self._doc_normalizer: DocumentNormalizer = (
            document_normalizer or DefaultDocumentNormalizer()
        )
        self._dialogue_normalizer: DialogueNormalizer = (
            dialogue_normalizer or DefaultDialogueNormalizer()
        )
        self._deduplicator: Deduplicator = deduplicator or ContentDeduplicator()
        self._store: DocumentStore | None = store
        self._doc_options = doc_options
        self._dialogue_options = dialogue_options

    async def normalize_document(self, doc: RawDocument) -> NormalizedDocument | None:
        """Normalize a single raw document according to its source type."""
        if not doc.content.strip():
            return None

        if doc.source_type.lower() in {"transcript", "dialogue", "subtitles", "srt"}:
            normalized = self._dialogue_normalizer.normalize_transcript(
                doc, self._dialogue_options
            )
        else:
            normalized = self._doc_normalizer.normalize(doc, self._doc_options)

        if not normalized.text.strip():
            return None

        return normalized

    async def process_batch(
        self, docs: Sequence[RawDocument]
    ) -> tuple[list[NormalizedDocument], DeduplicationReport]:
        """Normalize a batch of raw documents, deduplicate, and persist to store."""
        normalized_list: list[NormalizedDocument] = []
        for raw_doc in docs:
            normalized = await self.normalize_document(raw_doc)
            if normalized is not None:
                normalized_list.append(normalized)

        unique_docs, report = self._deduplicator.filter_duplicates(normalized_list)

        if self._store is not None:
            for u_doc in unique_docs:
                await self._store.put_normalized(u_doc)

        return unique_docs, report
