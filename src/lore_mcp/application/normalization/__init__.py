"""Content normalization pipeline components (Phase 3)."""

from lore_mcp.application.normalization.deduplicator import (
    ContentDeduplicator,
)
from lore_mcp.application.normalization.dialogue_normalizer import (
    DefaultDialogueNormalizer,
)
from lore_mcp.application.normalization.document_normalizer import (
    DefaultDocumentNormalizer,
)
from lore_mcp.application.normalization.pipeline import (
    DefaultNormalizationPipeline,
)

__all__ = [
    "ContentDeduplicator",
    "DefaultDialogueNormalizer",
    "DefaultDocumentNormalizer",
    "DefaultNormalizationPipeline",
]
