"""Domain models for narrative chronology, temporal metadata, and context (Task 9.1)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.types import ChunkId, EntityId, ValidatedSeriesId

OrderingStrategy = Literal["narrative", "as_shown", "unknown"]


class TemporalMetadata(BaseModel):
    """Explicit temporal coordinates representing narrative, broadcast,
    and scene order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    scene_index: int | None = Field(
        default=None,
        description="Scene index within episode (scene order)",
    )
    season: int | None = Field(
        default=None,
        description="Season number (episode order)",
    )
    episode: int | None = Field(
        default=None,
        description="Episode number (episode order)",
    )
    narrative_order: int | None = Field(
        default=None,
        description="Position in fictional in-universe timeline (nullable)",
    )
    narrative_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score in the narrative order assignment (0.0 to 1.0)",
    )


class Evidence(BaseModel):
    """Single item of verified evidence returned to an MCP client."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: ChunkId = Field(description="Unique chunk identifier")
    series_id: ValidatedSeriesId = Field(description="Owning series identifier")
    content: str = Field(description="Evidence textual content")
    entity_id: EntityId | None = Field(
        default=None, description="Primary entity associated with evidence"
    )
    season: int | None = Field(default=None, description="Season number if applicable")
    episode: int | None = Field(
        default=None, description="Episode number if applicable"
    )
    scene_index: int | None = Field(
        default=None, description="Scene index within the depicting episode"
    )
    narrative_order: int | None = Field(
        default=None, description="Chronological timeline position"
    )
    narrative_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence in narrative order assignment (0.0 to 1.0)",
    )
    score: float = Field(description="Final relevance score")
    dense_score: float | None = Field(
        default=None, description="Dense vector similarity score"
    )
    sparse_score: float | None = Field(default=None, description="Sparse BM25 score")
    rerank_score: float | None = Field(default=None, description="Reranker score")
    provenance: Provenance = Field(
        description="Source provenance for citation and attribution"
    )

    def temporal_metadata(self) -> TemporalMetadata:
        """Extract explicit TemporalMetadata representation."""
        return TemporalMetadata(
            season=self.season,
            episode=self.episode,
            scene_index=self.scene_index,
            narrative_order=self.narrative_order,
            narrative_confidence=self.narrative_confidence,
        )


class ChronologicalGroup(BaseModel):
    """Evidence items grouped by entity or event, partitioned into ordered
    and unordered subsets.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(description="Group key (entity_id, event_id, or 'general')")
    ordered_items: list[Evidence] = Field(
        default_factory=list[Evidence],
        description="Items with deterministic sequence ordering",
    )
    unordered_remainder: list[Evidence] = Field(
        default_factory=list[Evidence],
        description="Items lacking chronological metadata (never arbitrarily placed)",
    )
    ordering: OrderingStrategy = Field(
        default="unknown",
        description="Ordering scheme applied: 'narrative', 'as_shown', or 'unknown'",
    )
    uncertain_ids: list[ChunkId] = Field(
        default_factory=list[ChunkId],
        description="Chunk IDs within this group having uncertain or missing order",
    )
    conflicts: list[str] = Field(
        default_factory=list[str],
        description="Explanations of detected chronological contradictions",
    )


class TemporalContext(BaseModel):
    """Complete chronological context built from retrieved evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ordering: OrderingStrategy = Field(
        default="unknown",
        description="Overall chronological sequence scheme",
    )
    groups: list[ChronologicalGroup] = Field(
        default_factory=list[ChronologicalGroup],
        description="Chronologically organized evidence groups",
    )
    timeline_notes: list[str] = Field(
        default_factory=list[str],
        description="Sequential timeline notes and annotations",
    )
    uncertain_ids: list[ChunkId] = Field(
        default_factory=list[ChunkId],
        description="All chunk IDs with missing or conflicting temporal information",
    )
    conflicts: list[str] = Field(
        default_factory=list[str],
        description="Detected chronological contradictions or conflicting sources",
    )
    has_uncertainty: bool = Field(
        default=False,
        description="Whether any temporal uncertainty or conflict was detected",
    )
