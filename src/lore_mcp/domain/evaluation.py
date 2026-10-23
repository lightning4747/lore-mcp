"""Domain models and schemas for retrieval quality evaluation (Phase 11)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lore_mcp.domain.types import EntityId, ValidatedSeriesId

CaseType = Literal[
    "simple_fact",
    "alias_entity",
    "dialogue",
    "multi_part",
    "chronology",
    "spoiler_bound",
    "unanswerable",
]

PipelineType = Literal["dense", "sparse", "hybrid"]
FusionMode = Literal["client", "server"]


class SceneLocator(BaseModel):
    """Locator identifying an episode and optionally a specific scene."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    season: int = Field(ge=1, description="Season number (1-indexed)")
    episode: int = Field(ge=1, description="Episode number (1-indexed)")
    scene_index: int | None = Field(
        default=None, ge=0, description="Optional scene index within episode"
    )


class ExpectedEvidence(BaseModel):
    """Ground-truth expectation decoupled from volatile chunk identifiers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_ids: list[EntityId] = Field(
        default_factory=list[EntityId],
        description="Canonical franchise entity identifiers that satisfy relevance",
    )
    scenes: list[SceneLocator] = Field(
        default_factory=list[SceneLocator],
        description="Target scenes that contain the required evidence",
    )
    dialogue_quotes: list[str] = Field(
        default_factory=list[str],
        description="Key quotes or verbatim phrases expected in retrieved text",
    )
    chronological_order: list[str] = Field(
        default_factory=list[str],
        description=(
            "Ordered sequence of event/entity descriptors for timeline validation"
        ),
    )


class GoldenCase(BaseModel):
    """Canonical evaluation test case for the golden dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(description="Unique stable identifier for this benchmark case")
    query: str = Field(description="Natural language query under evaluation")
    series: ValidatedSeriesId = Field(description="Series scope for the query")
    case_type: CaseType = Field(description="Category of retrieval test case")
    season_bound: int | None = Field(
        default=None,
        ge=1,
        description="Optional maximum season bound for spoiler testing",
    )
    episode_bound: int | None = Field(
        default=None,
        ge=1,
        description="Optional maximum episode bound for spoiler testing",
    )
    expected_evidence: ExpectedEvidence = Field(
        default_factory=ExpectedEvidence,
        description="Decoupled ground-truth evidence specifications",
    )


class EvaluationConfig(BaseModel):
    """Configuration options and pipeline overrides for evaluation runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pipeline: PipelineType = Field(
        default="hybrid", description="Retrieval pipeline to test"
    )
    fusion_mode: FusionMode = Field(
        default="client", description="Fusion execution location: client or server"
    )
    enable_rerank: bool = Field(
        default=True, description="Whether to apply semantic/heuristic reranking"
    )
    rrf_k: int = Field(
        default=60, ge=1, description="Reciprocal Rank Fusion k constant"
    )
    max_results: int = Field(
        default=10, ge=1, description="Number of results to retrieve per query"
    )
    dataset_path: str = Field(
        default="eval/golden/loki.yaml", description="Path to golden YAML dataset"
    )
    series: str = Field(default="loki", description="Series identifier filter")
    tolerance: float = Field(
        default=0.05, ge=0.0, description="Allowed metric regression tolerance"
    )


class EvidenceMatch(BaseModel):
    """Summary of a retrieved evidence item and its relevance score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int = Field(ge=1, description="1-indexed rank in result list")
    chunk_id: str = Field(description="Chunk ID of retrieved item")
    series_id: str = Field(description="Series ID of retrieved item")
    relevance_score: int = Field(
        ge=0, le=2, description="Graded relevance score (0=none, 1=partial, 2=exact)"
    )
    entity_ids: list[str] = Field(default_factory=list[str])
    season: int | None = None
    episode: int | None = None
    scene_index: int | None = None
    narrative_order: int | None = None
    text_snippet: str = ""


class CaseEvaluationResult(BaseModel):
    """Diagnostic and metric outcomes for an individual test case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    case_type: CaseType
    query: str
    hit_at_5: bool
    hit_at_10: bool
    reciprocal_rank: float
    ndcg_at_5: float
    ndcg_at_10: float
    cross_series_leaks: int
    spoiler_leaks: int
    chronology_tau: float | None
    fallback_correct: bool
    latency_ms: float
    cache_hit: bool
    matches: list[EvidenceMatch] = Field(default_factory=list[EvidenceMatch])


class AggregateMetrics(BaseModel):
    """Aggregated retrieval quality statistics across the benchmark dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_cases: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float
    total_cross_series_leaks: int
    total_spoiler_leaks: int
    mean_chronology_tau: float | None
    fallback_accuracy: float
    cache_hit_rate: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    by_case_type: dict[str, dict[str, float]] = Field(
        default_factory=dict[str, dict[str, float]]
    )


class EvaluationReport(BaseModel):
    """Complete serialized evaluation run report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: str
    config: EvaluationConfig
    metrics: AggregateMetrics
    case_results: list[CaseEvaluationResult]
