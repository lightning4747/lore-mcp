"""Domain models for lore search queries, retrieval hits, and evidence bundles."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.types import ChunkId, EntityId, ValidatedSeriesId


class SearchQuery(BaseModel):
    """Retrieval search query locked to a required series scope."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: ValidatedSeriesId = Field(
        description="Target series identifier (mandatory scope filter)"
    )
    query: str = Field(description="Search query text")
    season: int | None = Field(
        default=None, description="Exact season filter if specified"
    )
    episode: int | None = Field(
        default=None, description="Exact episode filter if specified"
    )
    max_season: int | None = Field(
        default=None, description="Spoiler guard maximum season"
    )
    max_episode: int | None = Field(
        default=None, description="Spoiler guard maximum episode"
    )
    entity: str | None = Field(
        default=None, description="Target entity or concept focus filter"
    )
    max_results: int = Field(
        default=8, description="Requested number of evidence results"
    )
    include_unscoped: bool = Field(
        default=False,
        description=(
            "Whether to include general background lore unscoped to season/episode"
        ),
    )


class SearchHit(BaseModel):
    """Candidate chunk matching a search query with per-signal retrieval scores."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk: Chunk = Field(description="Retrieved chunk payload")
    score: float = Field(description="Combined fused or reranked score")
    dense_score: float | None = Field(
        default=None, description="Semantic dense vector similarity score"
    )
    sparse_score: float | None = Field(
        default=None, description="Lexical BM25 sparse similarity score"
    )
    rerank_score: float | None = Field(
        default=None, description="Cross-encoder or model reranking score"
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
    score: float = Field(description="Final relevance score")
    dense_score: float | None = Field(
        default=None, description="Dense vector similarity score"
    )
    sparse_score: float | None = Field(default=None, description="Sparse BM25 score")
    rerank_score: float | None = Field(default=None, description="Reranker score")
    provenance: Provenance = Field(
        description="Source provenance for citation and attribution"
    )


class EvidenceBundle(BaseModel):
    """Complete evidence bundle returned for an MCP retrieval tool invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(description="Original user/client query")
    series_id: ValidatedSeriesId = Field(description="Series scope of the query")
    status: Literal["ok", "insufficient", "fallback_used"] = Field(
        default="ok",
        description="Retrieval status: 'ok', 'insufficient', or 'fallback_used'",
    )
    evidence: list[Evidence] = Field(
        default_factory=list[Evidence],
        description="Ranked evidence candidates matching criteria",
    )
    truncated: bool = Field(
        default=False,
        description="Whether response was capped to fit context budget",
    )
    chronology: list[str] = Field(
        default_factory=list[str],
        description="Chronological event sequence or temporal context notes",
    )
    notices: list[str] = Field(
        default_factory=list[str],
        description="Informational notices (e.g. spoiler guards, cache status)",
    )


class WebSearchResult(BaseModel):
    """External web search result item retrieved during fallback."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(description="URL of the web result")
    title: str = Field(description="Title of the web search result")
    snippet: str = Field(description="Textual snippet from search result")
