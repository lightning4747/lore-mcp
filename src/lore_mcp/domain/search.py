"""Domain models for lore search queries, retrieval hits, and evidence bundles."""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lore_mcp.domain.chronology import Evidence, TemporalContext
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.types import ChunkId, ValidatedSeriesId


class SearchQuery(BaseModel):
    """Retrieval search query locked to a required series scope."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: ValidatedSeriesId = Field(
        description="Target series identifier (mandatory scope filter)"
    )
    query: str = Field(description="Search query text")
    sparse_query: str | None = Field(
        default=None,
        description=(
            "Expanded lexical query with canonical entity names for sparse retrieval"
        ),
    )
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
    entity_type: str | None = Field(
        default=None, description="Entity type classification filter"
    )
    source_type: str | None = Field(
        default=None, description="Source provenance type filter"
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


class DialogueSnippet(BaseModel):
    """Concise dialogue quotation snippet with episode and timecode locators."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str = Field(description="Short dialogue quotation snippet")
    speaker: str | None = Field(
        default=None, description="Identified speaker name if known"
    )
    season: int | None = Field(default=None, description="Season number locator")
    episode: int | None = Field(default=None, description="Episode number locator")
    timestamp: str | None = Field(
        default=None, description="Formatted timecode or timestamp locator"
    )
    score: float = Field(default=1.0, description="Retrieval relevance score")
    chunk_id: ChunkId = Field(
        description="Source transcript chunk identifier for full expansion"
    )
    provenance: Provenance = Field(description="Source attribution and citation")


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
    temporal_context: TemporalContext | None = Field(
        default=None,
        description="Structured chronological organization of retrieved evidence",
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
    retrieval_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the web search result was retrieved",
    )


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[SearchHit]],
    *,
    k: int = 60,
    max_results: int | None = None,
) -> list[SearchHit]:
    """Fuse multiple ranked candidate lists using Reciprocal Rank Fusion [S3].

    Computes RRF score for each unique chunk:
        RRF_score(d) = sum_{list in ranked_lists} (1 / (k + rank_{list}(d)))
    where rank starts at 1.

    Invariants:
    - Pure domain function with default k=60.
    - Removes duplicate chunks across rankings by chunk_id.
    - Preserves per-signal scores (dense_score, sparse_score, rerank_score)
      present on any occurrence of the candidate.
    - Preserves the full Chunk payload.
    - Returns candidates sorted strictly by fused score descending.
    """
    if k <= 0:
        raise ValueError(f"RRF smoothing parameter k must be positive, got {k}")

    rrf_scores: dict[ChunkId, float] = {}
    merged_hits: dict[ChunkId, SearchHit] = {}

    for ranking in ranked_lists:
        seen_in_ranking: set[ChunkId] = set()
        for rank_idx, hit in enumerate(ranking, start=1):
            cid = hit.chunk.chunk_id
            if cid in seen_in_ranking:
                continue
            seen_in_ranking.add(cid)

            rrf_score = 1.0 / (k + rank_idx)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score

            existing = merged_hits.get(cid)
            if existing is None:
                merged_hits[cid] = hit
            else:
                # Merge per-signal scores from other lists
                dense_score = (
                    hit.dense_score
                    if hit.dense_score is not None
                    else existing.dense_score
                )
                sparse_score = (
                    hit.sparse_score
                    if hit.sparse_score is not None
                    else existing.sparse_score
                )
                rerank_score = (
                    hit.rerank_score
                    if hit.rerank_score is not None
                    else existing.rerank_score
                )
                merged_hits[cid] = SearchHit(
                    chunk=existing.chunk,
                    score=0.0,
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    rerank_score=rerank_score,
                )

    fused_candidates: list[SearchHit] = []
    for cid, combined_score in rrf_scores.items():
        base_hit = merged_hits[cid]
        fused_candidates.append(
            SearchHit(
                chunk=base_hit.chunk,
                score=combined_score,
                dense_score=base_hit.dense_score,
                sparse_score=base_hit.sparse_score,
                rerank_score=base_hit.rerank_score,
            )
        )

    # Sort descending by fused score
    fused_candidates.sort(key=lambda h: h.score, reverse=True)

    if max_results is not None:
        fused_candidates = fused_candidates[:max_results]

    return fused_candidates


# Default source quality tier weights (primary canonical sources prioritized)
SOURCE_QUALITY_WEIGHTS: dict[str, float] = {
    "transcript": 1.0,
    "fandom": 0.9,
    "wikipedia": 0.8,
    "web": 0.6,
}


def rerank_candidates(
    candidates: Sequence[SearchHit],
    query: SearchQuery,
    *,
    entity_match_boost: float = 0.2,
    source_quality_weights: dict[str, float] | None = None,
    chronology_tie_break: bool = True,
    max_results: int | None = None,
) -> list[SearchHit]:
    """Rerank fused candidates using heuristic signals [S3].

    Reranking signals:
    1. Base fused / relevance score.
    2. Entity/concept match: boosted when candidate explicitly associates with
       the target query entity or concept terms.
    3. Source quality: weighted by provenance source tier
       (transcript > fandom > wiki > web).
    4. Chronology: strictly used as a secondary tie-breaker (never displaces relevance).

    The returned candidates have:
    - `rerank_score`: computed heuristic reranking score.
    - `score`: updated to rerank_score.
    - `dense_score` & `sparse_score`: fully preserved.
    """
    if not candidates:
        return []

    weights = source_quality_weights or SOURCE_QUALITY_WEIGHTS
    query_entity_str = str(query.entity).lower() if query.entity else None

    # Lightweight tokenization of query for concept matching
    query_tokens = {
        token.lower()
        for token in query.query.replace("-", " ").replace("_", " ").split()
        if len(token) > 2
    }

    scored_candidates: list[tuple[SearchHit, float, tuple[int, int, int]]] = []

    for hit in candidates:
        base_score = hit.score

        # 1. Entity / concept match bonus
        entity_bonus = 0.0
        chunk_entity_strings = {str(e).lower() for e in hit.chunk.entity_ids}

        # Check explicit query entity match
        if query_entity_str and (
            query_entity_str in chunk_entity_strings
            or any(query_entity_str in e for e in chunk_entity_strings)
        ):
            entity_bonus += entity_match_boost
        elif query_tokens and chunk_entity_strings:
            # Concept match: token overlap with chunk entities
            matching_tokens = query_tokens.intersection(chunk_entity_strings)
            if matching_tokens:
                entity_bonus += entity_match_boost * (
                    len(matching_tokens) / len(query_tokens)
                )

        # 2. Source quality factor
        src_type = hit.chunk.provenance.source_type.lower()
        quality_factor = weights.get(src_type, 0.7)

        # Combined score: base_score * (1 + entity_bonus) * quality_factor
        rerank_score = (base_score * (1.0 + entity_bonus)) * quality_factor

        # 3. Chronology tie-breaker key (season, episode, narrative_order/scene_index)
        # Using 0 for None so earlier lore is deterministically ordered
        season_key = hit.chunk.season if hit.chunk.season is not None else 0
        episode_key = hit.chunk.episode if hit.chunk.episode is not None else 0
        order_key = (
            hit.chunk.narrative_order
            if hit.chunk.narrative_order is not None
            else (hit.chunk.scene_index if hit.chunk.scene_index is not None else 0)
        )
        tie_break_key = (season_key, episode_key, order_key)

        reranked_hit = SearchHit(
            chunk=hit.chunk,
            score=rerank_score,
            dense_score=hit.dense_score,
            sparse_score=hit.sparse_score,
            rerank_score=rerank_score,
        )
        scored_candidates.append((reranked_hit, rerank_score, tie_break_key))

    # Sort descending by rerank_score (rounded to 6 decimals to allow tie-breaking)
    # followed by chronological tie-breaker ascending (or descending)
    if chronology_tie_break:
        scored_candidates.sort(
            key=lambda item: (-round(item[1], 6), item[2]),
        )
    else:
        scored_candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

    result = [item[0] for item in scored_candidates]
    limit = max_results if max_results is not None else query.max_results
    return result[:limit]
