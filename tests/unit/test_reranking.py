"""Unit tests for heuristic candidate reranking [S3],
evaluating entity/concept match, source quality weights, and chronological tie-breaking.
"""

from datetime import UTC, datetime

import pytest

from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import (
    SearchHit,
    SearchQuery,
    rerank_candidates,
)
from lore_mcp.domain.types import ChunkId, EntityId, SeriesId
from lore_mcp.testing.fakes import FakeVectorIndex


def make_chunk(
    chunk_id: str,
    series_id: str = "loki",
    text: str = "Sample chunk text",
    season: int | None = 1,
    episode: int | None = 1,
    narrative_order: int | None = 10,
    scene_index: int | None = 1,
    entity_ids: list[str] | None = None,
    source_type: str = "transcript",
) -> Chunk:
    return Chunk(
        chunk_id=ChunkId(chunk_id),
        series_id=SeriesId(series_id),
        text=text,
        season=season,
        episode=episode,
        narrative_order=narrative_order,
        scene_index=scene_index,
        content_hash=f"hash_{chunk_id}",
        entity_ids=[EntityId(e) for e in (entity_ids or [])],
        provenance=Provenance(
            source_type=source_type,
            source_url="https://example.com",
            retrieval_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )


def make_hit(
    chunk_id: str,
    score: float = 0.5,
    dense_score: float | None = 0.5,
    sparse_score: float | None = 0.5,
    season: int | None = 1,
    episode: int | None = 1,
    narrative_order: int | None = 10,
    entity_ids: list[str] | None = None,
    source_type: str = "transcript",
) -> SearchHit:
    chunk = make_chunk(
        chunk_id=chunk_id,
        season=season,
        episode=episode,
        narrative_order=narrative_order,
        entity_ids=entity_ids,
        source_type=source_type,
    )
    return SearchHit(
        chunk=chunk,
        score=score,
        dense_score=dense_score,
        sparse_score=sparse_score,
    )


# ============================================================================
# 1. Heuristic Reranker Unit Tests: Entity / Concept Match & Source Quality
# ============================================================================


@pytest.mark.unit
def test_rerank_empty_candidates() -> None:
    query = SearchQuery(series_id=SeriesId("loki"), query="test")
    assert rerank_candidates([], query) == []


@pytest.mark.unit
def test_rerank_entity_match_boost() -> None:
    # Two candidates with equal initial score and same source type.
    # Candidate c1 has target entity 'mobius', candidate c2 does not.
    query = SearchQuery(
        series_id=SeriesId("loki"),
        query="interrogation scene",
        entity="mobius",
    )
    h1 = make_hit("c1", score=0.5, entity_ids=["mobius"])
    h2 = make_hit("c2", score=0.5, entity_ids=["hunter-b15"])

    reranked = rerank_candidates([h2, h1], query, entity_match_boost=0.2)

    assert len(reranked) == 2
    # c1 boosted ahead of c2
    assert reranked[0].chunk.chunk_id == ChunkId("c1")
    assert reranked[1].chunk.chunk_id == ChunkId("c2")
    assert reranked[0].score > reranked[1].score
    assert reranked[0].rerank_score is not None
    # Per-signal scores preserved
    assert reranked[0].dense_score == 0.5
    assert reranked[0].sparse_score == 0.5


@pytest.mark.unit
def test_rerank_source_quality_tier_prioritization() -> None:
    # Candidates with equal base scores and no entity match.
    # Quality weights: transcript (1.0) > fandom (0.9) > wikipedia (0.8) > web (0.6)
    query = SearchQuery(series_id=SeriesId("loki"), query="Time-Keepers origin")
    h_web = make_hit("c_web", score=0.5, source_type="web")
    h_fandom = make_hit("c_fandom", score=0.5, source_type="fandom")
    h_transcript = make_hit("c_transcript", score=0.5, source_type="transcript")
    h_wiki = make_hit("c_wiki", score=0.5, source_type="wikipedia")

    reranked = rerank_candidates(
        [h_web, h_fandom, h_transcript, h_wiki],
        query,
    )

    ids = [h.chunk.chunk_id for h in reranked]
    assert ids == [
        ChunkId("c_transcript"),
        ChunkId("c_fandom"),
        ChunkId("c_wiki"),
        ChunkId("c_web"),
    ]
    # Verify scores follow source quality order
    scores = [h.score for h in reranked]
    assert scores[0] > scores[1] > scores[2] > scores[3]


@pytest.mark.unit
def test_rerank_chronology_strictly_as_tie_break() -> None:
    query = SearchQuery(series_id=SeriesId("loki"), query="multiverse events")

    # Case 1: Relevance is different -> relevance strictly wins, chronology ignored
    h_future_high_score = make_hit(
        "c_future", score=0.9, season=2, episode=6, narrative_order=100
    )
    h_past_low_score = make_hit(
        "c_past", score=0.4, season=1, episode=1, narrative_order=1
    )

    reranked_case1 = rerank_candidates(
        [h_past_low_score, h_future_high_score],
        query,
        chronology_tie_break=True,
    )
    # Higher score candidate remains #1 despite being later in chronology
    assert reranked_case1[0].chunk.chunk_id == ChunkId("c_future")
    assert reranked_case1[1].chunk.chunk_id == ChunkId("c_past")

    # Case 2: Exact tie in score and source quality -> chronology breaks tie
    # c_s1e1 (earlier) vs c_s2e6 (later)
    h_tie_early = make_hit(
        "c_early",
        score=0.5,
        season=1,
        episode=1,
        narrative_order=5,
        source_type="transcript",
    )
    h_tie_late = make_hit(
        "c_late",
        score=0.5,
        season=2,
        episode=6,
        narrative_order=90,
        source_type="transcript",
    )

    reranked_case2 = rerank_candidates(
        [h_tie_late, h_tie_early],
        query,
        chronology_tie_break=True,
    )
    # Earlier chronological event wins the tie-break
    assert reranked_case2[0].chunk.chunk_id == ChunkId("c_early")
    assert reranked_case2[1].chunk.chunk_id == ChunkId("c_late")


@pytest.mark.unit
def test_rerank_preserves_per_signal_scores() -> None:
    query = SearchQuery(series_id=SeriesId("loki"), query="test")
    hit = make_hit(
        "c1",
        score=0.7,
        dense_score=0.88,
        sparse_score=14.2,
        source_type="transcript",
    )

    reranked = rerank_candidates([hit], query)
    assert len(reranked) == 1
    assert reranked[0].dense_score == 0.88
    assert reranked[0].sparse_score == 14.2
    assert reranked[0].rerank_score is not None
    assert reranked[0].score == reranked[0].rerank_score


# ============================================================================
# 2. Integration with HybridRetriever
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hybrid_retriever_applies_reranking() -> None:
    fake_index = FakeVectorIndex()
    retriever = HybridRetriever(fake_index, enable_rerank=True)

    # c1 is from fandom, c2 is transcript
    c1 = make_chunk(
        "c1",
        text="He Who Remains explains the Sacred Timeline",
        source_type="fandom",
        entity_ids=["he-who-remains"],
    )
    c2 = make_chunk(
        "c2",
        text="He Who Remains offers a choice to Loki and Sylvie",
        source_type="transcript",
        entity_ids=["he-who-remains"],
    )
    await fake_index.upsert([c1, c2])

    query = SearchQuery(
        series_id=SeriesId("loki"),
        query="Sacred Timeline choice",
        entity="he-who-remains",
    )

    hits = await retriever.retrieve(query)
    assert len(hits) == 2
    # Both have rerank_score set
    for h in hits:
        assert h.rerank_score is not None
        assert h.score == h.rerank_score
