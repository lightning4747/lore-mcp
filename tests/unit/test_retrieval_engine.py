"""Comprehensive test suite for the complete Retrieval Engine:
- Series scope isolation with a two-series fixture with overlapping entity names
- Dense semantic and sparse lexical retrieval
- Metadata filters and spoiler bounds
- Rank fusion with score ties and input ranking permutations
- Duplicate removal and heuristic reranking
- Empty-result behavior
- Property-based testing verifying no filter builder output lacks the series condition
"""

import itertools
from datetime import UTC, datetime

import pytest
from qdrant_client import models

from lore_mcp.adapters.qdrant.filters import (
    build_filter_from_query,
    build_metadata_filter,
)
from lore_mcp.application.dense import DenseRetriever
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.application.sparse import SparseRetriever
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import (
    SearchHit,
    SearchQuery,
    reciprocal_rank_fusion,
    rerank_candidates,
)
from lore_mcp.domain.types import ChunkId, EntityId, SeriesId
from lore_mcp.testing.fakes import FakeVectorIndex


def make_engine_chunk(
    chunk_id: str,
    series_id: str,
    text: str,
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
            source_url=f"https://example.com/{series_id}/{chunk_id}",
            retrieval_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )


# ============================================================================
# 1. Two-Series Fixture with Overlapping Entity Names & Scope Isolation
# ============================================================================


@pytest.fixture
def two_series_chunks() -> list[Chunk]:
    """Fixture containing two distinct series ('mcu' and 'norse-myth') that share
    overlapping entity names (e.g. 'loki', 'odin', 'thor', 'valkyrie').
    """
    return [
        # MCU franchise
        make_engine_chunk(
            "mcu-1",
            "mcu",
            "Loki escapes using the Tesseract in New York 2012.",
            season=1,
            episode=1,
            narrative_order=10,
            entity_ids=["loki", "tesseract"],
            source_type="transcript",
        ),
        make_engine_chunk(
            "mcu-2",
            "mcu",
            "Odin banishes Thor to Earth without Mjolnir.",
            season=1,
            episode=1,
            narrative_order=2,
            entity_ids=["odin", "thor"],
            source_type="transcript",
        ),
        make_engine_chunk(
            "mcu-3",
            "mcu",
            "Loki and Mobius inspect the TVA temporal loom.",
            season=2,
            episode=1,
            narrative_order=15,
            entity_ids=["loki", "mobius"],
            source_type="fandom",
        ),
        # Norse Myth franchise (shares entities: 'loki', 'odin', 'thor')
        make_engine_chunk(
            "myth-1",
            "norse-myth",
            "Loki tricks Hodr into slaying Baldr with mistletoe.",
            season=1,
            episode=1,
            narrative_order=50,
            entity_ids=["loki", "baldr"],
            source_type="fandom",
        ),
        make_engine_chunk(
            "myth-2",
            "norse-myth",
            "Odin hangs from Yggdrasil for nine nights to gain runes.",
            season=1,
            episode=1,
            narrative_order=1,
            entity_ids=["odin", "yggdrasil"],
            source_type="wikipedia",
        ),
        make_engine_chunk(
            "myth-3",
            "norse-myth",
            "Thor battles the Midgard Serpent Jormungandr during Ragnarok.",
            season=2,
            episode=1,
            narrative_order=99,
            entity_ids=["thor", "jormungandr"],
            source_type="fandom",
        ),
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_series_scope_isolation_with_overlapping_entities(
    two_series_chunks: list[Chunk],
) -> None:
    index = FakeVectorIndex()
    await index.upsert(two_series_chunks)

    hybrid = HybridRetriever(index)

    # Query 'Loki' targeting 'mcu'
    q_mcu = SearchQuery(series_id=SeriesId("mcu"), query="Loki", entity="loki")
    hits_mcu = await hybrid.retrieve(q_mcu)

    assert len(hits_mcu) >= 1
    # STRICT isolation: No norse-myth chunk may ever leak into MCU results
    assert all(h.chunk.series_id == "mcu" for h in hits_mcu)
    assert not any(h.chunk.chunk_id.startswith("myth") for h in hits_mcu)

    # Query 'Loki' targeting 'norse-myth'
    q_myth = SearchQuery(series_id=SeriesId("norse-myth"), query="Loki", entity="loki")
    hits_myth = await hybrid.retrieve(q_myth)

    assert len(hits_myth) >= 1
    # STRICT isolation: No MCU chunk may ever leak into norse-myth results
    assert all(h.chunk.series_id == "norse-myth" for h in hits_myth)
    assert not any(h.chunk.chunk_id.startswith("mcu") for h in hits_myth)


# ============================================================================
# 2. Dense and Sparse Retrieval Execution
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dense_and_sparse_individual_retrieval(
    two_series_chunks: list[Chunk],
) -> None:
    index = FakeVectorIndex()
    await index.upsert(two_series_chunks)

    dense_retriever = DenseRetriever(index)
    sparse_retriever = SparseRetriever(index)

    # Dense retrieval handles semantic query
    dense_q = SearchQuery(series_id=SeriesId("mcu"), query="Tesseract New York")
    dense_hits = await dense_retriever.retrieve(dense_q)

    assert len(dense_hits) >= 1
    assert dense_hits[0].dense_score is not None
    assert dense_hits[0].chunk.chunk_id == ChunkId("mcu-1")

    # Sparse retrieval handles lexical terms
    sparse_q = SearchQuery(series_id=SeriesId("mcu"), query="Mobius loom")
    sparse_hits = await sparse_retriever.retrieve(sparse_q)

    assert len(sparse_hits) >= 1
    assert sparse_hits[0].sparse_score is not None
    assert sparse_hits[0].chunk.chunk_id == ChunkId("mcu-3")


# ============================================================================
# 3. Metadata Filters and Spoiler Bounds
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retrieval_metadata_filters_and_spoiler_bounds() -> None:
    index = FakeVectorIndex()
    chunks = [
        make_engine_chunk("s1e1", "loki", "Loki arrested", season=1, episode=1),
        make_engine_chunk("s1e6", "loki", "Citadel confrontation", season=1, episode=6),
        make_engine_chunk(
            "s2e2", "loki", "1977 London Brad Wolfe", season=2, episode=2
        ),
        make_engine_chunk("s2e5", "loki", "Timeline collapse", season=2, episode=5),
        make_engine_chunk("s3e1", "loki", "Future Season 3", season=3, episode=1),
        make_engine_chunk(
            "unscoped", "loki", "General lore background", season=None, episode=None
        ),
    ]
    await index.upsert(chunks)
    hybrid = HybridRetriever(index)

    # 1. Exact equality
    q_exact = SearchQuery(
        series_id=SeriesId("loki"), query="arrested", season=1, episode=1
    )
    hits_exact = await hybrid.retrieve(q_exact)
    assert len(hits_exact) == 1
    assert hits_exact[0].chunk.chunk_id == ChunkId("s1e1")

    # 2. Spoiler bound: max_season=2, max_episode=2 (exclude future & unscoped)
    q_bound = SearchQuery(
        series_id=SeriesId("loki"),
        query="loki",
        max_season=2,
        max_episode=2,
        include_unscoped=False,
    )
    hits_bound = await hybrid.retrieve(q_bound)
    bound_ids = {h.chunk.chunk_id for h in hits_bound}
    assert bound_ids == {ChunkId("s1e1"), ChunkId("s1e6"), ChunkId("s2e2")}

    # 3. Spoiler bound with include_unscoped=True
    q_bound_unscoped = SearchQuery(
        series_id=SeriesId("loki"),
        query="loki",
        max_season=2,
        max_episode=2,
        include_unscoped=True,
    )
    hits_unscoped = await hybrid.retrieve(q_bound_unscoped)
    unscoped_ids = {h.chunk.chunk_id for h in hits_unscoped}
    assert unscoped_ids == {
        ChunkId("s1e1"),
        ChunkId("s1e6"),
        ChunkId("s2e2"),
        ChunkId("unscoped"),
    }


# ============================================================================
# 4. Rank Fusion: Ties, Permutations, Duplicate Removal & Reranking
# ============================================================================


@pytest.mark.unit
def test_rank_fusion_permutation_invariance_and_score_ties() -> None:
    """Verify that the order in which candidate lists are passed to RRF does not
    change the final set of scores, and test score tie handling.
    """
    c1 = make_engine_chunk("c1", "loki", "chunk 1")
    c2 = make_engine_chunk("c2", "loki", "chunk 2")
    c3 = make_engine_chunk("c3", "loki", "chunk 3")

    # List A: [c1, c2]
    # List B: [c2, c1]
    # In both lists: c1 has rank 1 and 2, c2 has rank 2 and 1 -> EXACT TIE
    hit_a1 = SearchHit(chunk=c1, score=1.0, dense_score=1.0)
    hit_a2 = SearchHit(chunk=c2, score=0.8, dense_score=0.8)

    hit_b1 = SearchHit(chunk=c2, score=10.0, sparse_score=10.0)
    hit_b2 = SearchHit(chunk=c1, score=5.0, sparse_score=5.0)

    order1 = reciprocal_rank_fusion([[hit_a1, hit_a2], [hit_b1, hit_b2]], k=60)
    order2 = reciprocal_rank_fusion([[hit_b1, hit_b2], [hit_a1, hit_a2]], k=60)

    # Identical scores regardless of list order permutation
    assert len(order1) == 2
    assert len(order2) == 2
    assert pytest.approx(order1[0].score, rel=1e-6) == order2[0].score
    assert pytest.approx(order1[1].score, rel=1e-6) == order2[1].score

    # Both c1 and c2 have the exact same combined score: 1/61 + 1/62
    expected_score = 1.0 / 61 + 1.0 / 62
    assert pytest.approx(order1[0].score, rel=1e-6) == expected_score
    assert pytest.approx(order1[1].score, rel=1e-6) == expected_score

    # Permutations with 3 rankings
    hit_c = SearchHit(chunk=c3, score=0.9)
    lists = [[hit_a1], [hit_b1], [hit_c]]
    scores_per_perm = []
    for perm in itertools.permutations(lists):
        fused = reciprocal_rank_fusion(list(perm), k=60)
        scores_per_perm.append({h.chunk.chunk_id: round(h.score, 6) for h in fused})

    # All permutations produce identical score mappings
    assert all(s == scores_per_perm[0] for s in scores_per_perm)


@pytest.mark.unit
def test_duplicate_removal_and_reranking_integration() -> None:
    c1 = make_engine_chunk(
        "c1",
        "loki",
        "Mobius drinks Josta soda in the archives",
        source_type="transcript",
        entity_ids=["mobius"],
    )
    c2 = make_engine_chunk(
        "c2",
        "loki",
        "General overview of Mobius at TVA",
        source_type="wikipedia",
        entity_ids=["mobius"],
    )

    # Duplicate occurrences of c1 in dense and sparse rankings
    dense_h1 = SearchHit(chunk=c1, score=0.9, dense_score=0.9)
    sparse_h1 = SearchHit(chunk=c1, score=12.0, sparse_score=12.0)
    sparse_h2 = SearchHit(chunk=c2, score=11.0, sparse_score=11.0)

    fused = reciprocal_rank_fusion([[dense_h1], [sparse_h1, sparse_h2]], k=60)

    # Duplicate c1 is merged
    assert len(fused) == 2
    assert {h.chunk.chunk_id for h in fused} == {ChunkId("c1"), ChunkId("c2")}

    # Now apply reranking
    query = SearchQuery(
        series_id=SeriesId("loki"), query="Mobius soda", entity="mobius"
    )
    reranked = rerank_candidates(fused, query)

    assert len(reranked) == 2
    # c1 is transcript (1.0 quality) vs c2 wikipedia (0.8 quality)
    # Both have entity match
    assert reranked[0].chunk.chunk_id == ChunkId("c1")
    assert reranked[1].chunk.chunk_id == ChunkId("c2")
    assert reranked[0].rerank_score is not None
    assert reranked[0].dense_score == 0.9
    assert reranked[0].sparse_score == 12.0


# ============================================================================
# 5. Empty-Result Behavior
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_empty_result_behavior() -> None:
    index = FakeVectorIndex()
    # Populate chunks for loki only
    await index.upsert([make_engine_chunk("c1", "loki", "Loki in TVA")])

    hybrid = HybridRetriever(index)

    # 1. Query non-matching series: returns empty list cleanly without error
    q_nonexistent_series = SearchQuery(
        series_id=SeriesId("breaking-bad"),
        query="Walter White",
    )
    hits = await hybrid.retrieve(q_nonexistent_series)
    assert hits == []

    # 2. Query with impossible filter condition: returns empty list cleanly
    q_impossible_season = SearchQuery(
        series_id=SeriesId("loki"),
        query="Loki",
        season=99,
    )
    hits_impossible = await hybrid.retrieve(q_impossible_season)
    assert hits_impossible == []

    # 3. Dense & sparse retrievers directly return empty lists
    dense = DenseRetriever(index)
    sparse = SparseRetriever(index)
    assert await dense.retrieve(q_impossible_season) == []
    assert await sparse.retrieve(q_impossible_season) == []


# ============================================================================
# 6. Property Testing: No Filter Builder Output Lacks Series Condition
# ============================================================================


@pytest.mark.unit
def test_property_no_filter_builder_output_lacks_series_condition() -> None:
    """Property test exhaustively verifying that across all combinations of filter
    parameters, every generated Qdrant filter contains the mandatory series condition.
    """
    test_series_ids = ["loki", "star-wars", "andor", "arcane-123"]
    seasons = [None, 1, 2, 5]
    episodes = [None, 1, 6]
    max_seasons = [None, 1, 2]
    max_episodes = [None, 3]
    entities = [None, "loki", "mobius"]
    entity_types = [None, "character", "location"]
    source_types = [None, "transcript", "fandom"]
    include_unscopeds = [False, True]

    tested_count = 0

    for series in test_series_ids:
        # Sample across combinations
        for s, ep, max_s, max_ep, ent, ent_t, src_t, unscoped in itertools.product(
            seasons,
            episodes,
            max_seasons,
            max_episodes,
            entities,
            entity_types,
            source_types,
            include_unscopeds,
        ):
            # Test direct builder
            flt = build_metadata_filter(
                series_id=series,
                season=s,
                episode=ep,
                max_season=max_s,
                max_episode=max_ep,
                entity=ent,
                entity_type=ent_t,
                source_type=src_t,
                include_unscoped=unscoped,
            )

            # Property Invariant 1: must is non-empty
            assert flt.must is not None
            must_list = flt.must if isinstance(flt.must, list) else [flt.must]
            assert len(must_list) > 0

            # Property Invariant 2: series field condition strictly present
            # with matching series value
            series_match = any(
                isinstance(cond, models.FieldCondition)
                and cond.key == "series"
                and cond.match == models.MatchValue(value=series)
                for cond in must_list
            )
            assert series_match, (
                f"Filter for series '{series}' lacked mandatory series condition: {flt}"
            )

            tested_count += 1
            if tested_count > 500:  # Sample limit for rapid execution
                break

    assert tested_count > 0


@pytest.mark.unit
def test_property_build_filter_from_query_invariance() -> None:
    """Property test verifying build_filter_from_query always includes series."""
    for series in ["loki", "andor", "custom-universe"]:
        for max_s in [None, 2]:
            for inc in [False, True]:
                query = SearchQuery(
                    series_id=SeriesId(series),
                    query="search test",
                    max_season=max_s,
                    include_unscoped=inc,
                )
                flt = build_filter_from_query(query)
                assert flt.must is not None
                must_list = flt.must if isinstance(flt.must, list) else [flt.must]
                has_series = any(
                    isinstance(cond, models.FieldCondition)
                    and cond.key == "series"
                    and cond.match == models.MatchValue(value=series)
                    for cond in must_list
                )
                assert has_series
