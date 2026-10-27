"""Unit tests for chronological context and temporal ordering (Tasks 9.2-9.5)."""

import random
from datetime import UTC, datetime

import pytest

from lore_mcp.application.chronology.builder import DefaultChronologyBuilder
from lore_mcp.domain.chronology import Evidence
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.types import ChunkId, EntityId, SeriesId


@pytest.fixture
def sample_provenance() -> Provenance:
    return Provenance(
        source_type="fandom",
        source_url="https://marvel.fandom.com/wiki/Loki",
        retrieval_timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )


def make_evidence(
    chunk_id: str,
    content: str,
    provenance: Provenance,
    *,
    entity_id: str | None = None,
    season: int | None = None,
    episode: int | None = None,
    scene_index: int | None = None,
    narrative_order: int | None = None,
    narrative_confidence: float | None = None,
    score: float = 0.8,
) -> Evidence:
    return Evidence(
        chunk_id=ChunkId(chunk_id),
        series_id=SeriesId("mcu"),
        entity_id=EntityId(entity_id) if entity_id else None,
        content=content,
        season=season,
        episode=episode,
        scene_index=scene_index,
        narrative_order=narrative_order,
        narrative_confidence=narrative_confidence,
        score=score,
        provenance=provenance,
    )


@pytest.mark.unit
class TestChronologyBuilder:
    """Tests DefaultChronologyBuilder logic across all temporal conditions."""

    def test_empty_evidence_list(self) -> None:
        builder = DefaultChronologyBuilder()
        ctx = builder.build_context([])

        assert ctx.ordering == "unknown"
        assert ctx.groups == []
        assert ctx.timeline_notes == []
        assert ctx.uncertain_ids == []
        assert ctx.conflicts == []
        assert ctx.has_uncertainty is False

    def test_linear_narrative_chronology(self, sample_provenance: Provenance) -> None:
        builder = DefaultChronologyBuilder()
        ev1 = make_evidence(
            "chunk-1",
            "Loki discovers Frost Giant heritage",
            sample_provenance,
            entity_id="loki",
            season=1,
            episode=1,
            narrative_order=10,
        )
        ev2 = make_evidence(
            "chunk-2",
            "Loki attacks New York",
            sample_provenance,
            entity_id="loki",
            season=1,
            episode=2,
            narrative_order=20,
        )
        ev3 = make_evidence(
            "chunk-3",
            "Loki becomes God of Stories",
            sample_provenance,
            entity_id="loki",
            season=2,
            episode=6,
            narrative_order=30,
        )

        ctx = builder.build_context([ev3, ev1, ev2])

        assert ctx.ordering == "narrative"
        assert len(ctx.groups) == 1
        group = ctx.groups[0]
        assert group.key == "loki"
        assert group.ordering == "narrative"
        assert [it.chunk_id for it in group.ordered_items] == [
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ]
        assert group.unordered_remainder == []
        assert ctx.uncertain_ids == []
        assert ctx.conflicts == []
        assert ctx.has_uncertainty is False
        assert len(ctx.timeline_notes) == 3
        assert "order 10" in ctx.timeline_notes[0]

    def test_nonlinear_chronology_flashback_precedence(
        self, sample_provenance: Provenance
    ) -> None:
        builder = DefaultChronologyBuilder()
        # S2E1 features an ancient flashback (narrative order 5)
        ev_flashback = make_evidence(
            "chunk-s2",
            "He Who Remains establishes the Sacred Timeline in distant past",
            sample_provenance,
            entity_id="hwr",
            season=2,
            episode=1,
            narrative_order=5,
        )
        # S1E1 takes place much later (narrative order 25)
        ev_present = make_evidence(
            "chunk-s1",
            "Loki arrives at the TVA reception desk",
            sample_provenance,
            entity_id="hwr",
            season=1,
            episode=1,
            narrative_order=25,
        )

        ctx = builder.build_context([ev_present, ev_flashback])

        assert ctx.ordering == "narrative"
        group = ctx.groups[0]
        assert [it.chunk_id for it in group.ordered_items] == ["chunk-s2", "chunk-s1"]
        assert group.ordered_items[0].season == 2
        assert group.ordered_items[1].season == 1

    def test_partial_narrative_chronology(self, sample_provenance: Provenance) -> None:
        builder = DefaultChronologyBuilder()
        ev_ord1 = make_evidence(
            "c-1", "First", sample_provenance, entity_id="thor", narrative_order=10
        )
        ev_ord2 = make_evidence(
            "c-2", "Second", sample_provenance, entity_id="thor", narrative_order=20
        )
        ev_unord1 = make_evidence(
            "c-3", "Lore 1", sample_provenance, entity_id="thor", narrative_order=None
        )
        ev_unord2 = make_evidence(
            "c-4", "Lore 2", sample_provenance, entity_id="thor", narrative_order=None
        )

        ctx = builder.build_context([ev_unord1, ev_ord2, ev_unord2, ev_ord1])

        assert ctx.ordering == "narrative"
        group = ctx.groups[0]
        assert [it.chunk_id for it in group.ordered_items] == ["c-1", "c-2"]
        assert [it.chunk_id for it in group.unordered_remainder] == ["c-3", "c-4"]
        assert set(ctx.uncertain_ids) == {"c-3", "c-4"}
        assert ctx.has_uncertainty is True
        assert any("[unordered: c-3]" in note for note in ctx.timeline_notes)

    def test_broadcast_fallback_ordering(self, sample_provenance: Provenance) -> None:
        builder = DefaultChronologyBuilder()
        ev_s1e2 = make_evidence(
            "c-s1e2", "Variant", sample_provenance, season=1, episode=2, scene_index=1
        )
        ev_s1e1 = make_evidence(
            "c-s1e1", "Purpose", sample_provenance, season=1, episode=1, scene_index=3
        )
        ev_s2e1 = make_evidence(
            "c-s2e1", "Slipping", sample_provenance, season=2, episode=1, scene_index=0
        )

        ctx = builder.build_context([ev_s2e1, ev_s1e2, ev_s1e1])

        assert ctx.ordering == "as_shown"
        group = ctx.groups[0]
        assert [it.chunk_id for it in group.ordered_items] == [
            "c-s1e1",
            "c-s1e2",
            "c-s2e1",
        ]
        assert group.unordered_remainder == []
        assert ctx.uncertain_ids == []
        assert ctx.has_uncertainty is False

    def test_partial_broadcast_fallback(self, sample_provenance: Provenance) -> None:
        builder = DefaultChronologyBuilder()
        ev_broadcast = make_evidence(
            "c-bcast", "Episode scene", sample_provenance, season=1, episode=1
        )
        ev_nobcast = make_evidence(
            "c-meta", "General character description", sample_provenance
        )

        ctx = builder.build_context([ev_nobcast, ev_broadcast])

        assert ctx.ordering == "as_shown"
        group = ctx.groups[0]
        assert [it.chunk_id for it in group.ordered_items] == ["c-bcast"]
        assert [it.chunk_id for it in group.unordered_remainder] == ["c-meta"]
        assert ctx.uncertain_ids == ["c-meta"]
        assert ctx.has_uncertainty is True

    def test_unknown_chronology_when_metadata_missing(
        self, sample_provenance: Provenance
    ) -> None:
        builder = DefaultChronologyBuilder()
        ev1 = make_evidence("c-1", "Concept description", sample_provenance)
        ev2 = make_evidence("c-2", "Another general fact", sample_provenance)

        ctx = builder.build_context([ev1, ev2])

        assert ctx.ordering == "unknown"
        group = ctx.groups[0]
        assert group.ordered_items == []
        assert [it.chunk_id for it in group.unordered_remainder] == ["c-1", "c-2"]
        assert set(ctx.uncertain_ids) == {"c-1", "c-2"}
        assert ctx.has_uncertainty is True

    def test_conflicting_chronology_detection(
        self, sample_provenance: Provenance
    ) -> None:
        builder = DefaultChronologyBuilder()
        ev_source_a = make_evidence(
            "c-src-a",
            "Event according to Source A",
            sample_provenance,
            entity_id="sylvie",
            season=1,
            episode=2,
            scene_index=1,
            narrative_order=15,
        )
        ev_source_b = make_evidence(
            "c-src-b",
            "Same event according to Source B with conflicting narrative position",
            sample_provenance,
            entity_id="sylvie",
            season=1,
            episode=2,
            scene_index=1,
            narrative_order=45,
        )

        ctx = builder.build_context([ev_source_a, ev_source_b])

        assert ctx.ordering == "narrative"
        assert len(ctx.conflicts) == 1
        expected_msg = "Conflicting narrative orders for sylvie at S1E2: 15 vs 45"
        assert expected_msg in ctx.conflicts[0]
        assert "c-src-b" in ctx.uncertain_ids
        assert ctx.has_uncertainty is True
        # Both records must be preserved
        group = ctx.groups[0]
        all_ids = {it.chunk_id for it in group.ordered_items} | {
            it.chunk_id for it in group.unordered_remainder
        }
        assert all_ids == {"c-src-a", "c-src-b"}
        assert any("[conflict]" in note for note in ctx.timeline_notes)

    def test_multiple_entities_grouped_separately(
        self, sample_provenance: Provenance
    ) -> None:
        builder = DefaultChronologyBuilder()
        ev_mobius = make_evidence(
            "c-mobius",
            "Mobius backstory",
            sample_provenance,
            entity_id="mobius",
            narrative_order=5,
        )
        ev_loki = make_evidence(
            "c-loki",
            "Loki trial",
            sample_provenance,
            entity_id="loki",
            narrative_order=10,
        )
        ev_general = make_evidence(
            "c-tva",
            "TVA rules",
            sample_provenance,
            entity_id=None,
            narrative_order=1,
        )

        ctx = builder.build_context([ev_mobius, ev_general, ev_loki])

        assert len(ctx.groups) == 3
        # Groups are sorted by key alphabetically: "general", "loki", "mobius"
        assert [g.key for g in ctx.groups] == ["general", "loki", "mobius"]
        assert ctx.groups[0].ordered_items[0].chunk_id == "c-tva"
        assert ctx.groups[1].ordered_items[0].chunk_id == "c-loki"
        assert ctx.groups[2].ordered_items[0].chunk_id == "c-mobius"

    def test_permutation_invariance(self, sample_provenance: Provenance) -> None:
        builder = DefaultChronologyBuilder()
        items = [
            make_evidence(
                f"c-{i}",
                f"Content {i}",
                sample_provenance,
                entity_id="entity-a" if i % 2 == 0 else "entity-b",
                season=1 if i < 4 else 2,
                episode=i % 3 + 1,
                narrative_order=i * 10 if i % 3 != 0 else None,
            )
            for i in range(8)
        ]

        baseline_ctx = builder.build_context(items)

        rng = random.Random(42)
        for _ in range(10):
            shuffled = list(items)
            rng.shuffle(shuffled)
            perm_ctx = builder.build_context(shuffled)

            assert perm_ctx.ordering == baseline_ctx.ordering
            assert perm_ctx.uncertain_ids == baseline_ctx.uncertain_ids
            assert perm_ctx.conflicts == baseline_ctx.conflicts
            assert perm_ctx.timeline_notes == baseline_ctx.timeline_notes

            assert len(perm_ctx.groups) == len(baseline_ctx.groups)
            for g_perm, g_base in zip(
                perm_ctx.groups, baseline_ctx.groups, strict=True
            ):
                assert g_perm.key == g_base.key
                assert g_perm.ordering == g_base.ordering
                assert [it.chunk_id for it in g_perm.ordered_items] == [
                    it.chunk_id for it in g_base.ordered_items
                ]
                assert [it.chunk_id for it in g_perm.unordered_remainder] == [
                    it.chunk_id for it in g_base.unordered_remainder
                ]

    def test_tie_breaking_determinism_by_chunk_id(
        self, sample_provenance: Provenance
    ) -> None:
        builder = DefaultChronologyBuilder()
        ev_z = make_evidence(
            "chunk-z",
            "Z content",
            sample_provenance,
            entity_id="ent",
            narrative_order=10,
        )
        ev_a = make_evidence(
            "chunk-a",
            "A content",
            sample_provenance,
            entity_id="ent",
            narrative_order=10,
        )
        ev_m = make_evidence(
            "chunk-m",
            "M content",
            sample_provenance,
            entity_id="ent",
            narrative_order=10,
        )

        ctx = builder.build_context([ev_z, ev_m, ev_a])
        group = ctx.groups[0]
        # Same narrative order -> sorted deterministically by chunk_id
        assert [it.chunk_id for it in group.ordered_items] == [
            "chunk-a",
            "chunk-m",
            "chunk-z",
        ]

    def test_relevance_preservation_alongside_chronological_context(
        self, sample_provenance: Provenance
    ) -> None:
        """Verifies chronology ordering preserves ranked evidence list."""
        builder = DefaultChronologyBuilder()
        # Item 1 has lower score (0.65) but occurs earlier narratively (order 10)
        ev1 = make_evidence(
            "chunk-early",
            "Early backstory",
            sample_provenance,
            narrative_order=10,
            score=0.65,
        )
        # Item 2 has higher score (0.95) but occurs later narratively (order 90)
        ev2 = make_evidence(
            "chunk-late",
            "Climactic battle",
            sample_provenance,
            narrative_order=90,
            score=0.95,
        )

        ranked_evidence = [ev2, ev1]  # Ranked by search relevance

        ctx = builder.build_context(ranked_evidence)

        # 1. Input list was preserved in original order and items
        assert ranked_evidence[0].chunk_id == "chunk-late"
        assert ranked_evidence[1].chunk_id == "chunk-early"
        assert ranked_evidence[0].score == 0.95
        assert ranked_evidence[1].score == 0.65

        # 2. Chronological context has them ordered chronologically
        group = ctx.groups[0]
        assert [it.chunk_id for it in group.ordered_items] == [
            "chunk-early",
            "chunk-late",
        ]

    def test_no_items_invented_or_dropped(self, sample_provenance: Provenance) -> None:
        """Every input evidence chunk must appear in ordered or unordered."""
        builder = DefaultChronologyBuilder()
        ev_items = [
            make_evidence("c1", "text1", sample_provenance, narrative_order=10),
            make_evidence("c2", "text2", sample_provenance, season=1, episode=2),
            make_evidence("c3", "text3", sample_provenance),
            make_evidence("c4", "text4", sample_provenance, narrative_order=5),
        ]

        ctx = builder.build_context(ev_items)
        accounted_chunk_ids: list[str] = []
        for g in ctx.groups:
            accounted_chunk_ids.extend([it.chunk_id for it in g.ordered_items])
            accounted_chunk_ids.extend([it.chunk_id for it in g.unordered_remainder])

        assert sorted(accounted_chunk_ids) == sorted([it.chunk_id for it in ev_items])
