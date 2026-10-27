"""Unit tests for deterministic alias expansion (Task 8.2 & 8.4)."""

import pytest

from lore_mcp.application.classifier import QueryClassifier
from lore_mcp.domain.entities import AliasEntry, AliasTable, SeriesManifest
from lore_mcp.domain.types import EntityId, SeriesId


@pytest.mark.unit
class TestAliasExpansion:
    """Verifies lexical alias expansion, dense preservation, and isolation."""

    @pytest.fixture
    def classifier(self) -> QueryClassifier:
        return QueryClassifier()

    @pytest.fixture
    def manifest(self) -> SeriesManifest:
        return SeriesManifest(
            series_id=SeriesId("loki"),
            display_name="Loki",
            kind="show",
            aliases=["MCU Loki"],
            entity_aliases={
                "god of mischief": "Loki",
                "goddess of mischief": "Sylvie",
                "he": "He Who Remains",
                "kang": "He Who Remains",
                "space stone": "Tesseract",
            },
        )

    def test_alias_expanded_in_sparse_query_only(
        self, classifier: QueryClassifier, manifest: SeriesManifest
    ) -> None:
        raw = "When did the God of Mischief steal the Space Stone?"
        classified = classifier.expand_aliases(raw, manifest=manifest)

        # Dense query MUST be identical to raw query
        assert classified.raw_query == raw

        # Sparse query must contain the original query text plus canonical names
        assert classified.sparse_query.startswith(raw)
        assert "Loki" in classified.sparse_query
        assert "Tesseract" in classified.sparse_query
        assert "god of mischief" in classified.matched_aliases
        assert "space stone" in classified.matched_aliases

    def test_canonical_name_not_duplicated_if_already_present(
        self, classifier: QueryClassifier, manifest: SeriesManifest
    ) -> None:
        raw = "When did Loki, the God of Mischief, escape?"
        classified = classifier.expand_aliases(raw, manifest=manifest)

        # "Loki" is already present in raw query, so sparse_query should equal raw
        assert classified.raw_query == raw
        assert classified.sparse_query == raw
        assert "god of mischief" in classified.matched_aliases

    def test_word_boundary_avoids_unrelated_terminology(
        self, classifier: QueryClassifier, manifest: SeriesManifest
    ) -> None:
        # "he" is an alias for "He Who Remains", but should not match "The" or "Hero"
        raw = "The hero arrived in the Citadel."
        classified = classifier.expand_aliases(raw, manifest=manifest)

        assert "he" not in classified.matched_aliases
        assert "He Who Remains" not in classified.sparse_query
        assert classified.sparse_query == raw

    def test_unknown_alias_passes_through_unchanged(
        self, classifier: QueryClassifier, manifest: SeriesManifest
    ) -> None:
        raw = "What is the secret of the mysterious chronomonitor?"
        classified = classifier.expand_aliases(raw, manifest=manifest)

        assert classified.raw_query == raw
        assert classified.sparse_query == raw
        assert classified.matched_aliases == []
        assert classified.detected_entities == []

    def test_ambiguous_alias_expands_all_canonical_targets(
        self, classifier: QueryClassifier
    ) -> None:
        # An alias mapping to multiple variants / entities
        manifest = SeriesManifest(
            series_id=SeriesId("loki"),
            display_name="Loki",
            kind="show",
            entity_aliases={
                "variant": "Loki",
            },
        )
        alias_table = AliasTable(
            series_id=SeriesId("loki"),
            entries=[
                AliasEntry(
                    lookup_key="variant",
                    entity_id=EntityId("sylvie"),
                    canonical_name="Sylvie",
                    entity_type="character",
                )
            ],
        )

        raw = "Where did the variant hide?"
        classified = classifier.expand_aliases(
            raw, manifest=manifest, alias_table=alias_table
        )

        assert "Loki" in classified.sparse_query
        assert "Sylvie" in classified.sparse_query
        assert classified.raw_query == raw

    def test_alias_table_integration(self, classifier: QueryClassifier) -> None:
        alias_table = AliasTable(
            series_id=SeriesId("loki"),
            entries=[
                AliasEntry(
                    lookup_key="agent mobius",
                    entity_id=EntityId("mobius"),
                    canonical_name="Mobius M. Mobius",
                    entity_type="character",
                )
            ],
        )
        raw = "Who interrogated Agent Mobius?"
        classified = classifier.expand_aliases(raw, alias_table=alias_table)

        assert classified.raw_query == raw
        assert "Mobius M. Mobius" in classified.sparse_query
        assert "agent mobius" in classified.matched_aliases

    def test_zero_llm_or_external_network_calls(
        self, classifier: QueryClassifier, manifest: SeriesManifest
    ) -> None:
        # Classification and expansion must be synchronous, in-memory, deterministic
        raw = "Where was the God of Mischief born?"
        classified1 = classifier.expand_aliases(raw, manifest=manifest)
        classified2 = classifier.expand_aliases(raw, manifest=manifest)

        assert classified1 == classified2
        assert isinstance(classified1.sparse_query, str)
        assert isinstance(classified1.query_class, str)
