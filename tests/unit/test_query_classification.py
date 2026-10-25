"""Unit tests for deterministic query classification (Task 8.1 & 8.4)."""

import pytest

from lore_mcp.application.classifier import QueryClassifier
from lore_mcp.domain.entities import SeriesManifest
from lore_mcp.domain.types import SeriesId


@pytest.mark.unit
class TestQueryClassification:
    """Verifies deterministic classification into simple_factual, entity_focused,
    and complex_multi_entity.
    """

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
                "sylvie": "Sylvie Laufeydottir",
                "mobius": "Mobius M. Mobius",
                "ravonna": "Ravonna Renslayer",
                "alioth": "Alioth",
                "thor": "Thor Odinson",
                "tva": "Time Variance Authority",
            },
        )

    # --- Simple Factual Tests ---

    @pytest.mark.parametrize(
        "query",
        [
            "When was Loki arrested by the TVA?",
            "Where is the TVA located?",
            "What episode does Sylvie appear in?",
            "What season did He Who Remains reveal the Citadel?",
            "How many variants of Loki exist?",
            "Is Mobius a human variant?",
            "What is the name of Thor's hammer?",
        ],
    )
    def test_simple_factual_queries(
        self, classifier: QueryClassifier, manifest: SeriesManifest, query: str
    ) -> None:
        classified = classifier.expand_aliases(query, manifest=manifest)
        assert classified.query_class == "simple_factual"

    # --- Entity-Focused Tests ---

    @pytest.mark.parametrize(
        "query",
        [
            "Who is Sylvie?",
            "Tell me about Loki",
            "Background on the TVA",
            "What are the abilities of Loki?",
            "Origin of Sylvie",
            "Profile of Mobius",
            "History of the Time Variance Authority",
            "Loki",
            "Sylvie",
        ],
    )
    def test_entity_focused_queries(
        self, classifier: QueryClassifier, manifest: SeriesManifest, query: str
    ) -> None:
        classified = classifier.expand_aliases(query, manifest=manifest)
        assert classified.query_class == "entity_focused"

    # --- Complex / Multi-Entity Tests ---

    @pytest.mark.parametrize(
        "query",
        [
            "How did Loki and Sylvie enchant Alioth?",
            "Did Mobius ever meet Thor in the TVA?",
            "Why did Ravonna betray Mobius during the trial?",
            "Compare Loki and Thor",
            "What is the difference between Loki and Sylvie?",
            "Relationship between Ravonna and Mobius",
            "Loki vs Sylvie in the Citadel",
            "Who stole the Tesseract and why did he run?",
            "Who created the TVA and what happened to the timeline?",
            "Did Loki escape? Where did he go next?",
            "Loki escaped to the Gobi Desert; Mobius arrested him.",
        ],
    )
    def test_complex_multi_entity_queries(
        self, classifier: QueryClassifier, manifest: SeriesManifest, query: str
    ) -> None:
        classified = classifier.expand_aliases(query, manifest=manifest)
        assert classified.query_class == "complex_multi_entity"

    # --- Intent Preservation Tests ---

    def test_intent_preserved_alongside_query_class(
        self, classifier: QueryClassifier, manifest: SeriesManifest
    ) -> None:
        dialogue_query = "Who said 'I am burdened with glorious purpose'?"
        c_diag = classifier.expand_aliases(dialogue_query, manifest=manifest)
        assert c_diag.intent == "dialogue"

        event_query = "What happened during the battle of New York?"
        c_event = classifier.expand_aliases(event_query, manifest=manifest)
        assert c_event.intent == "event"

        char_query = "Who is the God of Mischief?"
        c_char = classifier.expand_aliases(char_query, manifest=manifest)
        assert c_char.intent == "character"
        assert c_char.query_class == "entity_focused"
