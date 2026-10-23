"""Unit tests for golden benchmark dataset schema and integrity (Phase 11.1 & 11.2)."""

from pathlib import Path

import yaml

from lore_mcp.domain.evaluation import CaseType, GoldenCase


def test_golden_loki_yaml_exists() -> None:
    path = Path("eval/golden/loki.yaml")
    assert path.exists(), f"Golden dataset file {path} must exist"


def test_golden_loki_cases_valid() -> None:
    path = Path("eval/golden/loki.yaml")
    with open(path, encoding="utf-8") as f:
        raw_cases = yaml.safe_load(f)

    assert isinstance(raw_cases, list), "YAML root must be a list of cases"
    assert len(raw_cases) >= 40, f"Expected at least 40 cases, got {len(raw_cases)}"

    cases: list[GoldenCase] = []
    case_ids: set[str] = set()
    category_counts: dict[str, int] = {}

    required_types: set[CaseType] = {
        "simple_fact",
        "alias_entity",
        "dialogue",
        "multi_part",
        "chronology",
        "spoiler_bound",
        "unanswerable",
    }

    for item in raw_cases:
        assert isinstance(item, dict)
        case = GoldenCase.model_validate(item)
        cases.append(case)

        # ID uniqueness
        assert case.case_id not in case_ids, f"Duplicate case_id: {case.case_id}"
        case_ids.add(case.case_id)

        # Series match
        assert case.series == "loki"

        # Count categories
        category_counts[case.case_type] = category_counts.get(case.case_type, 0) + 1

        # Check for banned chunk ID attributes
        assert not hasattr(case, "expected_chunk_ids")
        raw_evidence = item.get("expected_evidence", {})
        assert "chunk_id" not in raw_evidence
        assert "chunk_ids" not in raw_evidence

        # Specific constraints per type
        if case.case_type == "spoiler_bound":
            assert (
                case.season_bound is not None or case.episode_bound is not None
            ), f"Spoiler case {case.case_id} must specify season_bound or episode_bound"

        if case.case_type == "unanswerable":
            assert len(case.expected_evidence.entity_ids) == 0
            assert len(case.expected_evidence.scenes) == 0
            assert len(case.expected_evidence.dialogue_quotes) == 0
            assert len(case.expected_evidence.chronological_order) == 0

        if case.case_type == "chronology":
            assert len(case.expected_evidence.chronological_order) >= 2, (
                f"Chronology case {case.case_id} must have >= 2 sequence steps"
            )

    # Verify all 7 categories exist and each has >= 4 cases
    assert set(category_counts.keys()) == required_types
    for cat, count in category_counts.items():
        assert count >= 4, f"Category '{cat}' has only {count} cases, required >= 4"
