"""Unit tests for SeriesConfig schema and application loader."""

from pathlib import Path

import pytest

from lore_mcp.application.errors import (
    DataUnavailable,
    IncompatibleData,
    InvalidArgument,
)
from lore_mcp.application.series_loader import (
    load_all_series_configs,
    load_series_config,
)
from lore_mcp.domain.series_config import SeriesConfig, WikiEndpointConfig
from lore_mcp.domain.types import SeriesId


@pytest.mark.unit
def test_load_loki_series_config() -> None:
    config = load_series_config("series/loki.yaml")
    assert config.id == SeriesId("loki")
    assert "Loki" in config.display_names
    assert config.kind == "show"
    assert "MCU Loki" in config.aliases
    assert len(config.wiki_endpoints) == 1
    endpoint = config.wiki_endpoints[0]
    assert endpoint.api_url == "https://marvel.fandom.com/api.php"
    assert endpoint.wiki_type == "fandom"
    assert endpoint.license == "CC-BY-SA-3.0"
    assert config.category_to_entity_type["Characters"] == "character"
    assert "marvel.fandom.com" in config.web_fallback_domain_allowlist
    assert config.curated_timeline_path == "series/timelines/loki_timeline.json"
    assert config.transcripts_dir == "series/transcripts/loki"


@pytest.mark.unit
def test_load_all_series_configs() -> None:
    configs = load_all_series_configs("series")
    assert SeriesId("loki") in configs
    assert configs[SeriesId("loki")].id == "loki"


@pytest.mark.unit
def test_missing_file_raises_data_unavailable() -> None:
    with pytest.raises(DataUnavailable, match="not found"):
        load_series_config("series/nonexistent.yaml")


@pytest.mark.unit
def test_malformed_yaml_raises_incompatible_data(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("id: bad\n  unmatched_indent: [", encoding="utf-8")
    with pytest.raises(IncompatibleData, match="Syntax error parsing YAML"):
        load_series_config(bad_yaml)


@pytest.mark.unit
def test_non_dict_yaml_raises_invalid_argument(tmp_path: Path) -> None:
    scalar_yaml = tmp_path / "scalar.yaml"
    scalar_yaml.write_text("just a plain string", encoding="utf-8")
    with pytest.raises(InvalidArgument, match="must deserialize to a dictionary"):
        load_series_config(scalar_yaml)


@pytest.mark.unit
def test_id_mismatch_raises_invalid_argument(tmp_path: Path) -> None:
    file = tmp_path / "steve.yaml"
    file.write_text(
        """
id: loki
display_names: ["Loki"]
kind: show
wiki_endpoints:
  - api_url: "https://marvel.fandom.com/api.php"
    license: "CC-BY-SA-3.0"
category_to_entity_type:
  "Characters": "character"
""",
        encoding="utf-8",
    )
    with pytest.raises(InvalidArgument, match="Series ID mismatch"):
        load_series_config(file)


@pytest.mark.unit
def test_invalid_kind_validation() -> None:
    sample = """
id: sample
display_names: ["Sample"]
kind: movie
wiki_endpoints:
  - api_url: "https://wiki.org/api.php"
    license: "CC-BY-SA-3.0"
category_to_entity_type:
  "Characters": "character"
"""
    with pytest.raises(ValueError, match="Input should be 'show' or 'anime'"):
        SeriesConfig.from_yaml(sample)


@pytest.mark.unit
def test_empty_wiki_endpoints_validation() -> None:
    sample = """
id: sample
display_names: ["Sample"]
kind: show
wiki_endpoints: []
category_to_entity_type:
  "Characters": "character"
"""
    with pytest.raises(
        ValueError, match="wiki_endpoints must contain at least one endpoint"
    ):
        SeriesConfig.from_yaml(sample)


@pytest.mark.unit
def test_empty_display_names_validation() -> None:
    sample = """
id: sample
display_names: ["   "]
kind: show
wiki_endpoints:
  - api_url: "https://wiki.org/api.php"
    license: "CC-BY-SA-3.0"
category_to_entity_type:
  "Characters": "character"
"""
    with pytest.raises(
        ValueError, match="display_names must contain at least one non-empty name"
    ):
        SeriesConfig.from_yaml(sample)


@pytest.mark.unit
def test_wiki_endpoint_empty_fields_validation() -> None:
    with pytest.raises(ValueError, match="Field must not be empty"):
        WikiEndpointConfig(api_url="   ", license="CC-BY-SA-3.0")

    with pytest.raises(ValueError, match="Field must not be empty"):
        WikiEndpointConfig(api_url="https://wiki.org/api.php", license="  ")


@pytest.mark.unit
def test_extra_fields_forbidden() -> None:
    sample = """
id: sample
display_names: ["Sample"]
kind: show
wiki_endpoints:
  - api_url: "https://wiki.org/api.php"
    license: "CC-BY-SA-3.0"
category_to_entity_type:
  "Characters": "character"
extra_unknown_field: true
"""
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        SeriesConfig.from_yaml(sample)
