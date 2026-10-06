"""Application loader for series configuration files."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from lore_mcp.application.errors import (
    DataUnavailable,
    IncompatibleData,
    InvalidArgument,
)
from lore_mcp.domain.series_config import SeriesConfig
from lore_mcp.domain.types import SeriesId


def load_series_config(path: Path | str) -> SeriesConfig:
    """Load and validate a SeriesConfig from a YAML file path.

    Raises:
        DataUnavailable: If the config file does not exist.
        IncompatibleData: If the YAML is unparseable or has an invalid structure.
        InvalidArgument: If validation constraints are violated or filename mismatches.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise DataUnavailable(
            f"Series configuration file not found at '{file_path}'",
            resource_id=file_path.name,
        )

    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DataUnavailable(
            f"Cannot read series configuration file '{file_path}': {exc}",
            resource_id=file_path.name,
        ) from exc

    try:
        config = SeriesConfig.from_yaml(content)
    except yaml.YAMLError as exc:
        raise IncompatibleData(
            f"Syntax error parsing YAML in '{file_path}': {exc}"
        ) from exc
    except (ValidationError, ValueError) as exc:
        raise InvalidArgument(
            f"Validation failed for series configuration in '{file_path}': {exc}",
            argument_name="series_config",
        ) from exc

    # Enforce file name matches series id (e.g. loki.yaml -> id: loki)
    expected_id = file_path.stem.lower()
    if config.id != expected_id:
        raise InvalidArgument(
            f"Series ID mismatch in '{file_path}': file stem is '{expected_id}', "
            f"but config declared id '{config.id}'",
            argument_name="id",
        )

    return config


def load_all_series_configs(
    directory: Path | str = "series",
) -> dict[SeriesId, SeriesConfig]:
    """Scan and load all .yaml/.yml series configs from a directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return {}

    configs: dict[SeriesId, SeriesConfig] = {}
    for entry in sorted(dir_path.iterdir()):
        if entry.is_file() and entry.suffix.lower() in {".yaml", ".yml"}:
            cfg = load_series_config(entry)
            configs[cfg.id] = cfg

    return configs
