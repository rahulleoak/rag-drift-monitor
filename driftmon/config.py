"""Load the project configuration from ``config/config.yaml``.

All paths in the config are relative to the repository root; ``resolve_path``
turns them into absolute paths so scripts work from any directory.
Schema validation (pydantic) is added once the config stabilises.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config file and return it as a dict.

    Args:
        path: config file to read; defaults to ``config/config.yaml``.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    """Return ``path`` as an absolute path, treating relative paths as
    relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
