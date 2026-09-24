"""Loads config.yaml once and exposes it as a plain dict."""
from __future__ import annotations

from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

_cached_config: dict | None = None


def load_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> dict:
    global _cached_config
    if _cached_config is not None:
        return _cached_config
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config.yaml not found at {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    _resolve_paths(config)
    _cached_config = config
    return config


def _resolve_paths(config: dict) -> None:
    """Make relative paths absolute, rooted at the project directory."""
    config["database"]["path"] = str(_PROJECT_ROOT / config["database"]["path"])
    for key in ("videos_dir", "outputs_dir", "models_dir"):
        config["paths"][key] = str(_PROJECT_ROOT / config["paths"][key])


def project_root() -> Path:
    return _PROJECT_ROOT
