from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> str:
    """Return an absolute path inside the project root."""
    return str(PROJECT_ROOT.joinpath(*parts))


def resolve_path(path_value: str | os.PathLike[str], base_dir: str | os.PathLike[str] | None = None) -> str:
    """Expand a user/env path and resolve it against a base directory if needed."""
    expanded = Path(os.path.expandvars(os.path.expanduser(str(path_value))))
    if expanded.is_absolute():
        return str(expanded)
    base = Path(base_dir) if base_dir is not None else PROJECT_ROOT
    return str((base / expanded).resolve())


def _is_path_key(key: str) -> bool:
    return (
        key in {"dir", "project_root", "results_dir", "logs_dir"}
        or key.endswith("_path")
        or key.endswith("_dir")
    )


def resolve_config_paths(value: Any, base_dir: str | os.PathLike[str] | None = None) -> Any:
    """Recursively resolve known path fields inside a config object."""
    base = Path(base_dir) if base_dir is not None else PROJECT_ROOT

    if isinstance(value, dict):
        resolved = {}
        for key, item in value.items():
            if isinstance(item, str) and _is_path_key(key):
                resolved[key] = resolve_path(item, base)
            else:
                resolved[key] = resolve_config_paths(item, base)
        return resolved

    if isinstance(value, list):
        return [resolve_config_paths(item, base) for item in value]

    return value


def load_yaml_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the main YAML config and normalize all path-like fields."""
    import yaml

    cfg_path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    config = resolve_config_paths(config, PROJECT_ROOT)
    config.setdefault("paths", {})
    config["paths"]["project_root"] = project_path()
    config["paths"]["logs_dir"] = resolve_path(config["paths"].get("logs_dir", "logs"), PROJECT_ROOT)
    return config
