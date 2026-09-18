"""Small configuration helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping, optionally extending a file relative to this one."""
    return _load_yaml(Path(path).resolve(), ())


def _load_yaml(path: Path, parents: tuple[Path, ...]) -> dict[str, Any]:
    if path in parents:
        raise ValueError(f"Cyclic configuration inheritance: {path}")
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the root of {path}, got {type(data).__name__}.")
    inherited = data.pop("extends", None)
    if inherited is None:
        return data
    if not isinstance(inherited, str):
        raise ValueError(f"extends must be a filename in {path}.")
    base_path = (path.parent / inherited).resolve()
    base = _load_yaml(base_path, (*parents, path))
    # Keep inherited project-relative paths anchored to the base config.
    if "project_root" in base and "project_root" not in data:
        root = (base_path.parent / base["project_root"]).resolve()
        base["project_root"] = str(root)
    return deep_update(base, data)


def deep_update(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result
