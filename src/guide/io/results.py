"""Write generated designs, scores, and run settings."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from guide.types import GenerationResult
from guide.utils import json_ready


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_generation_result(
    result: GenerationResult,
    output_directory: str | Path,
) -> Path:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "designs": result.designs,
        "scores": result.scores,
    }
    if result.predicted_mean is not None:
        payload["predicted_mean"] = result.predicted_mean
    np.savez_compressed(directory / "generation.npz", **payload)
    _write_json(
        directory / "generation_manifest.json",
        {
            "schema_version": 1,
            "method": result.method,
            "target_id": result.target_id,
            "score_name": result.score_name,
            "n_candidates": int(result.designs.shape[0]),
            "design_dimension": int(result.designs.shape[1]),
            "metadata": result.metadata,
        },
    )
    return directory
