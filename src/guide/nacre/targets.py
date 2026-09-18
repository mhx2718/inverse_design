"""Explicit nacre target selection and response-point masks."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from guide.types import TargetSpecification


def load_nacre_target(path, index, *, tolerance, response_points, mask=None):
    """Load target responses, tolerances, and optional response masks.

    NPZ accepts ``responses``, optional ``target_ids``, ``tolerances`` and
    ``response_masks``. NPY holds only responses. A mask excludes coordinates
    from the response event; it is not a fracture-location feasibility test.
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            responses = np.asarray(data["responses"], dtype=float)
            ids = data["target_ids"] if "target_ids" in data else None
            tolerances = data["tolerances"] if "tolerances" in data else None
            masks = data["response_masks"] if "response_masks" in data else None
            strains = data["strains"] if "strains" in data else None
    else:
        responses = np.asarray(np.load(path, allow_pickle=False), dtype=float)
        ids = tolerances = masks = strains = None
    if responses.ndim == 1:
        responses = responses[None, :]
    if responses.ndim != 2 or responses.shape[1] != response_points:
        raise ValueError(f"Targets must have shape (N, {response_points}), got {responses.shape}.")
    if not 0 <= index < len(responses):
        raise IndexError(f"Target index {index} outside [0, {len(responses)-1}].")
    selected_tolerance = tolerance
    if selected_tolerance is None:
        if tolerances is None:
            raise ValueError("Set experiment.tolerance or provide per-target tolerances in the NPZ.")
        if tolerances.shape == responses.shape:
            selected_tolerance = tolerances[index]
        elif tolerances.shape == (len(responses),):
            selected_tolerance = float(tolerances[index])
        else:
            raise ValueError("NPZ tolerances must have shape (N,) or (N,T).")
    selected_mask = mask
    if selected_mask is None and masks is not None:
        if masks.shape != responses.shape:
            raise ValueError("response_masks must have the same shape as responses.")
        selected_mask = masks[index]
    target_id = str(ids[index]) if ids is not None else f"nacre-target-{index:03d}"
    if not target_id or target_id in {".", ".."} or any(c in target_id for c in ("/", "\\", "\x00")):
        raise ValueError("target_ids must be nonempty plain names without path separators.")
    target = TargetSpecification(
        response=responses[index], tolerance=selected_tolerance,
        target_id=target_id,
        metadata={"source": str(path), "index": index},
        response_mask=selected_mask,
    )
    return target, strains
