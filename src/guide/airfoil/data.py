"""Target loading and MCMC initialization for airfoil generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constraints import LinearInequalityConstraint


def load_targets(
    target_path: str | Path,
    index_path: str | Path | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64] | None]:
    """Load target curves and optional source indices."""

    targets = np.asarray(np.load(target_path, allow_pickle=False), dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 10:
        raise ValueError(f"Expected targets with shape (n_targets, 10), got {targets.shape}.")
    if not np.all(np.isfinite(targets)):
        raise ValueError("Target array contains non-finite values.")
    indices = None
    if index_path is not None:
        indices = np.asarray(np.load(index_path, allow_pickle=False), dtype=np.int64).reshape(-1)
        if indices.size != targets.shape[0]:
            raise ValueError("Target indices and target curves contain different numbers of entries.")
    return targets, indices


def draw_admissible_gaussian_initial_design(
    training_designs: ArrayLike,
    constraint: LinearInequalityConstraint,
    *,
    seed: int = 0,
    max_draws: int = 100_000,
) -> NDArray[np.float64]:
    """Draw a reproducible valid MCMC start from the empirical Gaussian prior.

    This implements the common initialization described for MCMC-BI and
    ABC-MCMC: a Gaussian fitted coordinate-wise to the training designs is used
    only to initialize the chain; the posterior prior remains uniform over the
    admissible domain.
    """

    training = np.asarray(training_designs, dtype=np.float64)
    if training.ndim != 2 or training.shape[1] != constraint.design_dimension:
        raise ValueError("training_designs has an incompatible shape.")
    mean = np.mean(training, axis=0)
    standard_deviation = np.std(training, axis=0, ddof=0)
    if np.any(standard_deviation <= 0):
        raise ValueError("Every training-design coordinate must have positive variance.")
    rng = np.random.default_rng(seed)
    batch_size = min(4096, max_draws)
    drawn = 0
    while drawn < max_draws:
        current = min(batch_size, max_draws - drawn)
        candidates = rng.normal(mean, standard_deviation, size=(current, training.shape[1]))
        valid = constraint.is_valid(candidates)
        if np.any(valid):
            return candidates[int(np.flatnonzero(valid)[0])]
        drawn += current
    raise RuntimeError(f"No admissible Gaussian initialization was found after {max_draws} draws.")
