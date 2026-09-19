"""Per-design airfoil data and MCMC initialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constraints import LinearInequalityConstraint


@dataclass(frozen=True)
class AirfoilSplit:
    """Model data and retained geometry/response records for one split."""

    latent: NDArray[np.float64]
    response: NDArray[np.float64]
    geometry: NDArray[np.float64]
    geometry_response: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.latent.ndim != 2 or self.latent.shape[1] != 16:
            raise ValueError("latent must have shape (n_designs, 16).")
        count = self.latent.shape[0]
        for name, shape in (
            ("latent", (count, 16)),
            ("response", (count, 10)),
            ("geometry", (count, 192, 2)),
            ("geometry_response", (count, 10)),
        ):
            values = getattr(self, name)
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain finite values with shape {shape}.")
        if count == 0:
            raise ValueError("Airfoil splits must contain at least one design.")


@dataclass(frozen=True)
class AirfoilDataset:
    """Compact splits, shared AoAs, and indices selecting geometry test responses."""

    train: AirfoilSplit
    val: AirfoilSplit
    test: AirfoilSplit
    angles_of_attack: NDArray[np.float64]
    target_indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        if self.angles_of_attack.shape != (10,) or not np.all(
            np.isfinite(self.angles_of_attack)
        ):
            raise ValueError("angles_of_attack must contain 10 finite angles in degrees.")
        indices = self.target_indices
        if indices.ndim != 1 or indices.size == 0 or not np.issubdtype(
            indices.dtype, np.integer
        ):
            raise ValueError("target_indices must be a nonempty one-dimensional integer array.")
        if np.any(indices < 0) or np.any(indices >= self.test.geometry_response.shape[0]):
            raise ValueError("target_indices contains an out-of-range geometry test index.")

    @classmethod
    def load(cls, path: str | Path) -> AirfoilDataset:
        with np.load(path, allow_pickle=False) as data:
            splits = {
                split: AirfoilSplit(**{
                    name: np.asarray(data[f"{name}_{split}"], dtype=np.float64)
                    for name in ("latent", "response", "geometry", "geometry_response")
                })
                for split in ("train", "val", "test")
            }
            return cls(
                **splits,
                angles_of_attack=np.asarray(data["angles_of_attack"], dtype=np.float64),
                target_indices=data["target_indices"],
            )


def load_targets(path: str | Path) -> NDArray[np.float64]:
    """Load target response curves from a NumPy array of shape (n_targets, 10)."""
    targets = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if targets.ndim != 2 or targets.shape[0] == 0 or targets.shape[1] != 10:
        raise ValueError("Airfoil targets must have shape (n_targets, 10).")
    if not np.all(np.isfinite(targets)):
        raise ValueError("Airfoil targets must contain only finite values.")
    return targets


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
