"""Linearized geometric admissibility constraints in standardized PCA space."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class LinearInequalityConstraint:
    """Constraint of the form ``A @ z <= b``."""

    matrix: NDArray[np.float64]
    bound: NDArray[np.float64]
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        self.bound = np.asarray(self.bound, dtype=np.float64).reshape(-1)
        if self.matrix.ndim != 2:
            raise ValueError("matrix must have shape (n_constraints, design_dim).")
        if self.matrix.shape[0] != self.bound.size:
            raise ValueError("matrix and bound contain different numbers of constraints.")

    @property
    def design_dimension(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def n_constraints(self) -> int:
        return int(self.matrix.shape[0])

    def residual(self, designs: ArrayLike) -> NDArray[np.float64]:
        array = np.asarray(designs, dtype=np.float64)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != self.design_dimension:
            raise ValueError(
                f"designs must have shape (n, {self.design_dimension}), got {array.shape}."
            )
        return array @ self.matrix.T - self.bound[None, :]

    def is_valid(self, designs: ArrayLike) -> NDArray[np.bool_]:
        return np.all(self.residual(designs) <= self.tolerance, axis=1)

    def violation(self, designs: ArrayLike) -> NDArray[np.float64]:
        return np.sum(np.maximum(self.residual(designs) - self.tolerance, 0.0), axis=1)

    @classmethod
    def load(cls, path: str | Path) -> LinearInequalityConstraint:
        with np.load(path) as data:
            return cls(
                matrix=data["A"],
                bound=data["b"],
                tolerance=float(data["tolerance"]) if "tolerance" in data else 1e-6,
            )
