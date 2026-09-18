"""Cohesive-interface admissibility in standardized ten-dimensional design space."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .representation import NacreRepresentation


@dataclass
class NacreCohesiveConstraint:
    """Positive physical inputs plus normal/shear initial-slope inequalities.

    Optional lower/upper bounds are in STANDARDIZED design coordinates.
    No empirical training-box upper bound is imposed by default. All physical
    values must be strictly positive.
    """

    representation: NacreRepresentation
    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None

    def __post_init__(self):
        for name in ("lower_bounds", "upper_bounds"):
            bound = getattr(self, name)
            if bound is not None:
                bound = np.asarray(bound, dtype=float).reshape(-1)
                if bound.shape != (10,) or not np.all(np.isfinite(bound)):
                    raise ValueError(f"{name} must contain ten finite standardized bounds.")
                setattr(self, name, bound)
        if self.lower_bounds is not None and self.upper_bounds is not None:
            if np.any(self.lower_bounds >= self.upper_bounds):
                raise ValueError("Every upper bound must exceed its lower bound.")

    @property
    def design_dimension(self):
        return 10

    def is_valid(self, designs):
        z = self.representation._designs(designs)
        x = self.representation.inverse_transform_designs(z)
        valid = np.all(np.isfinite(x), axis=1) & np.all(x > 0, axis=1)
        # Cross multiplication avoids dividing by zero, after positivity checks.
        with np.errstate(invalid="ignore", over="ignore"):
            valid &= x[:, 0] * x[:, 3] > x[:, 1] * x[:, 2]
            valid &= x[:, 5] * x[:, 8] > x[:, 6] * x[:, 7]
        if self.lower_bounds is not None:
            valid &= np.all(z >= self.lower_bounds, axis=1)
        if self.upper_bounds is not None:
            valid &= np.all(z <= self.upper_bounds, axis=1)
        return valid

    def violation(self, designs):
        z = self.representation._designs(designs)
        x = self.representation.inverse_transform_designs(z)
        v = np.maximum(-x / self.representation.scale[:10], 0).sum(axis=1)
        for sy, sd, d1, d2 in ((0, 1, 2, 3), (5, 6, 7, 8)):
            denominator = self.representation.scale[sy] * self.representation.scale[d2]
            with np.errstate(invalid="ignore", over="ignore"):
                v += np.maximum((x[:, sd] * x[:, d1] - x[:, sy] * x[:, d2]) / denominator, 0)
        if self.lower_bounds is not None:
            v += np.maximum(self.lower_bounds - z, 0).sum(axis=1)
        if self.upper_bounds is not None:
            v += np.maximum(z - self.upper_bounds, 0).sum(axis=1)
        v = np.where(self.is_valid(z), 0., np.maximum(v, np.finfo(float).eps))
        return np.nan_to_num(v, nan=np.inf, posinf=np.inf, neginf=np.inf)
