"""Multivariate normal rectangle probabilities evaluated with mvnun."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike


class RectangleProbability(Protocol):
    def probability(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        mean: ArrayLike,
        covariance: ArrayLike,
    ) -> float:
        ...


@lru_cache(maxsize=1)
def _get_mvnun():
    try:
        from scipy.stats._mvn import mvnun
    except ImportError as error:
        raise ImportError(
            "GUIDe requires scipy.stats._mvn.mvnun. Install the repository's "
            "SciPy dependency with `python -m pip install 'scipy>=1.11,<1.16'`."
        ) from error
    return mvnun


@dataclass(frozen=True)
class MVNUNRectangle:
    """Call SciPy's compiled mvnun routine for a joint rectangle probability."""

    max_points: int = 5_000_000
    absolute_tolerance: float = 1e-6
    relative_tolerance: float = 5e-3
    strict_convergence: bool = False

    def __post_init__(self):
        if not isinstance(self.max_points, int) or self.max_points <= 0:
            raise ValueError("max_points must be a positive integer.")
        tolerances = (self.absolute_tolerance, self.relative_tolerance)
        if (not all(np.isfinite(value) and value >= 0 for value in tolerances)
                or not any(value > 0 for value in tolerances)):
            raise ValueError("Integration tolerances must be nonnegative, finite, and not both zero.")
        _get_mvnun()

    def probability(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        mean: ArrayLike,
        covariance: ArrayLike,
    ) -> float:
        lower_array = np.asarray(lower, dtype=np.float64).reshape(-1)
        upper_array = np.asarray(upper, dtype=np.float64).reshape(-1)
        mean_array = np.asarray(mean, dtype=np.float64).reshape(-1)
        covariance_array = np.asarray(covariance, dtype=np.float64)
        dimension = mean_array.size
        if (dimension == 0 or lower_array.shape != (dimension,)
                or upper_array.shape != (dimension,)
                or covariance_array.shape != (dimension, dimension)):
            raise ValueError("Incompatible rectangle bounds, mean, or covariance shapes.")
        if (np.isnan(lower_array).any() or np.isnan(upper_array).any()
                or not np.isfinite(mean_array).all() or not np.isfinite(covariance_array).all()):
            raise ValueError("Bounds must not contain NaNs; mean and covariance must be finite.")
        if np.any(lower_array > upper_array):
            return 0.0
        probability, information = _get_mvnun()(
            lower_array,
            upper_array,
            mean_array,
            covariance_array,
            maxpts=self.max_points,
            abseps=self.absolute_tolerance,
            releps=self.relative_tolerance,
        )
        if information not in (0, 1):
            raise RuntimeError(f"mvnun rejected the integration problem (information={information}).")
        if information == 1 and self.strict_convergence:
            raise RuntimeError("mvnun did not converge within max_points (information=1).")
        if not np.isfinite(probability):
            raise RuntimeError("mvnun returned a non-finite probability.")
        return float(np.clip(probability, 0.0, 1.0))
