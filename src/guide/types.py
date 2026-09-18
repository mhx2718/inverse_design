"""Shared data contracts used across GUIDe and its baselines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def _as_float_array(value: Any, *, ndim: int | None = None, name: str = "array") -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


@dataclass(frozen=True)
class TargetSpecification:
    """Full response and tolerance, with an optional subset of evaluated points.

    ``response_mask`` selects the response coordinates used by likelihoods
    and discrepancies. Excluded coordinates remain in the target for
    conditioning and output.
    """

    response: FloatArray
    tolerance: FloatArray | float
    target_id: str = "target"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    response_mask: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        response = _as_float_array(self.response, ndim=1, name="response")
        tolerance = np.asarray(self.tolerance, dtype=np.float64)
        if tolerance.ndim == 0:
            tolerance = np.full(response.shape, float(tolerance), dtype=np.float64)
        if tolerance.shape != response.shape:
            raise ValueError(
                "tolerance must be scalar or have the same shape as response; "
                f"got {tolerance.shape} and {response.shape}."
            )
        if np.any(~np.isfinite(tolerance)) or np.any(tolerance < 0):
            raise ValueError("tolerance must be finite and non-negative.")
        mask = (
            np.ones(response.shape, dtype=bool)
            if self.response_mask is None
            else np.asarray(self.response_mask)
        )
        if mask.shape != response.shape or mask.dtype != np.dtype(bool):
            raise ValueError("response_mask must be a boolean vector matching response.")
        if not np.any(mask):
            raise ValueError("response_mask must select at least one response point.")
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "response_mask", mask.copy())


@dataclass(frozen=True)
class PredictiveDistribution:
    """Batch of predictive means and covariance matrices.

    ``mean`` has shape ``(n_designs, response_dim)`` and ``covariance`` has
    shape ``(n_designs, response_dim, response_dim)``.
    """

    mean: FloatArray
    covariance: FloatArray

    def __post_init__(self) -> None:
        mean = _as_float_array(self.mean, name="mean")
        covariance = _as_float_array(self.covariance, name="covariance")
        if mean.ndim == 1:
            mean = mean[None, :]
        if covariance.ndim == 2:
            covariance = covariance[None, :, :]
        if mean.ndim != 2 or covariance.ndim != 3:
            raise ValueError(
                "mean and covariance must be 2-D and 3-D after batching; "
                f"got {mean.shape} and {covariance.shape}."
            )
        n, k = mean.shape
        if covariance.shape != (n, k, k):
            raise ValueError(
                f"covariance must have shape {(n, k, k)}, got {covariance.shape}."
            )
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class SupportResult:
    """Output of the support-localization stage."""

    design: FloatArray
    objective: float
    likelihood: float
    found_nonzero_support: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "design", _as_float_array(self.design, ndim=1, name="design"))


@dataclass
class GenerationResult:
    """Standardized output contract shared by GUIDe and all baselines."""

    method: str
    target_id: str
    designs: FloatArray
    scores: FloatArray
    score_name: str
    predicted_mean: FloatArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.designs = _as_float_array(self.designs, name="designs")
        self.scores = _as_float_array(self.scores, name="scores").reshape(-1)
        if self.designs.ndim != 2:
            raise ValueError(f"designs must have shape (n, d), got {self.designs.shape}.")
        if self.designs.shape[0] != self.scores.shape[0]:
            raise ValueError("designs and scores must contain the same number of candidates.")
        if self.predicted_mean is not None:
            self.predicted_mean = _as_float_array(self.predicted_mean, name="predicted_mean")
            if self.predicted_mean.ndim != 2:
                raise ValueError("predicted_mean must have shape (n, response_dim).")
            if self.predicted_mean.shape[0] != self.designs.shape[0]:
                raise ValueError("predicted_mean and designs must contain the same candidates.")
