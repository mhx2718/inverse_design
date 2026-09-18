"""Method signatures required by the shared generation algorithms.

These Protocol classes describe interfaces. Their method bodies are provided
by the concrete airfoil and nacre models and constraints.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .types import PredictiveDistribution


@runtime_checkable
class ProbabilisticForwardModel(Protocol):
    """Model returning a joint predictive distribution for each design."""

    def predict_distribution(self, designs: ArrayLike) -> PredictiveDistribution:
        ...

    def predict_mean(self, designs: ArrayLike) -> NDArray[np.floating]:
        ...


@runtime_checkable
class DeterministicForwardModel(Protocol):
    """Model returning only a predictive mean curve."""

    def predict_mean(self, designs: ArrayLike) -> NDArray[np.floating]:
        ...


@runtime_checkable
class DesignConstraint(Protocol):
    """Physical or geometric admissibility check in the sampled design space."""

    def is_valid(self, designs: ArrayLike) -> NDArray[np.bool_]:
        ...

    def violation(self, designs: ArrayLike) -> NDArray[np.floating]:
        ...
