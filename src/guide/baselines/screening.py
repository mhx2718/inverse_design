"""CDM with post-hoc GUIDe forward-model likelihood screening (CDM-S)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from guide.core.likelihood import ToleranceTubeLikelihood
from guide.types import GenerationResult, TargetSpecification

from .cdm import CDMBaseline


@dataclass(frozen=True)
class ScreeningConfig:
    n_output: int = 50
    zero_likelihood_tolerance: float = 1e-12


class CDMScreeningBaseline:
    def __init__(
        self,
        cdm: CDMBaseline,
        likelihood: ToleranceTubeLikelihood,
        *,
        config: ScreeningConfig | None = None,
    ) -> None:
        self.cdm = cdm
        self.likelihood = likelihood
        self.config = config or ScreeningConfig()

    @staticmethod
    def ranking_indices(
        likelihood: ArrayLike,
        prediction_error: ArrayLike,
        *,
        zero_tolerance: float = 1e-12,
    ) -> np.ndarray:
        likelihood_array = np.asarray(likelihood, dtype=np.float64).reshape(-1)
        error_array = np.asarray(prediction_error, dtype=np.float64).reshape(-1)
        positive = np.flatnonzero(likelihood_array > zero_tolerance)
        zero = np.flatnonzero(likelihood_array <= zero_tolerance)
        positive_sorted = positive[
            np.lexsort((error_array[positive], -likelihood_array[positive]))
        ]
        zero_sorted = zero[np.argsort(error_array[zero])]
        return np.concatenate([positive_sorted, zero_sorted])

    def run(self, target: TargetSpecification) -> GenerationResult:
        pool = self.cdm.generate_pool(target)
        distribution = self.likelihood.predictive_distribution(pool)
        likelihood = self.likelihood.probability(pool, target, distribution=distribution)
        residual = (distribution.mean - target.response[None, :])[:, target.response_mask]
        mse = np.mean(residual ** 2, axis=1)
        ranking = self.ranking_indices(
            likelihood,
            mse,
            zero_tolerance=self.config.zero_likelihood_tolerance,
        )[: self.config.n_output]
        return GenerationResult(
            method="cdm-s",
            target_id=target.target_id,
            designs=pool[ranking],
            scores=likelihood[ranking],
            score_name="guide_likelihood",
            predicted_mean=distribution.mean[ranking],
            metadata={
                "pool_size": pool.shape[0],
                "zero_likelihood_fallback": "predictive_mean_mse",
            },
        )
