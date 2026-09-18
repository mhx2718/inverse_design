"""Broad uniform random-search baseline in standardized PCA space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from guide.interfaces import DesignConstraint, DeterministicForwardModel
from guide.types import GenerationResult, TargetSpecification


@dataclass(frozen=True)
class RandomSearchConfig:
    lower_quantile: float = 0.005
    upper_quantile: float = 0.995
    expansion_factor: float = 3.0
    pool_size: int = 35_000
    n_output: int = 50
    proposal_batch_size: int = 100_000
    max_total_draws: int = 20_000_000
    seed: int = 0


class RandomSearchBaseline:
    def __init__(
        self,
        forward_model: DeterministicForwardModel,
        constraint: DesignConstraint,
        training_designs: ArrayLike,
        *,
        config: RandomSearchConfig | None = None,
    ) -> None:
        self.forward_model = forward_model
        self.constraint = constraint
        self.training_designs = np.asarray(training_designs, dtype=np.float64)
        if self.training_designs.ndim != 2:
            raise ValueError("training_designs must have shape (n, design_dim).")
        self.config = config or RandomSearchConfig()

    def proposal_bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        cfg = self.config
        q_low = np.quantile(self.training_designs, cfg.lower_quantile, axis=0)
        q_high = np.quantile(self.training_designs, cfg.upper_quantile, axis=0)
        center = 0.5 * (q_low + q_high)
        radius = 0.5 * (q_high - q_low)
        return (
            center - cfg.expansion_factor * radius,
            center + cfg.expansion_factor * radius,
        )

    def _draw_valid_pool(self) -> NDArray[np.float64]:
        cfg = self.config
        lower, upper = self.proposal_bounds()
        rng = np.random.default_rng(cfg.seed)
        chunks: list[NDArray[np.float64]] = []
        collected = 0
        drawn = 0
        while collected < cfg.pool_size:
            batch = rng.uniform(
                lower,
                upper,
                size=(cfg.proposal_batch_size, lower.size),
            )
            valid = batch[self.constraint.is_valid(batch)]
            if valid.size:
                take = min(cfg.pool_size - collected, valid.shape[0])
                chunks.append(valid[:take])
                collected += take
            drawn += cfg.proposal_batch_size
            if drawn > cfg.max_total_draws:
                raise RuntimeError(
                    f"Collected only {collected}/{cfg.pool_size} admissible proposals after {drawn} draws."
                )
        return np.vstack(chunks)

    def run(self, target: TargetSpecification) -> GenerationResult:
        pool = self._draw_valid_pool()
        predictions = np.asarray(self.forward_model.predict_mean(pool), dtype=np.float64)
        residual = (predictions - target.response[None, :])[:, target.response_mask]
        mse = np.mean(residual ** 2, axis=1)
        selected = np.argsort(mse)[: self.config.n_output]
        return GenerationResult(
            method="random-search",
            target_id=target.target_id,
            designs=pool[selected],
            scores=mse[selected],
            score_name="predictive_mean_mse",
            predicted_mean=predictions[selected],
            metadata={
                "pool_size": self.config.pool_size,
                "proposal_expansion_factor": self.config.expansion_factor,
                "seed": self.config.seed,
            },
        )
