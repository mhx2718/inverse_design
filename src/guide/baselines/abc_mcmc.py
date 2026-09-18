"""Soft-kernel ABC-MCMC baseline."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike

from guide.core.likelihood import GaussianABCScore
from guide.core.mcmc import AdaptiveRandomWalkMetropolis, MCMCConfig
from guide.interfaces import DesignConstraint, DeterministicForwardModel
from guide.types import GenerationResult, TargetSpecification


@dataclass(frozen=True)
class ABCMCMCConfig:
    mcmc: MCMCConfig
    bandwidth: float | None = None
    n_output: int = 50
    seed: int = 0


class ABCMCMCBaseline:
    def __init__(
        self,
        forward_model: DeterministicForwardModel,
        constraint: DesignConstraint,
        *,
        config: ABCMCMCConfig,
    ) -> None:
        self.forward_model = forward_model
        self.constraint = constraint
        self.config = config

    def run(
        self,
        target: TargetSpecification,
        *,
        initial_design: ArrayLike,
    ) -> GenerationResult:
        score = GaussianABCScore(self.forward_model)

        def log_density(designs: np.ndarray) -> np.ndarray:
            probabilities = score.probability(
                designs,
                target,
                bandwidth=self.config.bandwidth,
            )
            return np.log(np.maximum(probabilities, 1e-300))

        sampler = AdaptiveRandomWalkMetropolis(
            log_density,
            self.constraint,
            replace(self.config.mcmc, seed=self.config.seed),
        )
        chain = sampler.sample(initial_design)
        states, log_probability = chain.flattened()
        rng = np.random.default_rng(self.config.seed)
        selected = rng.choice(
            states.shape[0],
            size=self.config.n_output,
            replace=states.shape[0] < self.config.n_output,
        )
        designs = states[selected]
        return GenerationResult(
            method="abc-mcmc",
            target_id=target.target_id,
            designs=designs,
            scores=np.exp(log_probability[selected]),
            score_name="abc_kernel_weight",
            predicted_mean=self.forward_model.predict_mean(designs),
            metadata={
                "acceptance_rate": chain.acceptance_rate.tolist(),
                "attempted_steps": chain.attempted_steps,
                "production_steps": chain.production_steps,
                "seed": self.config.seed,
            },
        )
