"""MCMC-based Bayesian inversion baseline."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike

from guide.core.likelihood import PredictivePointDensity
from guide.core.mcmc import AdaptiveRandomWalkMetropolis, MCMCConfig
from guide.interfaces import DesignConstraint, ProbabilisticForwardModel
from guide.types import GenerationResult, TargetSpecification


@dataclass(frozen=True)
class MCMCBIConfig:
    mcmc: MCMCConfig
    covariance_scale: float = 0.004325925666499213
    covariance_jitter: float = 1e-6
    n_output: int = 50
    seed: int = 0


class MCMCBIBaseline:
    def __init__(
        self,
        forward_model: ProbabilisticForwardModel,
        constraint: DesignConstraint,
        *,
        config: MCMCBIConfig,
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
        point_density = PredictivePointDensity(
            self.forward_model,
            covariance_scale=self.config.covariance_scale,
            covariance_jitter=self.config.covariance_jitter,
        )
        sampler = AdaptiveRandomWalkMetropolis(
            lambda designs: point_density.log_density(designs, target),
            self.constraint,
            replace(self.config.mcmc, seed=self.config.seed),
        )
        chain = sampler.sample(initial_design)
        states, log_density = chain.flattened()
        rng = np.random.default_rng(self.config.seed)
        replace_sample = states.shape[0] < self.config.n_output
        selected = rng.choice(
            states.shape[0],
            size=self.config.n_output,
            replace=replace_sample,
        )
        designs = states[selected]
        predictions = self.forward_model.predict_mean(designs)
        return GenerationResult(
            method="mcmc-bi",
            target_id=target.target_id,
            designs=designs,
            scores=log_density[selected],
            score_name="target_log_predictive_density",
            predicted_mean=predictions,
            metadata={
                "acceptance_rate": chain.acceptance_rate.tolist(),
                "attempted_steps": chain.attempted_steps,
                "production_steps": chain.production_steps,
                "seed": self.config.seed,
            },
        )
