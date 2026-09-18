"""Benchmark-independent generation through PSO support localization and MCMC sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from guide.core.likelihood import ToleranceTubeLikelihood
from guide.core.mcmc import AdaptiveRandomWalkMetropolis, MCMCConfig
from guide.interfaces import DesignConstraint
from guide.types import GenerationResult, TargetSpecification


class NoPosteriorSupportError(RuntimeError):
    """Raised when support localization cannot find a non-negligible likelihood."""


@dataclass(frozen=True)
class GUIDeGenerationConfig:
    mcmc: MCMCConfig
    unique_decimals: int | None = None
    n_output: int | None = None
    permit_low_support_sampling: bool = False


class GUIDeGenerator:
    def __init__(
        self,
        likelihood: ToleranceTubeLikelihood,
        support_finder: object,
        constraint: DesignConstraint,
        *,
        config: GUIDeGenerationConfig,
    ) -> None:
        self.likelihood = likelihood
        self.support_finder = support_finder
        self.constraint = constraint
        self.config = config

    def run(self, target: TargetSpecification) -> GenerationResult:
        support = self.support_finder.find(target)
        support_metadata = {
            "support_objective": support.objective,
            "support_likelihood": support.likelihood,
            "found_nonzero_support": support.found_nonzero_support,
            "support_search": dict(support.metadata),
        }
        if not support.found_nonzero_support and not self.config.permit_low_support_sampling:
            raise NoPosteriorSupportError(
                "Support localization did not identify non-negligible target-satisfaction "
                "likelihood. GUIDe abstained before posterior sampling."
            )

        def log_density(designs):
            return self.likelihood.log_probability(designs, target)
        sampler = AdaptiveRandomWalkMetropolis(
            log_density,
            self.constraint,
            self.config.mcmc,
        )
        chain = sampler.sample(support.design)
        if self.config.unique_decimals is None:
            designs, log_probability = chain.flattened()
        else:
            designs, log_probability = chain.unique_states(
                decimals=self.config.unique_decimals
            )

        if self.config.n_output is not None and designs.shape[0] > self.config.n_output:
            indices = np.linspace(
                0,
                designs.shape[0] - 1,
                self.config.n_output,
                dtype=int,
            )
            designs = designs[indices]
            log_probability = log_probability[indices]

        predictions = self.likelihood.forward_model.predict_mean(designs)
        return GenerationResult(
            method="guide",
            target_id=target.target_id,
            designs=designs,
            scores=np.exp(np.maximum(log_probability, np.log(1e-300))),
            score_name="target_satisfaction_likelihood",
            predicted_mean=predictions,
            metadata={
                **support_metadata,
                "acceptance_rate": chain.acceptance_rate.tolist(),
                "accepted_moves": chain.accepted_moves.tolist(),
                "attempted_steps": chain.attempted_steps,
                "production_steps": chain.production_steps,
                "temperature": self.config.mcmc.temperature,
            },
        )
