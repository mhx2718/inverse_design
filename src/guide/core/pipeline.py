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
    selection: str = "linspace"

    def __post_init__(self) -> None:
        if self.selection not in {"linspace", "random", "maxmin"}:
            raise ValueError("selection must be 'linspace', 'random', or 'maxmin'.")
        if self.n_output is not None and (
            isinstance(self.n_output, (bool, np.bool_))
            or not isinstance(self.n_output, (int, np.integer))
            or self.n_output <= 0
        ):
            raise ValueError("n_output must be a positive integer or null.")


def select_output_indices(
    designs: np.ndarray,
    n_output: int | None,
    *,
    selection: str,
    seed: int,
) -> np.ndarray:
    """Select rows after sampling; max-min uses standardized design coordinates.

    Random selection samples chain rows without replacement, preserving their
    empirical sampling weights. Max-min starts from a seeded random row and
    greedily maximizes the minimum Euclidean distance to the selected set.
    It stops if all remaining rows duplicate a selected design.
    """
    n_candidates = designs.shape[0]
    if n_output is None or n_candidates == 0:
        return np.arange(n_candidates, dtype=int)
    n_select = min(n_output, n_candidates)
    if selection == "linspace":
        return np.linspace(0, n_candidates - 1, n_select, dtype=int)

    rng = np.random.default_rng(seed)
    if selection == "random":
        return rng.choice(n_candidates, size=n_select, replace=False)
    if selection != "maxmin":
        raise ValueError("selection must be 'linspace', 'random', or 'maxmin'.")

    selected = np.empty(n_select, dtype=int)
    minimum_squared_distance = np.full(n_candidates, np.inf)
    next_index = int(rng.integers(n_candidates))
    for position in range(n_select):
        selected[position] = next_index
        if position + 1 == n_select:
            return selected
        difference = designs - designs[next_index]
        squared_distance = np.einsum("ij,ij->i", difference, difference)
        np.minimum(
            minimum_squared_distance,
            squared_distance,
            out=minimum_squared_distance,
        )
        minimum_squared_distance[selected[: position + 1]] = -np.inf
        next_index = int(np.argmax(minimum_squared_distance))
        if minimum_squared_distance[next_index] <= 0.0:
            return selected[: position + 1]
    return selected


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

        n_candidates = designs.shape[0]
        indices = select_output_indices(
            designs,
            self.config.n_output,
            selection=self.config.selection,
            seed=self.config.mcmc.seed,
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
                "output_selection": self.config.selection if self.config.n_output is not None else "all",
                "selection_seed": self.config.mcmc.seed,
                "selection_coordinates": "standardized_design",
                "n_candidates": n_candidates,
                "n_output": len(indices),
            },
        )
