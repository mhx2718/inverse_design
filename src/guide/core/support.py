"""Support-localization algorithms used before posterior sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from guide.interfaces import DesignConstraint
from guide.types import SupportResult, TargetSpecification

from .likelihood import GUIDeSupportObjective


@dataclass(frozen=True)
class PSOConfig:
    n_particles: int = 80
    iterations: int = 15
    cognitive: float = 1.49445
    social: float = 1.49445
    inertia: float = 0.729
    lower_bound: float | ArrayLike = -3.0
    upper_bound: float | ArrayLike = 3.0
    nonzero_threshold: float = 1e-10
    invalid_base_penalty: float = 100.0
    invalid_violation_scale: float = 1000.0
    seed: int = 0


class ParticleSwarmSupportFinder:
    """Global-best PSO wrapper with early recording of nonzero support."""

    def __init__(
        self,
        objective: GUIDeSupportObjective,
        constraint: DesignConstraint,
        dimension: int,
        config: PSOConfig | None = None,
    ) -> None:
        self.objective = objective
        self.constraint = constraint
        self.dimension = int(dimension)
        self.config = config or PSOConfig()

    def find(self, target: TargetSpecification) -> SupportResult:
        try:
            import pyswarms as ps
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "ParticleSwarmSupportFinder requires pyswarms. Install with "
                "`python -m pip install -e '.[airfoil]'` or `python -m pip install -e '.[nacre]'`."
            ) from error

        np.random.seed(self.config.seed)
        lower = np.broadcast_to(
            np.asarray(self.config.lower_bound, dtype=np.float64),
            (self.dimension,),
        ).copy()
        upper = np.broadcast_to(
            np.asarray(self.config.upper_bound, dtype=np.float64),
            (self.dimension,),
        ).copy()
        if np.any(lower >= upper):
            raise ValueError("Every PSO lower bound must be smaller than its upper bound.")

        state: dict[str, object] = {
            "found": False,
            "design": None,
            "likelihood": 0.0,
            "objective": -np.inf,
            "fallback_design": None,
            "fallback_cost": np.inf,
        }

        def minimization_objective(particles: NDArray[np.float64]) -> NDArray[np.float64]:
            violation = np.asarray(self.constraint.violation(particles), dtype=np.float64)
            # Strict physical inequalities can have zero violation at an
            # inadmissible boundary, so validity must use the constraint API.
            valid = np.asarray(self.constraint.is_valid(particles), dtype=bool)
            costs = np.empty(particles.shape[0], dtype=np.float64)
            costs[~valid] = (
                self.config.invalid_base_penalty
                + self.config.invalid_violation_scale * violation[~valid]
            )
            if np.any(valid):
                scores, probabilities = self.objective.evaluate(particles[valid], target)
                costs[valid] = -np.where(probabilities > 0.0, probabilities, scores)
                for design, score, probability, cost in zip(
                    particles[valid], scores, probabilities, costs[valid], strict=True
                ):
                    if cost < state["fallback_cost"]:
                        state.update(
                            fallback_design=np.asarray(design, dtype=np.float64).copy(),
                            fallback_cost=float(cost),
                            fallback_likelihood=float(probability),
                            fallback_objective=float(score),
                        )
                    if probability >= self.config.nonzero_threshold and not state["found"]:
                        state.update(
                            found=True,
                            design=np.asarray(design, dtype=np.float64).copy(),
                            likelihood=float(probability),
                            objective=float(score),
                        )
            return costs

        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.config.n_particles,
            dimensions=self.dimension,
            options={
                "c1": self.config.cognitive,
                "c2": self.config.social,
                "w": self.config.inertia,
            },
            bounds=(lower, upper),
        )
        best_cost, best_position = optimizer.optimize(
            minimization_objective,
            iters=self.config.iterations,
            verbose=False,
        )

        if state["found"]:
            design = np.asarray(state["design"], dtype=np.float64)
            likelihood = float(state["likelihood"])
            objective_value = float(state["objective"])
        else:
            if state["fallback_design"] is None:
                raise RuntimeError("PSO support search did not find any admissible candidate.")
            design = np.asarray(state["fallback_design"], dtype=np.float64)
            likelihood = float(state["fallback_likelihood"])
            objective_value = float(state["fallback_objective"])

        return SupportResult(
            design=design,
            objective=objective_value,
            likelihood=likelihood,
            found_nonzero_support=bool(state["found"]),
            metadata={
                "optimizer": "global_best_pso",
                "best_cost": float(best_cost),
                "best_position": np.asarray(best_position).tolist(),
                "config": self.config.__dict__,
            },
        )
