"""Surrogate-assisted genetic algorithm baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from guide.interfaces import DesignConstraint, DeterministicForwardModel
from guide.types import GenerationResult, TargetSpecification


@dataclass(frozen=True)
class GAConfig:
    generations: int = 400
    population_size: int = 200
    parents_per_generation: int = 30
    design_dimension: int = 16
    initial_lower_bound: float = -6.0
    initial_upper_bound: float = 6.0
    mutation_probability: float = 0.30
    mutation_standard_deviation: float = 1.0
    elite_count: int = 30
    kept_parent_count: int = 30
    top_k: int = 50
    unique_decimals: int = 6
    seed: int = 0


class GeneticAlgorithmBaseline:
    def __init__(
        self,
        forward_model: DeterministicForwardModel,
        constraint: DesignConstraint,
        *,
        config: GAConfig | None = None,
    ) -> None:
        self.forward_model = forward_model
        self.constraint = constraint
        self.config = config or GAConfig()

    def run(self, target: TargetSpecification) -> GenerationResult:
        try:
            import pygad
        except ImportError as error:
            raise ImportError(
                "The GA baseline requires PyGAD. Install with "
                "`python -m pip install -e '.[airfoil]'` or `python -m pip install -e '.[nacre]'`."
            ) from error

        cfg = self.config
        np.random.seed(cfg.seed)
        history: list[tuple[np.ndarray, float]] = []

        def evaluate_mse(solution: np.ndarray) -> float:
            solution_2d = np.asarray(solution, dtype=np.float64)[None, :]
            if not bool(self.constraint.is_valid(solution_2d)[0]):
                return float("inf")
            prediction = self.forward_model.predict_mean(solution_2d)[0]
            return float(np.mean((prediction - target.response)[target.response_mask] ** 2))

        def fitness_function(_instance: object, solution: np.ndarray, _index: int) -> float:
            mse = evaluate_mse(solution)
            return -1e16 if not np.isfinite(mse) else 1.0 / (mse + 1e-8)

        def mutation_operator(offspring: np.ndarray, _instance: object) -> np.ndarray:
            mask = np.random.random(offspring.shape) < cfg.mutation_probability
            noise = np.random.normal(
                0.0,
                cfg.mutation_standard_deviation,
                size=offspring.shape,
            )
            offspring[mask] += noise[mask]
            return offspring

        def on_generation(instance: object) -> None:
            population = np.asarray(instance.population)
            for solution in population:
                mse = evaluate_mse(solution)
                if np.isfinite(mse):
                    history.append((solution.copy(), mse))

        ga = pygad.GA(
            num_generations=cfg.generations,
            num_parents_mating=cfg.parents_per_generation,
            fitness_func=fitness_function,
            sol_per_pop=cfg.population_size,
            num_genes=cfg.design_dimension,
            init_range_low=cfg.initial_lower_bound,
            init_range_high=cfg.initial_upper_bound,
            mutation_type=mutation_operator,
            crossover_type="uniform",
            parent_selection_type="tournament",
            keep_elitism=cfg.elite_count,
            keep_parents=cfg.kept_parent_count,
            on_generation=on_generation,
            random_seed=cfg.seed,
            suppress_warnings=True,
        )
        ga.run()
        if not history:
            raise RuntimeError("The GA did not produce any physically admissible candidate.")

        unique: dict[tuple[float, ...], tuple[np.ndarray, float]] = {}
        for design, mse in history:
            key = tuple(np.round(design, cfg.unique_decimals))
            existing = unique.get(key)
            if existing is None or mse < existing[1]:
                unique[key] = (design, mse)
        ordered = sorted(unique.values(), key=lambda item: item[1])[: cfg.top_k]
        designs = np.asarray([item[0] for item in ordered], dtype=np.float64)
        mse = np.asarray([item[1] for item in ordered], dtype=np.float64)
        predictions = self.forward_model.predict_mean(designs)

        return GenerationResult(
            method="ga",
            target_id=target.target_id,
            designs=designs,
            scores=mse,
            score_name="predictive_mean_mse",
            predicted_mean=predictions,
            metadata={
                "generations": cfg.generations,
                "population_size": cfg.population_size,
                "evaluations_nominal": cfg.generations * cfg.population_size,
                "seed": cfg.seed,
            },
        )
