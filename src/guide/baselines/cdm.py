"""Conditional diffusion model baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from guide.models.diffusion import ConditionalDiffusionGenerator
from guide.types import GenerationResult, TargetSpecification


@dataclass(frozen=True)
class CDMConfig:
    pool_size: int = 35_000
    n_output: int = 50
    seed: int = 0


class CDMBaseline:
    def __init__(
        self,
        generator: ConditionalDiffusionGenerator,
        *,
        config: CDMConfig | None = None,
    ) -> None:
        self.generator = generator
        self.config = config or CDMConfig()

    def generate_pool(self, target: TargetSpecification) -> np.ndarray:
        return self.generator.generate(
            target.response,
            n_candidates=self.config.pool_size,
        )

    def run(self, target: TargetSpecification) -> GenerationResult:
        pool = self.generate_pool(target)
        rng = np.random.default_rng(self.config.seed)
        selected = rng.choice(
            pool.shape[0],
            size=self.config.n_output,
            replace=pool.shape[0] < self.config.n_output,
        )
        designs = pool[selected]
        return GenerationResult(
            method="cdm",
            target_id=target.target_id,
            designs=designs,
            scores=np.zeros(designs.shape[0], dtype=np.float64),
            score_name="unranked_random_selection",
            metadata={
                "pool_size": self.config.pool_size,
                "seed": self.config.seed,
            },
        )
