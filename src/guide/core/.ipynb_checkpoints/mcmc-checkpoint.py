"""Adaptive constrained random-walk Metropolis sampling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from guide.interfaces import DesignConstraint

LogDensity = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class MCMCConfig:
    dimension: int
    num_chains: int = 1
    burn_in_accepts: int = 3_500
    production_accepts: int = 35_000
    max_steps: int = 100_000_000
    initial_covariance_scale: float = 0.01 * 1.96
    adaptation_scale: float = 2.38
    covariance_jitter: float = 1e-6
    temperature: float = 1.0
    seed: int = 0
    keep_every_state: bool = True
    progress: bool = True

    def __post_init__(self) -> None:
        if self.dimension <= 0 or self.num_chains <= 0:
            raise ValueError("dimension and num_chains must be positive.")
        if self.burn_in_accepts < 0 or self.production_accepts <= 0:
            raise ValueError("Invalid accepted-move budgets.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.initial_covariance_scale <= 0:
            raise ValueError("initial_covariance_scale must be positive.")
        if self.adaptation_scale <= 0:
            raise ValueError("adaptation_scale must be positive.")
        if self.covariance_jitter < 0:
            raise ValueError("covariance_jitter must be non-negative.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")


@dataclass
class MCMCResult:
    states: NDArray[np.float64]
    log_density: NDArray[np.float64]
    accepted_moves: NDArray[np.int64]
    attempted_steps: int
    production_steps: int
    acceptance_rate: NDArray[np.float64]
    proposal_covariance: NDArray[np.float64]
    metadata: dict[str, object] = field(default_factory=dict)

    def flattened(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return (
            self.states.reshape(-1, self.states.shape[-1]),
            self.log_density.reshape(-1),
        )

    def unique_states(self, *, decimals: int | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        states, log_density = self.flattened()
        keys = np.round(states, decimals=decimals) if decimals is not None else states
        _, first_indices = np.unique(keys, axis=0, return_index=True)
        first_indices.sort()
        return states[first_indices], log_density[first_indices]


class AdaptiveRandomWalkMetropolis:
    """Constrained Gaussian random walk with one post-burn-in covariance update."""

    def __init__(
        self,
        log_density: LogDensity,
        constraint: DesignConstraint,
        config: MCMCConfig,
    ) -> None:
        self.log_density_function = log_density
        self.constraint = constraint
        self.config = config

    def _evaluate_log_density(self, designs: NDArray[np.float64]) -> NDArray[np.float64]:
        valid = np.asarray(self.constraint.is_valid(designs), dtype=bool)
        output = np.full(designs.shape[0], -np.inf, dtype=np.float64)
        if np.any(valid):
            values = np.asarray(self.log_density_function(designs[valid]), dtype=np.float64).reshape(-1)
            if values.shape[0] != int(np.sum(valid)):
                raise ValueError("log_density returned an unexpected number of values.")
            output[valid] = values
        return output

    def sample(self, initial_design: ArrayLike) -> MCMCResult:
        cfg = self.config
        initial = np.asarray(initial_design, dtype=np.float64)
        if initial.ndim == 1:
            initial = np.repeat(initial[None, :], cfg.num_chains, axis=0)
        if initial.shape != (cfg.num_chains, cfg.dimension):
            raise ValueError(
                f"initial_design must have shape {(cfg.num_chains, cfg.dimension)}, "
                f"got {initial.shape}."
            )
        if not np.all(self.constraint.is_valid(initial)):
            raise ValueError("Every initial MCMC state must satisfy the design constraints.")

        rng = np.random.default_rng(cfg.seed)
        current = initial.copy()
        current_log_density = self._evaluate_log_density(current)
        if not np.all(np.isfinite(current_log_density)):
            raise ValueError("Every initial state must have finite nonzero target density.")

        proposal_covariance = np.repeat(
            (cfg.initial_covariance_scale * np.eye(cfg.dimension))[None, :, :],
            cfg.num_chains,
            axis=0,
        )
        cholesky = np.linalg.cholesky(
            proposal_covariance
            + cfg.covariance_jitter * np.eye(cfg.dimension)[None, :, :]
        )

        accepted = np.zeros(cfg.num_chains, dtype=np.int64)
        production_accepted = np.zeros(cfg.num_chains, dtype=np.int64)
        production_steps = 0
        burn_history: list[NDArray[np.float64]] = []
        states: list[NDArray[np.float64]] = []
        log_values: list[NDArray[np.float64]] = []
        burn_in_done = cfg.burn_in_accepts == 0
        adapted = np.zeros(cfg.num_chains, dtype=bool)

        burn_progress = None
        production_progress = None
        
        if cfg.progress:
            try:
                from tqdm.auto import tqdm
        
                if cfg.burn_in_accepts > 0:
                    burn_progress = tqdm(
                        total=cfg.burn_in_accepts,
                        desc="MCMC burn-in",
                    )
                else:
                    production_progress = tqdm(
                        total=cfg.production_accepts,
                        desc="MCMC production",
                    )
            except Exception:
                burn_progress = None
                production_progress = None
        
        last_burn_progress = 0
        last_production_progress = 0

        _attempted_steps = 0
        for _attempted_steps in range(1, cfg.max_steps + 1):
            noise = rng.standard_normal((cfg.num_chains, cfg.dimension))
            proposals = current + np.einsum("bij,bj->bi", cholesky, noise)
            proposal_log_density = self._evaluate_log_density(proposals)

            log_ratio = (proposal_log_density - current_log_density) / cfg.temperature
            log_uniform = np.log(rng.random(cfg.num_chains))
            accept_mask = log_uniform <= np.minimum(log_ratio, 0.0)
            current[accept_mask] = proposals[accept_mask]
            current_log_density[accept_mask] = proposal_log_density[accept_mask]
            accepted += accept_mask.astype(np.int64)
            if not burn_in_done:
                current_burn_progress = int(np.min(accepted))
                if (
                    burn_progress is not None
                    and current_burn_progress > last_burn_progress
                ):
                    burn_progress.update(
                        current_burn_progress - last_burn_progress
                    )
                last_burn_progress = current_burn_progress

            if burn_in_done:
                production_steps += 1
                production_accepted += accept_mask.astype(np.int64)
                if cfg.keep_every_state or np.any(accept_mask):
                    states.append(current.copy())
                    log_values.append(current_log_density.copy())
            else:
                burn_history.append(current.copy())
                for chain_index in np.flatnonzero(
                    (accepted >= cfg.burn_in_accepts) & ~adapted
                ):
                    chain_states = np.asarray(burn_history)[:, chain_index, :]
                    if chain_states.shape[0] > 10:
                        chain_states = chain_states[chain_states.shape[0] // 2 :]
                    if chain_states.shape[0] > 1:
                        empirical_covariance = np.cov(chain_states, rowvar=False)
                        proposal_covariance[chain_index] = (
                            cfg.adaptation_scale**2
                            / cfg.dimension
                            * empirical_covariance
                        )
                    adapted[chain_index] = True
                    cholesky[chain_index] = np.linalg.cholesky(
                        proposal_covariance[chain_index]
                        + cfg.covariance_jitter * np.eye(cfg.dimension)
                    )

                if np.all(accepted >= cfg.burn_in_accepts):
                    burn_in_done = True
                    production_accepted.fill(0)
                    states = [current.copy()]
                    log_values = [current_log_density.copy()]
                    if burn_progress is not None:
                        burn_progress.close()
                        burn_progress = None
                    
                    if production_progress is not None:
                        production_progress.close()
                        production_progress = None
                    
                    if cfg.progress:
                        try:
                            from tqdm.auto import tqdm
                    
                            production_progress = tqdm(
                                total=cfg.production_accepts,
                                desc="MCMC production",
                            )
                        except Exception:
                            production_progress = None
                    
                    last_production_progress = 0
                    continue

            if burn_in_done:
                current_production_progress = int(
                    np.min(production_accepted)
                )
                if (
                    production_progress is not None
                    and current_production_progress
                    > last_production_progress
                ):
                    production_progress.update(
                        current_production_progress
                        - last_production_progress
                    )
                last_production_progress = current_production_progress
                if np.all(production_accepted >= cfg.production_accepts):
                    break
        else:
            if burn_progress is not None:
                burn_progress.close()
        
            if production_progress is not None:
                production_progress.close()
        
            raise RuntimeError(
                "MCMC reached max_steps before every chain reached the accepted-move budget. "
                f"Accepted production moves: {production_accepted.tolist()}."
            )
        
        if burn_progress is not None:
            burn_progress.close()
        
        if production_progress is not None:
            production_progress.close()

        if not states:
            states = [current.copy()]
            log_values = [current_log_density.copy()]

        state_array = np.asarray(states, dtype=np.float64)
        log_array = np.asarray(log_values, dtype=np.float64)
        denominator = max(production_steps, 1)
        acceptance_rate = production_accepted / denominator

        return MCMCResult(
            states=state_array,
            log_density=log_array,
            accepted_moves=production_accepted,
            attempted_steps=_attempted_steps,
            production_steps=production_steps,
            acceptance_rate=acceptance_rate,
            proposal_covariance=proposal_covariance,
            metadata={
                "burn_in_accepts": cfg.burn_in_accepts,
                "production_accepts": cfg.production_accepts,
                "temperature": cfg.temperature,
                "adaptation_scale": cfg.adaptation_scale,
            },
        )
