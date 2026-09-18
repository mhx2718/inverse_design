"""Noise schedule and ancestral DDPM sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from guide.interfaces import DesignConstraint

from .unet1d import ConditionalUNet1D, ConditionalUNetConfig


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int = 1_000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    sampling_batch_size: int = 512
    seed: int = 0

    def __post_init__(self) -> None:
        if self.timesteps <= 0 or self.sampling_batch_size <= 0:
            raise ValueError("timesteps and sampling_batch_size must be positive.")
        if not 0.0 < self.beta_start <= self.beta_end < 1.0:
            raise ValueError("Expected 0 < beta_start <= beta_end < 1.")


def alpha_schedule(
    config: DiffusionConfig,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beta = torch.linspace(
        config.beta_start,
        config.beta_end,
        config.timesteps,
        device=device,
    )
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar


@torch.inference_mode()
def ancestral_sample(
    model: ConditionalUNet1D,
    response: torch.Tensor,
    config: DiffusionConfig,
) -> torch.Tensor:
    device = next(model.parameters()).device
    beta, alpha, alpha_bar = alpha_schedule(config, device=device)
    design_dimension = model.net.config.design_dimension
    design = torch.randn((response.size(0), design_dimension), device=device)
    model.eval()

    for timestep in reversed(range(config.timesteps)):
        timestep_tensor = torch.full(
            (design.size(0),),
            timestep,
            dtype=torch.long,
            device=device,
        )
        noise = torch.randn_like(design) if timestep > 0 else torch.zeros_like(design)
        predicted_noise = model(design, timestep_tensor, response)
        design = (
            1.0
            / torch.sqrt(alpha[timestep])
            * (
                design
                - beta[timestep]
                / torch.sqrt(1 - alpha_bar[timestep])
                * predicted_noise
            )
            + torch.sqrt(beta[timestep]) * noise
        )
    return design


def _state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint
    raise TypeError("Checkpoint does not contain a model state dictionary.")


def load_conditional_diffusion_model(
    checkpoint_path: str | Path,
    *,
    network_config: ConditionalUNetConfig | None = None,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> ConditionalUNet1D:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network_config = network_config or ConditionalUNetConfig()
    model = ConditionalUNet1D(config=network_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(_state_dict(checkpoint), strict=strict)
    model.to(device)
    model.eval()
    return model


class ConditionalDiffusionGenerator:
    """Generate a fixed number of physically admissible conditional designs."""

    def __init__(
        self,
        model: ConditionalUNet1D,
        constraint: DesignConstraint,
        *,
        config: DiffusionConfig | None = None,
    ) -> None:
        self.model = model
        self.constraint = constraint
        self.config = config or DiffusionConfig()

    def generate(
        self,
        target_response: ArrayLike,
        *,
        n_candidates: int,
        max_batches: int = 100_000,
    ) -> NDArray[np.float64]:
        response_array = np.asarray(target_response, dtype=np.float32).reshape(1, -1)
        if response_array.shape[1] != self.model.net.config.response_dimension:
            raise ValueError("Target response dimension does not match the diffusion model.")

        torch.manual_seed(self.config.seed)
        device = next(self.model.parameters()).device
        accepted: list[NDArray[np.float64]] = []
        n_accepted = 0
        for _ in range(max_batches):
            batch_size = min(
                self.config.sampling_batch_size,
                max(n_candidates - n_accepted, 1),
            )
            response = torch.as_tensor(
                np.repeat(response_array, batch_size, axis=0),
                dtype=torch.float32,
                device=device,
            )
            generated = ancestral_sample(self.model, response, self.config).cpu().numpy()
            valid = self.constraint.is_valid(generated)
            if np.any(valid):
                chunk = generated[valid]
                n_take = min(n_candidates - n_accepted, chunk.shape[0])
                accepted.append(chunk[:n_take].astype(np.float64))
                n_accepted += n_take
            if n_accepted >= n_candidates:
                return np.vstack(accepted)
        raise RuntimeError(
            f"Only generated {n_accepted}/{n_candidates} valid CDM candidates after {max_batches} batches."
        )
