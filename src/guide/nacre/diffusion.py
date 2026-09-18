"""Nacre conditional diffusion model with an attention U-Net.

The input designs are standardized cohesive parameters, and conditions are
100-point stress curves in MPa. The network uses a 100-point internal sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from guide.interfaces import DesignConstraint
from guide.io.checkpoints import load_model_checkpoint
from guide.models.diffusion.process import (
    ConditionalDiffusionGenerator,
    DiffusionConfig,
    _state_dict,
)
from guide.models.diffusion.unet1d import (
    DoubleConv,
    Down,
    MidBlock,
    SinusoidalTimeEmbedding,
    Up,
)


@dataclass(frozen=True)
class NacreConditionalUNetConfig:
    """Nacre conditional diffusion architecture."""

    design_dimension: int = 10
    response_dimension: int = 100
    base_channels: int = 64
    time_embedding_dimension: int = 128
    depth: int = 3
    dropout: float = 0.1
    use_cbam: bool = True
    use_multihead_attention: bool = True
    attention_heads: int = 4
    attention_residual_scale: float = 0.1

    def __post_init__(self) -> None:
        if self.design_dimension != 10:
            raise ValueError("Nacre designs must contain ten cohesive parameters.")
        if self.depth < 0 or self.response_dimension < 2**self.depth:
            raise ValueError("response_dimension must allow all downsampling stages.")
        if self.base_channels <= 0 or self.base_channels % 8:
            raise ValueError("base_channels must be a positive multiple of eight.")
        if self.time_embedding_dimension <= 0 or self.time_embedding_dimension % 2:
            raise ValueError("time_embedding_dimension must be a positive even number.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive.")
        if self.use_multihead_attention and self.base_channels % self.attention_heads:
            raise ValueError("base_channels must be divisible by attention_heads.")
        if not 0 <= self.attention_residual_scale:
            raise ValueError("attention_residual_scale must be nonnegative.")


class NacreUNet1D(nn.Module):
    """Attention U-Net for ten-dimensional nacre designs."""

    def __init__(self, config: NacreConditionalUNetConfig) -> None:
        super().__init__()
        self.config = config
        self.depth = config.depth
        time_dim = config.time_embedding_dimension
        base_ch = config.base_channels
        self.t_embed = SinusoidalTimeEmbedding(time_dim)
        self.y_global = nn.Sequential(nn.Linear(config.response_dimension, time_dim), nn.SiLU())
        self.y_proj = nn.Sequential(nn.Conv1d(1, base_ch, 3, padding=1), nn.SiLU())
        self.x_proj = nn.Linear(config.design_dimension, config.response_dimension)
        self.drop = nn.Dropout(config.dropout)

        channels = [base_ch * 2**index for index in range(config.depth + 1)]
        block_kwargs = {"use_cbam": config.use_cbam}
        attention_kwargs = {
            **block_kwargs,
            "use_attention": config.use_multihead_attention,
            "attention_heads": config.attention_heads,
            "attention_residual_scale": config.attention_residual_scale,
        }
        self.downs = nn.ModuleList(
            [DoubleConv(channels[0], channels[0], time_dim, time_dim, **block_kwargs)]
        )
        for index in range(config.depth):
            self.downs.append(
                Down(channels[index], channels[index + 1], time_dim, time_dim, **attention_kwargs)
            )
        self.mid = MidBlock(channels[-1], time_dim, config.dropout, time_dim, **block_kwargs)
        self.ups = nn.ModuleList()
        for index in reversed(range(config.depth)):
            self.ups.append(
                Up(
                    channels[index + 1], channels[index], channels[index],
                    time_dim, time_dim, **attention_kwargs,
                )
            )
        self.out_conv = nn.Sequential(
            nn.Conv1d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(base_ch, 1, 3, padding=1),
        )
        self.final = nn.Linear(config.response_dimension, config.design_dimension)

    def forward(
        self, noisy_design: torch.Tensor, timestep: torch.Tensor, response: torch.Tensor
    ) -> torch.Tensor:
        condition = torch.cat([self.drop(self.t_embed(timestep)), self.y_global(response)], dim=-1)
        response_features = self.y_proj(response.unsqueeze(1))
        design_features = self.x_proj(noisy_design).unsqueeze(1).expand_as(response_features)
        hidden = response_features + design_features
        skips: list[torch.Tensor] = []
        for index, down in enumerate(self.downs):
            hidden = down(hidden, condition)
            if index < self.depth:
                skips.append(hidden)
        hidden = self.mid(hidden, condition)
        for up in self.ups:
            hidden = up(hidden, skips.pop(), condition)
        return self.final(self.out_conv(hidden).squeeze(1))


class NacreConditionalUNet1D(nn.Module):
    """Conditional U-Net with the checkpoint's ``net`` parameter prefix."""

    def __init__(self, config: NacreConditionalUNetConfig | None = None) -> None:
        super().__init__()
        self.net = NacreUNet1D(config or NacreConditionalUNetConfig())

    def forward(
        self, noisy_design: torch.Tensor, timestep: torch.Tensor, response: torch.Tensor
    ) -> torch.Tensor:
        return self.net(noisy_design, timestep, response)


def load_nacre_diffusion_model(
    checkpoint_path: str | Path,
    *,
    network_config: NacreConditionalUNetConfig | None = None,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> NacreConditionalUNet1D:
    """Load a state dictionary or a checkpoint with network metadata.

    An explicit network config overrides checkpoint metadata. Strict loading is
    the default, so an airfoil checkpoint or incompatible nacre variant fails.
    """
    checkpoint = load_model_checkpoint(checkpoint_path)
    if network_config is None:
        saved_config = checkpoint.get("network_config") if isinstance(checkpoint, dict) else None
        network_config = NacreConditionalUNetConfig(**saved_config) if saved_config else NacreConditionalUNetConfig()
    model = NacreConditionalUNet1D(network_config)
    model.load_state_dict(_state_dict(checkpoint), strict=strict)
    model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model


class NacreConditionalDiffusionGenerator(ConditionalDiffusionGenerator):
    """Generate standardized admissible nacre designs for ``CDMBaseline``.

    ``constraint`` must accept standardized designs (e.g. ``NacreConstraint``).
    The inherited DDPM sampler uses raw target stresses and performs no implicit
    scaling or clipping. It filters designs using the supplied constraint.
    """

    def __init__(
        self,
        model: NacreConditionalUNet1D,
        constraint: DesignConstraint,
        *,
        config: DiffusionConfig | None = None,
    ) -> None:
        super().__init__(model, constraint, config=config)

    def generate(self, target_response, *, n_candidates: int, max_batches: int = 100_000):
        if n_candidates <= 0 or max_batches <= 0:
            raise ValueError("n_candidates and max_batches must be positive.")
        return super().generate(target_response, n_candidates=n_candidates, max_batches=max_batches)
