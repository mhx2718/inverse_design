"""Conditional one-dimensional U-Net used by the airfoil CDM baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ConditionalUNetConfig:
    design_dimension: int = 16
    response_dimension: int = 10
    base_channels: int = 32
    time_embedding_dimension: int = 128
    depth: int = 2
    dropout: float = 0.3
    latent_length: int = 128
    use_cbam: bool = True
    use_multihead_attention: bool = True
    attention_heads: int = 4
    attention_residual_scale: float = 0.1

    def __post_init__(self) -> None:
        if self.design_dimension <= 0 or self.response_dimension <= 0:
            raise ValueError("Design and response dimensions must be positive.")
        if self.base_channels <= 0 or self.base_channels % 8:
            raise ValueError("base_channels must be a positive multiple of 8 for GroupNorm.")
        if self.time_embedding_dimension <= 0 or self.time_embedding_dimension % 2:
            raise ValueError("time_embedding_dimension must be a positive even number.")
        if self.depth < 0 or self.latent_length <= 0:
            raise ValueError("depth must be non-negative and latent_length must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive.")
        channels = [self.base_channels * (2**index) for index in range(self.depth + 1)]
        if self.use_multihead_attention and any(
            channel % self.attention_heads for channel in channels
        ):
            raise ValueError("Every attention channel count must be divisible by attention_heads.")


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, time_dimension: int = 128, max_period: int = 10_000) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            max_period ** (torch.arange(0, time_dimension, 2).float() / time_dimension)
        )
        self.register_buffer("inv_freq", inverse_frequency)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        sinusoid = timestep.float().unsqueeze(1) * self.inv_freq
        return torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)


class MHAttention1d(nn.Module):
    def __init__(
        self,
        channels: int,
        n_heads: int = 4,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.scale = residual_scale

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(inputs).transpose(1, 2)
        hidden, _ = self.attn(hidden, hidden, hidden)
        return inputs + self.scale * hidden.transpose(1, 2)


class CBAM1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, kernel_size: int = 7) -> None:
        super().__init__()
        reduced_channels = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, reduced_channels, 1),
            nn.SiLU(),
            nn.Conv1d(reduced_channels, channels, 1),
        )
        self.conv_spa = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = F.adaptive_avg_pool1d(inputs, 1) + F.adaptive_max_pool1d(inputs, 1)
        output = inputs * torch.sigmoid(self.mlp(weights))
        average = output.mean(1, keepdim=True)
        maximum = output.max(1, keepdim=True).values
        spatial = torch.sigmoid(self.conv_spa(torch.cat([average, maximum], dim=1)))
        return output * spatial


class FiLM(nn.Module):
    def __init__(self, channels: int, condition_dimension: int) -> None:
        super().__init__()
        self.film = nn.Linear(condition_dimension, channels * 2)

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        return inputs * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)


class DoubleConv(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        time_dimension: int,
        condition_dimension: int,
        *,
        use_cbam: bool,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(output_channels, output_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, output_channels)
        self.norm2 = nn.GroupNorm(8, output_channels)
        self.act = nn.SiLU()
        self.film1 = FiLM(output_channels, time_dimension + condition_dimension)
        self.film2 = FiLM(output_channels, time_dimension + condition_dimension)
        self.cbam = CBAM1d(output_channels)
        self.use_cbam = use_cbam

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        output = self.conv1(inputs)
        output = self.film1(self.act(self.norm1(output)), condition)
        output = self.conv2(output)
        output = self.film2(self.act(self.norm2(output)), condition)
        return self.cbam(output) if self.use_cbam else output


class Down(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        time_dimension: int,
        condition_dimension: int,
        *,
        use_cbam: bool,
        use_attention: bool,
        attention_heads: int,
        attention_residual_scale: float,
    ) -> None:
        super().__init__()
        self.pool = nn.Conv1d(input_channels, input_channels, 4, stride=2, padding=1)
        self.block = DoubleConv(
            input_channels,
            output_channels,
            time_dimension,
            condition_dimension,
            use_cbam=use_cbam,
        )
        self.attn: nn.Module = (
            MHAttention1d(
                output_channels,
                n_heads=attention_heads,
                residual_scale=attention_residual_scale,
            )
            if use_attention
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.attn(self.block(self.pool(inputs), condition))


class Up(nn.Module):
    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        output_channels: int,
        time_dimension: int,
        condition_dimension: int,
        *,
        use_cbam: bool,
        use_attention: bool,
        attention_heads: int,
        attention_residual_scale: float,
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.conv1 = nn.Conv1d(input_channels + skip_channels, output_channels, 3, padding=1)
        self.norm = nn.GroupNorm(8, output_channels)
        self.act = nn.SiLU()
        self.block = DoubleConv(
            output_channels,
            output_channels,
            time_dimension,
            condition_dimension,
            use_cbam=use_cbam,
        )
        self.attn: nn.Module = (
            MHAttention1d(
                output_channels,
                n_heads=attention_heads,
                residual_scale=attention_residual_scale,
            )
            if use_attention
            else nn.Identity()
        )

    def forward(
        self,
        inputs: torch.Tensor,
        skip: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        output = self.up(inputs)
        if output.size(-1) != skip.size(-1):
            difference = skip.size(-1) - output.size(-1)
            if difference < 0:
                output = output[..., : skip.size(-1)]
            else:
                output = F.pad(output, (0, difference))
        output = torch.cat([output, skip], dim=1)
        output = self.act(self.norm(self.conv1(output)))
        return self.attn(self.block(output, condition))


class SelfAttention1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, channels, _ = inputs.shape
        query, key, value = self.qkv(inputs).chunk(3, dim=1)
        attention = torch.matmul(query.transpose(1, 2), key) / (channels**0.5)
        attention = attention.softmax(dim=-1)
        output = torch.matmul(attention, value.transpose(1, 2)).transpose(1, 2)
        return self.proj(output) + inputs


class MidBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        time_dimension: int,
        dropout: float,
        condition_dimension: int,
        *,
        use_cbam: bool,
    ) -> None:
        super().__init__()
        self.block1 = DoubleConv(
            channels,
            channels,
            time_dimension,
            condition_dimension,
            use_cbam=use_cbam,
        )
        self.attn = SelfAttention1d(channels)
        self.drop = nn.Dropout(dropout)
        self.block2 = DoubleConv(
            channels,
            channels,
            time_dimension,
            condition_dimension,
            use_cbam=use_cbam,
        )

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        output = self.block1(inputs, condition)
        output = self.attn(output)
        output = self.drop(output)
        return self.block2(output, condition)


class UNet1D(nn.Module):
    def __init__(self, config: ConditionalUNetConfig) -> None:
        super().__init__()
        self.config = config
        self.depth = config.depth
        self.latent_len = config.latent_length
        condition_dimension = config.time_embedding_dimension

        self.t_embed = SinusoidalTimeEmbedding(config.time_embedding_dimension)
        self.y_global = nn.Sequential(
            nn.Linear(config.response_dimension, config.time_embedding_dimension),
            nn.SiLU(),
        )
        self.y_local_proj = nn.Sequential(
            nn.Linear(config.response_dimension, config.latent_length),
            nn.SiLU(),
        )
        self.y_conv_proj = nn.Sequential(
            nn.Conv1d(1, config.base_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.x_proj = nn.Linear(config.design_dimension, config.latent_length)
        self.drop = nn.Dropout(config.dropout)

        channels = [config.base_channels * (2**index) for index in range(config.depth + 1)]
        self.downs = nn.ModuleList()
        self.downs.append(
            DoubleConv(
                channels[0],
                channels[0],
                config.time_embedding_dimension,
                condition_dimension,
                use_cbam=config.use_cbam,
            )
        )
        for index in range(config.depth):
            self.downs.append(
                Down(
                    channels[index],
                    channels[index + 1],
                    config.time_embedding_dimension,
                    condition_dimension,
                    use_cbam=config.use_cbam,
                    use_attention=config.use_multihead_attention,
                    attention_heads=config.attention_heads,
                    attention_residual_scale=config.attention_residual_scale,
                )
            )

        self.mid = MidBlock(
            channels[config.depth],
            config.time_embedding_dimension,
            config.dropout,
            condition_dimension,
            use_cbam=config.use_cbam,
        )
        self.ups = nn.ModuleList()
        for index in reversed(range(config.depth)):
            self.ups.append(
                Up(
                    channels[index + 1],
                    channels[index],
                    channels[index],
                    config.time_embedding_dimension,
                    condition_dimension,
                    use_cbam=config.use_cbam,
                    use_attention=config.use_multihead_attention,
                    attention_heads=config.attention_heads,
                    attention_residual_scale=config.attention_residual_scale,
                )
            )

        self.out_conv = nn.Sequential(
            nn.Conv1d(config.base_channels, config.base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(config.base_channels, 1, 3, padding=1),
        )
        self.final = nn.Linear(config.latent_length, config.design_dimension)

    def forward(
        self,
        noisy_design: torch.Tensor,
        timestep: torch.Tensor,
        response: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = self.drop(self.t_embed(timestep))
        response_embedding = self.y_global(response)
        condition = torch.cat([time_embedding, response_embedding], dim=-1)

        response_features = self.y_conv_proj(self.y_local_proj(response).unsqueeze(1))
        design_features = self.x_proj(noisy_design).unsqueeze(1)
        design_features = design_features.expand(-1, response_features.size(1), -1)
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


class ConditionalUNet1D(nn.Module):
    """Wrapper retaining the checkpoint key prefix ``net``."""

    def __init__(
        self,
        design_dimension: int = 16,
        response_dimension: int = 10,
        *,
        config: ConditionalUNetConfig | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = ConditionalUNetConfig(
                design_dimension=design_dimension,
                response_dimension=response_dimension,
            )
        self.net = UNet1D(config)

    def forward(
        self,
        noisy_design: torch.Tensor,
        timestep: torch.Tensor,
        response: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(noisy_design, timestep, response)
