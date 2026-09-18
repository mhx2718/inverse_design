"""Spectrally normalized residual feature extractors."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import spectral_norm_fc


class FCResNet(nn.Module):
    """Fully connected residual network introduced in SNGP."""

    def __init__(
        self,
        input_dim: int,
        features: int,
        depth: int,
        spectral_normalization: bool,
        coeff: float = 0.95,
        n_power_iterations: int = 1,
        dropout_rate: float = 0.01,
        num_outputs: int | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.first = nn.Linear(input_dim, features)
        self.residuals = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(depth)]
        )
        self.dropout = nn.Dropout(dropout_rate)

        if spectral_normalization:
            self.first = spectral_norm_fc(
                self.first,
                coeff=coeff,
                n_power_iterations=n_power_iterations,
            )
            for index in range(len(self.residuals)):
                self.residuals[index] = spectral_norm_fc(
                    self.residuals[index],
                    coeff=coeff,
                    n_power_iterations=n_power_iterations,
                )

        self.num_outputs = num_outputs
        if num_outputs is not None:
            self.last = nn.Linear(features, num_outputs)
            if spectral_normalization:
                self.last = spectral_norm_fc(
                    self.last,
                    coeff=coeff,
                    n_power_iterations=n_power_iterations,
                )

        if activation == "relu":
            self.activation = F.relu
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.first(inputs)
        for residual in self.residuals:
            features = features + self.dropout(self.activation(residual(features)))
        if self.num_outputs is not None:
            features = self.last(features)
        return features


class PointNetFC(nn.Module):
    """Airfoil shape/AoA feature extractor used by the SNGP checkpoint."""

    def __init__(
        self,
        features: int,
        depth: int,
        spectral_normalization: bool,
        coeff: float = 0.95,
        n_power_iterations: int = 1,
        dropout_rate: float = 0.01,
        out_dim: int = 200,
        num_outputs: int | None = None,
        activation: str = "relu",
        pca_dim: int | None = None,
        pca_hidden: Sequence[int] = (256, 256),
        aoa_dim: int = 32,
    ) -> None:
        super().__init__()
        del num_outputs, activation
        self.out_dim = out_dim
        self.aoa_dim = aoa_dim
        self.pca_dim = pca_dim

        def wrap_fc(layer: nn.Linear) -> nn.Module:
            if spectral_normalization:
                return spectral_norm_fc(
                    layer,
                    coeff=coeff,
                    n_power_iterations=n_power_iterations,
                )
            return layer

        self.point_mlp = FCResNet(
            input_dim=2,
            features=features,
            depth=depth,
            spectral_normalization=spectral_normalization,
            coeff=coeff,
            n_power_iterations=n_power_iterations,
            dropout_rate=dropout_rate,
            num_outputs=out_dim,
        )
        self.aoa_encoder = nn.Sequential(
            wrap_fc(nn.Linear(1, 16)),
            nn.ReLU(),
            wrap_fc(nn.Linear(16, aoa_dim)),
        )
        self.fusion_mlp = nn.Sequential(
            wrap_fc(nn.Linear(out_dim + aoa_dim, 256)),
            nn.ReLU(),
            wrap_fc(nn.Linear(256, features)),
            nn.LayerNorm(features),
        )

        if pca_dim is not None:
            pca_layers: list[nn.Module] = []
            input_dimension = pca_dim
            for hidden_dimension in pca_hidden:
                pca_layers.extend(
                    [wrap_fc(nn.Linear(input_dimension, hidden_dimension)), nn.ReLU()]
                )
                if dropout_rate > 0:
                    pca_layers.append(nn.Dropout(dropout_rate))
                input_dimension = hidden_dimension
            pca_layers.append(wrap_fc(nn.Linear(input_dimension, out_dim)))
            self.pca_encoder: nn.Sequential | None = nn.Sequential(*pca_layers)
        else:
            self.pca_encoder = None

    def forward(self, shape: torch.Tensor, aoa: torch.Tensor) -> torch.Tensor:
        if shape.dim() == 3:
            batch_size, n_points, _ = shape.shape
            point_features = self.point_mlp(shape.reshape(batch_size * n_points, 2))
            point_features = point_features.reshape(batch_size, n_points, -1)
            global_feature = point_features.max(dim=1).values
        elif shape.dim() == 2:
            if self.pca_encoder is None:
                raise ValueError(
                    f"Received PCA input with shape {tuple(shape.shape)} but pca_dim=None."
                )
            global_feature = self.pca_encoder(shape)
        else:
            raise ValueError(f"Unexpected shape tensor {tuple(shape.shape)}.")

        aoa_feature = self.aoa_encoder(aoa)
        return self.fusion_mlp(torch.cat([global_feature, aoa_feature], dim=-1))
