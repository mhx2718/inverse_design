"""SNGP inference with stored random features and predictive covariance."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def random_ortho(n: int, m: int) -> torch.Tensor:
    matrix, _ = torch.linalg.qr(torch.randn(n, m))
    return matrix


class RandomFourierFeatures(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_random_features: int,
        feature_scale: float | None = None,
    ) -> None:
        super().__init__()
        if feature_scale is None:
            feature_scale = math.sqrt(num_random_features / 2)
        self.register_buffer("feature_scale", torch.tensor(float(feature_scale)))

        if num_random_features <= in_dim:
            weights = random_ortho(in_dim, num_random_features)
        else:
            dimensions_left = num_random_features
            blocks: list[torch.Tensor] = []
            while dimensions_left > in_dim:
                blocks.append(random_ortho(in_dim, in_dim))
                dimensions_left -= in_dim
            blocks.append(random_ortho(in_dim, dimensions_left))
            weights = torch.cat(blocks, dim=1)

        feature_norm = torch.randn(weights.shape) ** 2
        weights = weights * feature_norm.sum(0).sqrt()
        self.register_buffer("W", weights)
        self.register_buffer(
            "b",
            torch.empty(num_random_features).uniform_(0, 2 * math.pi),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cos(inputs @ self.W + self.b) / self.feature_scale


class Laplace(nn.Module):
    """SNGP random-feature regression head."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        num_deep_features: int,
        num_gp_features: int,
        normalize_gp_features: bool,
        num_random_features: int,
        num_outputs: int,
        ridge_penalty: float = 1.0,
        feature_scale: float | None = None,
        *,
        scale_covariance_by_ridge: bool = False,
        inference_only: bool = False,
        jl_projection_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.ridge_penalty = float(ridge_penalty)
        self.inference_only = bool(inference_only)
        self.scale_covariance_by_ridge = bool(scale_covariance_by_ridge)
        if jl_projection_std <= 0:
            raise ValueError("Projection std must be positive.")

        if num_gp_features > 0:
            self.num_gp_features = num_gp_features
            self.register_buffer(
                "random_matrix",
                torch.normal(
                    0,
                    jl_projection_std,
                    (num_gp_features, num_deep_features),
                ),
            )
            self.jl = lambda x: nn.functional.linear(x, self.random_matrix)
        else:
            self.num_gp_features = num_deep_features
            self.jl = nn.Identity()

        self.normalize_gp_features = normalize_gp_features
        if normalize_gp_features:
            self.normalize = nn.LayerNorm(num_gp_features)

        self.rff = RandomFourierFeatures(
            num_gp_features,
            num_random_features,
            feature_scale,
        )
        self.beta = nn.Linear(num_random_features, num_outputs)
        self.register_buffer("seen_data", torch.tensor(0))

        if self.inference_only:
            self.register_buffer("precision", None)
        else:
            self.register_buffer(
                "precision",
                torch.eye(num_random_features, dtype=torch.float64) * self.ridge_penalty,
            )
        self.register_buffer("covariance", torch.eye(num_random_features, dtype=torch.float64))

    def random_features(self, shape: torch.Tensor, aoa: torch.Tensor) -> torch.Tensor:
        deep_features = self.feature_extractor(shape, aoa)
        reduced_features = self.jl(deep_features)
        if self.normalize_gp_features:
            reduced_features = self.normalize(reduced_features)
        return self.rff(reduced_features)

    def mean_only(self, shape: torch.Tensor, aoa: torch.Tensor) -> torch.Tensor:
        """Return the predictive mean without touching the covariance."""

        return self.beta(self.random_features(shape, aoa))

    def _covariance_factor(self) -> float:
        return self.ridge_penalty if self.scale_covariance_by_ridge else 1.0

    def covariance_from_random_features(
        self,
        random_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return covariance for K shaped (N, M) or (B, N, M), in FP64."""

        with torch.no_grad():
            features = random_features.detach().to(torch.float64)
            covariance = features @ self.covariance.to(torch.float64) @ features.transpose(-1, -2)
            covariance = self._covariance_factor() * covariance
            covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        return covariance

    def forward(
        self,
        shape: torch.Tensor,
        aoa: torch.Tensor,
        return_covariance: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        random_features = self.random_features(shape, aoa)
        prediction = self.beta(random_features)

        if not return_covariance:
            return prediction

        predictive_covariance = self.covariance_from_random_features(random_features)
        predictive_covariance = predictive_covariance.to(prediction.dtype)
        return prediction, predictive_covariance
