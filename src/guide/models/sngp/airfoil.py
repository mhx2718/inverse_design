"""Airfoil SNGP construction, checkpoint loading, and inference adapter."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from guide.io.checkpoints import load_model_checkpoint
from guide.types import PredictiveDistribution

from .feature_extractor import PointNetFC
from .laplace import Laplace

DEFAULT_STANDARDIZED_AOA = np.array(
    [
        -1.5666989,
        -1.2185436,
        -0.8703883,
        -0.52223295,
        -0.17407766,
        0.17407766,
        0.52223295,
        0.8703883,
        1.2185436,
        1.5666989,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class AirfoilSNGPConfig:
    latent_dimension: int = 16
    features: int = 256
    depth: int = 6
    spectral_normalization: bool = True
    spectral_norm_coefficient: float = 0.95
    n_power_iterations: int = 1
    dropout_rate: float = 0.0
    shape_embedding_dimension: int = 200
    aoa_embedding_dimension: int = 32
    gp_projection_dimension: int = 256
    random_feature_dimension: int = 4096
    normalize_gp_features: bool = True
    random_feature_scale: float = 2.0
    ridge_penalty: float = 0.1

    # Whether raw predictive covariance includes an external ridge multiplier.
    covariance_ridge_scaling: bool = False
    jl_projection_std: float = 0.05

    num_outputs: int = 1

    def __post_init__(self) -> None:
        integer_fields = {
            "latent_dimension": self.latent_dimension,
            "features": self.features,
            "depth": self.depth,
            "shape_embedding_dimension": self.shape_embedding_dimension,
            "aoa_embedding_dimension": self.aoa_embedding_dimension,
            "gp_projection_dimension": self.gp_projection_dimension,
            "random_feature_dimension": self.random_feature_dimension,
            "num_outputs": self.num_outputs,
        }
        if any(value <= 0 for value in integer_fields.values()):
            raise ValueError(f"SNGP dimensions and budgets must be positive: {integer_fields}")
        if self.n_power_iterations <= 0:
            raise ValueError("n_power_iterations must be positive.")
        if self.spectral_norm_coefficient <= 0:
            raise ValueError("spectral_norm_coefficient must be positive.")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must lie in [0, 1).")
        if self.ridge_penalty <= 0 or self.random_feature_scale <= 0:
            raise ValueError("ridge_penalty and random_feature_scale must be positive.")
        if self.jl_projection_std <= 0:
            raise ValueError("jl_projection_std must be positive.")


def build_airfoil_sngp(
    config: AirfoilSNGPConfig | None = None,
    *,
    inference_only: bool = False,
) -> Laplace:
    config = config or AirfoilSNGPConfig()
    feature_extractor = PointNetFC(
        features=config.features,
        depth=config.depth,
        spectral_normalization=config.spectral_normalization,
        coeff=config.spectral_norm_coefficient,
        n_power_iterations=config.n_power_iterations,
        dropout_rate=config.dropout_rate,
        out_dim=config.shape_embedding_dimension,
        pca_dim=config.latent_dimension,
        aoa_dim=config.aoa_embedding_dimension,
    )
    return Laplace(
        feature_extractor=feature_extractor,
        num_deep_features=config.features,
        num_gp_features=config.gp_projection_dimension,
        normalize_gp_features=config.normalize_gp_features,
        num_random_features=config.random_feature_dimension,
        num_outputs=config.num_outputs,
        ridge_penalty=config.ridge_penalty,
        feature_scale=config.random_feature_scale,
        scale_covariance_by_ridge=config.covariance_ridge_scaling,
        inference_only=inference_only,
        jl_projection_std=config.jl_projection_std,
    )


def _checkpoint_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in checkpoint.items()
        ):
            return checkpoint
    raise TypeError("Checkpoint does not contain a recognizable model state dictionary.")


def _checkpoint_config(checkpoint: Any) -> dict[str, Any] | None:
    if isinstance(checkpoint, dict):
        value = checkpoint.get("model_config")
        if isinstance(value, dict):
            return value
    return None


def _align_uq_buffer_dtypes(model: Laplace, state: dict[str, torch.Tensor]) -> None:
    """Preserve the stored precision and covariance dtypes on load."""

    for name in ("precision", "covariance"):
        if name not in state:
            continue
        current = getattr(model, name, None)
        source = state[name]
        if current is None:
            continue
        if current.dtype != source.dtype:
            setattr(model, name, current.to(dtype=source.dtype))


def load_airfoil_sngp(
    checkpoint_path: str | Path,
    *,
    config: AirfoilSNGPConfig | None = None,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> Laplace:
    """Load network weights and the precomputed predictive covariance."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_model_checkpoint(checkpoint_path)
    state = _checkpoint_state(checkpoint)

    stored_config = _checkpoint_config(checkpoint)
    if stored_config is not None:
        inference_fields = {field.name for field in fields(AirfoilSNGPConfig)}
        stored_config = {key: value for key, value in stored_config.items() if key in inference_fields}
    if config is None and stored_config is not None:
        config = AirfoilSNGPConfig(**stored_config)
    config = config or AirfoilSNGPConfig()
    if stored_config is not None:
        differences = {
            key: (getattr(config, key, None), value)
            for key, value in stored_config.items()
            if getattr(config, key, None) != value
        }
        if differences:
            raise ValueError(f"SNGP configuration differs from checkpoint (config, saved): {differences}")

    covariance = state.get("covariance")
    if covariance is None or not bool(torch.isfinite(covariance).all()):
        raise ValueError("The SNGP checkpoint must contain a finite predictive covariance.")

    compact = "precision" not in state
    model = build_airfoil_sngp(config, inference_only=compact)
    _align_uq_buffer_dtypes(model, state)

    incompatible = model.load_state_dict(state, strict=strict)
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        raise RuntimeError(
            "SNGP checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    model.to(device)
    model.eval()

    return model


class AirfoilSNGPForwardModel:
    """Adapter exposing joint-distribution and efficient predictive-mean APIs."""

    def __init__(
        self,
        model: Laplace,
        *,
        standardized_aoa: ArrayLike = DEFAULT_STANDARDIZED_AOA,
        mean_batch_size: int = 4096,
        distribution_batch_size: int = 16,
    ) -> None:
        self.model = model
        aoa = np.asarray(standardized_aoa, dtype=np.float32).reshape(-1)
        if aoa.size == 0 or not np.all(np.isfinite(aoa)):
            raise ValueError("standardized_aoa must contain finite values.")
        if mean_batch_size <= 0 or distribution_batch_size <= 0:
            raise ValueError("Inference batch sizes must be positive.")
        self.standardized_aoa = aoa
        self.mean_batch_size = int(mean_batch_size)
        self.distribution_batch_size = int(distribution_batch_size)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _design_array(self, designs: ArrayLike) -> NDArray[np.float32]:
        array = np.asarray(designs, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise ValueError("designs must have shape (n_designs, latent_dimension).")
        expected_dimension = self.model.feature_extractor.pca_dim
        if expected_dimension is not None and array.shape[1] != expected_dimension:
            raise ValueError(
                f"Expected latent dimension {expected_dimension}, got {array.shape[1]}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("designs contains non-finite values.")
        return array

    def predict_mean(self, designs: ArrayLike) -> NDArray[np.float64]:
        array = self._design_array(designs)
        n_designs = array.shape[0]
        n_aoa = self.standardized_aoa.size
        repeated_designs = np.repeat(array, n_aoa, axis=0)
        tiled_aoa = np.tile(self.standardized_aoa, n_designs)[:, None]
        predictions: list[np.ndarray] = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, repeated_designs.shape[0], self.mean_batch_size):
                end = min(start + self.mean_batch_size, repeated_designs.shape[0])
                design_tensor = torch.as_tensor(
                    repeated_designs[start:end],
                    dtype=torch.float32,
                    device=self.device,
                )
                aoa_tensor = torch.as_tensor(
                    tiled_aoa[start:end],
                    dtype=torch.float32,
                    device=self.device,
                )
                batch = self.model.mean_only(design_tensor, aoa_tensor)
                predictions.append(batch.squeeze(-1).cpu().numpy())
        return np.concatenate(predictions).reshape(n_designs, n_aoa).astype(np.float64)

    def predict_distribution(self, designs: ArrayLike) -> PredictiveDistribution:
        array = self._design_array(designs)
        n_designs = array.shape[0]
        n_aoa = self.standardized_aoa.size
        means: list[NDArray[np.float64]] = []
        covariances: list[NDArray[np.float64]] = []

        self.model.eval()
        with torch.inference_mode():
            for start in range(0, n_designs, self.distribution_batch_size):
                batch = array[start : start + self.distribution_batch_size]
                batch_size = batch.shape[0]
                repeated_designs = np.repeat(batch, n_aoa, axis=0)
                tiled_aoa = np.tile(self.standardized_aoa, batch_size)[:, None]
                design_tensor = torch.as_tensor(
                    repeated_designs,
                    dtype=torch.float32,
                    device=self.device,
                )
                aoa_tensor = torch.as_tensor(
                    tiled_aoa,
                    dtype=torch.float32,
                    device=self.device,
                )
                random_features = self.model.random_features(design_tensor, aoa_tensor)
                prediction = self.model.beta(random_features).reshape(batch_size, n_aoa)
                grouped_features = random_features.reshape(
                    batch_size,
                    n_aoa,
                    random_features.shape[-1],
                )

                covariance = self.model.covariance_from_random_features(grouped_features)
                covariance = covariance.to(prediction.dtype)

                means.append(prediction.cpu().numpy().astype(np.float64))
                covariances.append(covariance.cpu().numpy().astype(np.float64))

        return PredictiveDistribution(
            mean=np.concatenate(means, axis=0),
            covariance=np.concatenate(covariances, axis=0),
        )
