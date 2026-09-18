"""Convert between standardized PCA designs and airfoil contours."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class AirfoilPCARepresentation:
    """Standardized PCA representation of aligned airfoil contours.

    The stored latent coordinates are standardized according to

    ``z = (w - latent_mean) / latent_scale``

    where ``w = (y - pca_mean) @ pca_components.T``.
    """

    common_x: NDArray[np.float64]
    pca_mean: NDArray[np.float64]
    pca_components: NDArray[np.float64]
    latent_mean: NDArray[np.float64]
    latent_scale: NDArray[np.float64]
    aoa_mean: NDArray[np.float64]
    aoa_scale: NDArray[np.float64]
    explained_variance_ratio: NDArray[np.float64]
    n_points_per_surface: int = 96
    trailing_edge_thickness: float = 0.0

    def __post_init__(self) -> None:
        self.common_x = np.asarray(self.common_x, dtype=np.float64).reshape(-1)
        self.pca_mean = np.asarray(self.pca_mean, dtype=np.float64).reshape(-1)
        self.pca_components = np.asarray(self.pca_components, dtype=np.float64)
        self.latent_mean = np.asarray(self.latent_mean, dtype=np.float64).reshape(-1)
        self.latent_scale = np.asarray(self.latent_scale, dtype=np.float64).reshape(-1)
        self.aoa_mean = np.asarray(self.aoa_mean, dtype=np.float64).reshape(-1)
        self.aoa_scale = np.asarray(self.aoa_scale, dtype=np.float64).reshape(-1)
        self.explained_variance_ratio = np.asarray(
            self.explained_variance_ratio,
            dtype=np.float64,
        ).reshape(-1)

        expected_shape_dimension = 2 * int(self.n_points_per_surface)
        if self.common_x.size != self.n_points_per_surface:
            raise ValueError("common_x does not match n_points_per_surface.")
        if self.pca_mean.size != expected_shape_dimension:
            raise ValueError("pca_mean has an incompatible shape dimension.")
        if self.pca_components.ndim != 2:
            raise ValueError("pca_components must have shape (latent_dim, shape_dim).")
        if self.pca_components.shape[1] != expected_shape_dimension:
            raise ValueError("pca_components has an incompatible shape dimension.")
        if self.pca_components.shape[0] != self.latent_mean.size:
            raise ValueError("latent_mean does not match the PCA dimension.")
        if self.latent_scale.shape != self.latent_mean.shape:
            raise ValueError("latent_scale does not match latent_mean.")
        if self.explained_variance_ratio.size != self.latent_dimension:
            raise ValueError("explained_variance_ratio does not match the PCA dimension.")
        if self.aoa_mean.size != 1 or self.aoa_scale.size != 1:
            raise ValueError("The AoA scaler must contain one mean and one scale.")
        if np.any(self.latent_scale <= 0) or np.any(self.aoa_scale <= 0):
            raise ValueError("All scaler values must be positive.")

    @property
    def latent_dimension(self) -> int:
        return int(self.pca_components.shape[0])

    @property
    def shape_dimension(self) -> int:
        return int(self.pca_components.shape[1])

    @property
    def explained_variance_ratio_sum(self) -> float:
        return float(np.sum(self.explained_variance_ratio))

    def standardize_latent(self, latent: ArrayLike) -> NDArray[np.float64]:
        array = np.asarray(latent, dtype=np.float64)
        return (array - self.latent_mean) / self.latent_scale

    def unstandardize_latent(self, standardized_latent: ArrayLike) -> NDArray[np.float64]:
        array = np.asarray(standardized_latent, dtype=np.float64)
        return array * self.latent_scale + self.latent_mean

    def inverse_transform(self, standardized_latent: ArrayLike) -> NDArray[np.float64]:
        latent = np.asarray(standardized_latent, dtype=np.float64)
        if latent.ndim == 1:
            latent = latent[None, :]
        if latent.ndim != 2 or latent.shape[1] != self.latent_dimension:
            raise ValueError(
                f"standardized_latent must have shape (n, {self.latent_dimension})."
            )
        unscaled = self.unstandardize_latent(latent)
        return unscaled @ self.pca_components + self.pca_mean

    def reconstruct_contours(self, standardized_latent: ArrayLike) -> NDArray[np.float64]:
        """Reconstruct closed airfoil contours from standardized latent designs."""

        aligned_y = self.inverse_transform(standardized_latent)
        contours = np.empty(
            (aligned_y.shape[0], 2 * self.n_points_per_surface, 2),
            dtype=np.float64,
        )
        contours[:, : self.n_points_per_surface, 0] = self.common_x[None, :]
        contours[:, : self.n_points_per_surface, 1] = aligned_y[
            :, : self.n_points_per_surface
        ]
        contours[:, self.n_points_per_surface :, 0] = self.common_x[::-1][None, :]
        contours[:, self.n_points_per_surface :, 1] = aligned_y[
            :, self.n_points_per_surface :
        ][:, ::-1]
        return contours

    def transform_aoa(self, aoa: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(aoa, dtype=np.float64).reshape(-1, 1)
        return (values - self.aoa_mean) / self.aoa_scale

    def inverse_transform_aoa(self, standardized_aoa: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(standardized_aoa, dtype=np.float64).reshape(-1, 1)
        return values * self.aoa_scale + self.aoa_mean

    @classmethod
    def load(cls, path: str | Path) -> AirfoilPCARepresentation:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                common_x=data["common_x"],
                pca_mean=data["pca_mean"],
                pca_components=data["pca_components"],
                latent_mean=data["latent_mean"],
                latent_scale=data["latent_scale"],
                aoa_mean=data["aoa_mean"],
                aoa_scale=data["aoa_scale"],
                explained_variance_ratio=data["explained_variance_ratio"],
                n_points_per_surface=int(data["n_points_per_surface"]),
                trailing_edge_thickness=float(data["trailing_edge_thickness"]),
            )
