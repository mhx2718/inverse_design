"""Nacre input scaling and physical design coordinates."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

DESIGN_NAMES = (
    "sigma_n_y", "sigma_n_d", "delta_n_1", "delta_n_2", "delta_n_3",
    "sigma_s_y", "sigma_s_d", "delta_s_1", "delta_s_2", "delta_s_3",
)


@dataclass
class NacreRepresentation:
    """Preserve the fitted input scaler and physical strain grid.

    ``mean`` and ``scale`` have 11 entries: ten cohesive parameters followed by
    strain. Stress responses are in MPa and are not transformed by this scaler.
    """

    mean: ArrayLike
    scale: ArrayLike
    physical_strains: ArrayLike

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        self.scale = np.asarray(self.scale, dtype=np.float64).reshape(-1)
        self.physical_strains = np.asarray(self.physical_strains, dtype=np.float64).reshape(-1)
        if self.mean.shape != (11,) or self.scale.shape != (11,):
            raise ValueError("The nacre input scaler must contain exactly 11 features.")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale)):
            raise ValueError("Scaler statistics must be finite.")
        if np.any(self.scale <= 0):
            raise ValueError("Scaler scales must be positive.")
        if (len(self.physical_strains) < 2 or not np.all(np.isfinite(self.physical_strains))
                or np.any(np.diff(self.physical_strains) <= 0)):
            raise ValueError("physical_strains must be finite and strictly increasing.")

    @property
    def design_dimension(self):
        return 10

    @property
    def response_dimension(self):
        return len(self.physical_strains)

    @property
    def model_strains(self):
        return (self.physical_strains - self.mean[10]) / self.scale[10]

    def _designs(self, designs):
        z = np.asarray(designs, dtype=np.float64)
        if z.ndim == 1:
            z = z[None, :]
        if z.ndim != 2 or z.shape[1] != 10:
            raise ValueError(f"designs must have shape (n, 10), got {z.shape}.")
        return z

    def transform_designs(self, designs):
        return (self._designs(designs) - self.mean[:10]) / self.scale[:10]

    def inverse_transform_designs(self, designs):
        return self._designs(designs) * self.scale[:10] + self.mean[:10]

    def make_inputs(self, designs):
        z = self._designs(designs)
        if not np.all(np.isfinite(z)):
            raise ValueError("Model designs must be finite.")
        x = np.empty((len(z), self.response_dimension, 11), dtype=np.float32)
        x[:, :, :10] = z[:, None, :]
        x[:, :, 10] = self.model_strains[None, :]
        return x

    @classmethod
    def load(cls, path: str | Path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing nacre preprocessor: {path}.")
        with np.load(path, allow_pickle=False) as arrays:
            return cls(arrays["mean"], arrays["scale"], arrays["physical_strains"])
