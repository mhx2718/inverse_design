"""Forward and generative models used by the paper implementation."""

from .sngp import AirfoilSNGPForwardModel, load_airfoil_sngp

__all__ = [
    "AirfoilSNGPForwardModel",
    "load_airfoil_sngp",
]
