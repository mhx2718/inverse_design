"""Airfoil representation and geometric constraints."""

from .constraints import LinearInequalityConstraint
from .representation import AirfoilPCARepresentation

__all__ = ["AirfoilPCARepresentation", "LinearInequalityConstraint"]
