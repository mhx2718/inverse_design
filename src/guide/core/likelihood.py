"""Tolerance-tube likelihood and GUIDe support-localization objective."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

from guide.interfaces import ProbabilisticForwardModel
from guide.types import PredictiveDistribution, TargetSpecification

from .probability import RectangleProbability


def regularize_covariance(
    covariance: ArrayLike,
    *,
    scale: float = 1.0,
    jitter: float = 0.0,
) -> NDArray[np.float64]:
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    matrix = scale * 0.5 * (matrix + matrix.T)
    if jitter < 0:
        raise ValueError("jitter must be non-negative.")
    if jitter:
        matrix = matrix + jitter * np.eye(matrix.shape[0], dtype=np.float64)
    return matrix


@dataclass
class ToleranceTubeLikelihood:
    """Joint tolerance probability over the target's selected response points."""

    forward_model: ProbabilisticForwardModel
    rectangle_probability: RectangleProbability
    covariance_scale: float = 1.0
    covariance_jitter: float = 1e-6
    minimum_probability: float = 1e-300

    def predictive_distribution(self, designs: ArrayLike) -> PredictiveDistribution:
        return self.forward_model.predict_distribution(designs)

    def probability(
        self,
        designs: ArrayLike,
        target: TargetSpecification,
        *,
        distribution: PredictiveDistribution | None = None,
    ) -> NDArray[np.float64]:
        if distribution is None:
            distribution = self.predictive_distribution(designs)
        probabilities = np.empty(distribution.mean.shape[0], dtype=np.float64)
        mask = target.response_mask
        lower = (target.response - target.tolerance)[mask]
        upper = (target.response + target.tolerance)[mask]

        if distribution.mean.shape[1] != target.response.size:
            raise ValueError(
                "Forward-model response dimension does not match the target: "
                f"{distribution.mean.shape[1]} versus {target.response.size}."
            )

        for index, (mean, covariance) in enumerate(
            zip(distribution.mean, distribution.covariance, strict=True)
        ):
            effective_covariance = regularize_covariance(
                covariance[np.ix_(mask, mask)],
                scale=self.covariance_scale,
                jitter=self.covariance_jitter,
            )
            probabilities[index] = self.rectangle_probability.probability(
                lower,
                upper,
                mean[mask],
                effective_covariance,
            )
        return np.clip(probabilities, 0.0, 1.0)

    def log_probability(
        self,
        designs: ArrayLike,
        target: TargetSpecification,
        *,
        distribution: PredictiveDistribution | None = None,
    ) -> NDArray[np.float64]:
        probabilities = self.probability(designs, target, distribution=distribution)
        return np.log(np.maximum(probabilities, self.minimum_probability))


@dataclass
class GUIDeSupportObjective:
    """Covariance-aware objective used to locate non-negligible support.

    The manuscript form is

    ``-log(t + d_M/sqrt(k)) + lambda/k * sum(log marginal_coverage)``.

    ``distance_statistic='squared'`` uses the squared Mahalanobis statistic
    in the logarithm.
    """

    likelihood: ToleranceTubeLikelihood
    coverage_weight: float = 1.0
    stabilizer: float = 1e-12
    support_covariance_jitter: float = 1e-3
    distance_statistic: str = "mahalanobis"
    minimum_marginal_probability: float = 1e-20

    def evaluate(
        self,
        designs: ArrayLike,
        target: TargetSpecification,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        distribution = self.likelihood.predictive_distribution(designs)
        probabilities = self.likelihood.probability(
            designs,
            target,
            distribution=distribution,
        )
        scores = np.empty(distribution.mean.shape[0], dtype=np.float64)
        mask = target.response_mask
        response = target.response[mask]
        tolerance = target.tolerance[mask]
        response_dimension = response.size
        root_dimension = np.sqrt(float(response_dimension))

        for index, (mean, covariance) in enumerate(
            zip(distribution.mean, distribution.covariance, strict=True)
        ):
            effective_covariance = regularize_covariance(
                covariance[np.ix_(mask, mask)],
                scale=self.likelihood.covariance_scale,
                jitter=self.support_covariance_jitter,
            )
            mean = mean[mask]
            difference = mean - response
            squared_distance = float(
                difference @ np.linalg.solve(effective_covariance, difference)
            )
            squared_distance = max(squared_distance, 0.0)
            if self.distance_statistic == "mahalanobis":
                distance_value = np.sqrt(squared_distance)
            elif self.distance_statistic == "squared":
                distance_value = squared_distance
            else:
                raise ValueError(
                    "distance_statistic must be 'mahalanobis' or 'squared'."
                )

            distance_term = -np.log(
                self.stabilizer + distance_value / root_dimension
            )
            marginal_std = np.sqrt(np.maximum(np.diag(effective_covariance), 1e-300))
            standardized_upper = (
                response + tolerance - mean
            ) / marginal_std
            standardized_lower = (
                response - tolerance - mean
            ) / marginal_std
            marginal_coverage = ndtr(standardized_upper) - ndtr(standardized_lower)
            coverage_term = (
                self.coverage_weight
                / response_dimension
                * np.sum(
                    np.log(
                        np.maximum(
                            marginal_coverage,
                            self.minimum_marginal_probability,
                        )
                    )
                )
            )
            scores[index] = distance_term + coverage_term

        return scores, probabilities


@dataclass
class PredictiveMeanMSE:
    forward_model: object

    def score(self, designs: ArrayLike, target: TargetSpecification) -> NDArray[np.float64]:
        predictions = np.asarray(self.forward_model.predict_mean(designs), dtype=np.float64)
        residual = (predictions - target.response[None, :])[:, target.response_mask]
        return np.mean(residual ** 2, axis=1)


@dataclass
class GaussianABCScore:
    forward_model: object

    def probability(
        self,
        designs: ArrayLike,
        target: TargetSpecification,
        *,
        bandwidth: float | None = None,
    ) -> NDArray[np.float64]:
        predictions = np.asarray(self.forward_model.predict_mean(designs), dtype=np.float64)
        residual = (predictions - target.response[None, :])[:, target.response_mask]
        max_error = np.max(np.abs(residual), axis=1)
        if bandwidth is None:
            tolerance = target.tolerance[target.response_mask]
            if not np.allclose(tolerance, tolerance[0]):
                raise ValueError("A scalar bandwidth is required for nonuniform tolerances.")
            bandwidth = float(tolerance[0])
        if bandwidth <= 0:
            raise ValueError("ABC bandwidth must be positive.")
        return np.exp(-(max_error**2) / (2.0 * bandwidth**2))


@dataclass
class PredictivePointDensity:
    forward_model: ProbabilisticForwardModel
    covariance_scale: float = 1.0
    covariance_jitter: float = 1e-6

    def log_density(
        self,
        designs: ArrayLike,
        target: TargetSpecification,
    ) -> NDArray[np.float64]:
        distribution = self.forward_model.predict_distribution(designs)
        output = np.empty(distribution.mean.shape[0], dtype=np.float64)
        mask = target.response_mask
        k = int(np.count_nonzero(mask))
        constant = k * np.log(2.0 * np.pi)
        for index, (mean, covariance) in enumerate(
            zip(distribution.mean, distribution.covariance, strict=True)
        ):
            effective_covariance = regularize_covariance(
                covariance[np.ix_(mask, mask)],
                scale=self.covariance_scale,
                jitter=self.covariance_jitter,
            )
            sign, log_determinant = np.linalg.slogdet(effective_covariance)
            if sign <= 0:
                output[index] = -np.inf
                continue
            residual = (target.response - mean)[mask]
            quadratic = residual @ np.linalg.solve(effective_covariance, residual)
            output[index] = -0.5 * (quadratic + log_determinant + constant)
        return output
