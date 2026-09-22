"""Command-line entry point for one-target airfoil generation experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from guide.airfoil.constraints import LinearInequalityConstraint
from guide.airfoil.data import (
    AirfoilDataset,
    draw_admissible_gaussian_initial_design,
    load_targets,
)
from guide.airfoil.representation import AirfoilPCARepresentation
from guide.core.pipeline import GUIDeGenerationConfig, GUIDeGenerator
from guide.baselines import (
    ABCMCMCBaseline,
    ABCMCMCConfig,
    CDMBaseline,
    CDMConfig,
    CDMScreeningBaseline,
    GAConfig,
    GeneticAlgorithmBaseline,
    MCMCBIBaseline,
    MCMCBIConfig,
    RandomSearchBaseline,
    RandomSearchConfig,
    ScreeningConfig,
)
from guide.config import load_yaml
from guide.core import (
    GUIDeSupportObjective,
    MCMCConfig,
    ParticleSwarmSupportFinder,
    PSOConfig,
    MVNUNRectangle,
    ToleranceTubeLikelihood,
)
from guide.io import save_generation_result
from guide.models.diffusion import (
    ConditionalDiffusionGenerator,
    ConditionalUNetConfig,
    DiffusionConfig,
    load_conditional_diffusion_model,
)
from guide.models.sngp import (
    AirfoilSNGPConfig,
    AirfoilSNGPForwardModel,
    load_airfoil_sngp,
)
from guide.types import TargetSpecification
from guide.utils import json_ready, set_global_seed

METHOD_CONFIG_NAMES = {
    "ga": "ga.yaml",
    "cdm": "cdm.yaml",
    "cdm-s": "cdm-s.yaml",
    "mcmc-bi": "mcmc-bi.yaml",
    "abc-mcmc": "abc-mcmc.yaml",
    "random-search": "random-search.yaml",
}


def _resolve_project_paths(
    config_path: Path,
    config: dict[str, Any],
) -> tuple[Path, dict[str, Path]]:
    project_root = (config_path.parent / config.get("project_root", ".")).resolve()
    resolved = {
        key: (project_root / value).resolve()
        for key, value in config.get("paths", {}).items()
    }
    return project_root, resolved


def _require(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {label}: {path}\n"
            "See README.md for the bundled data and checkpoints."
        )
    return path


def _load_target(
    target_path: Path,
    target_index: int,
    tolerance: float,
) -> TargetSpecification:
    targets = load_targets(target_path)
    if not 0 <= target_index < targets.shape[0]:
        raise IndexError(
            f"target_index={target_index} is outside [0, {targets.shape[0] - 1}]."
        )
    metadata: dict[str, Any] = {
        "source": str(target_path),
        "index": target_index,
    }
    return TargetSpecification(
        response=targets[target_index],
        tolerance=tolerance,
        target_id=f"airfoil-target-{target_index:03d}",
        metadata=metadata,
    )


def _build_sngp(
    config: dict[str, Any],
    paths: dict[str, Path],
    device: str | None,
    standardized_aoa: np.ndarray,
) -> AirfoilSNGPForwardModel:
    model_config = AirfoilSNGPConfig(**config["forward_model"]["sngp"])
    model = load_airfoil_sngp(
        _require(paths["sngp_checkpoint"], "SNGP checkpoint"),
        config=model_config,
        device=device,
        strict=True,
    )
    return AirfoilSNGPForwardModel(
        model,
        standardized_aoa=standardized_aoa,
        mean_batch_size=int(config["forward_model"].get("mean_batch_size", 4096)),
        distribution_batch_size=int(
            config["forward_model"].get("distribution_batch_size", 16)
        ),
    )


def _covariance_scale(settings: dict[str, Any]) -> float:
    value = settings.get("covariance_scale", 1.0)
    if value is None:
        raise ValueError(
            "Set likelihood.covariance_scale in the selected YAML. "
            "Use the scale calibrated for this exact model; "
            "1.0 explicitly requests uncalibrated covariance."
        )
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("covariance_scale must be finite and positive.")
    return value


def _build_likelihood(
    config: dict[str, Any],
    forward_model: AirfoilSNGPForwardModel,
) -> ToleranceTubeLikelihood:
    settings = config["likelihood"]
    if settings.get("backend", "mvnun") != "mvnun":
        raise ValueError("The likelihood backend must be mvnun.")
    backend = MVNUNRectangle(**{
        key: settings[key]
        for key in ("max_points", "absolute_tolerance", "relative_tolerance", "strict_convergence")
        if key in settings
    })
    return ToleranceTubeLikelihood(
        forward_model=forward_model,
        rectangle_probability=backend,
        covariance_scale=_covariance_scale(settings),
        covariance_jitter=float(settings.get("covariance_jitter", 1e-6)),
    )


def _load_method_config(
    project_root: Path,
    method: str,
    override: str | None,
    directory: str,
) -> dict[str, Any]:
    path = (
        Path(override).resolve()
        if override is not None
        else project_root / directory / METHOD_CONFIG_NAMES[method]
    )
    return load_yaml(_require(path, f"{method} configuration"))


def _build_cdm(
    method_config: dict[str, Any],
    paths: dict[str, Path],
    constraint: LinearInequalityConstraint,
    device: str | None,
) -> CDMBaseline:
    network_config = ConditionalUNetConfig(**method_config["network"])
    diffusion_config = DiffusionConfig(**method_config["diffusion"])
    model = load_conditional_diffusion_model(
        _require(paths["cdm_checkpoint"], "CDM checkpoint"),
        network_config=network_config,
        device=device,
        strict=True,
    )
    generator = ConditionalDiffusionGenerator(
        model,
        constraint,
        config=diffusion_config,
    )
    return CDMBaseline(generator, config=CDMConfig(**method_config["parameters"]))


def run_from_args(arguments: argparse.Namespace) -> Path:
    config_path = Path(arguments.config).resolve()
    common = load_yaml(config_path)
    project_root, paths = _resolve_project_paths(config_path, common)
    experiment = common["experiment"]
    target_index = (
        int(arguments.target_index)
        if arguments.target_index is not None
        else int(experiment.get("target_index", 0))
    )
    tolerance = (
        float(arguments.tolerance)
        if arguments.tolerance is not None
        else float(experiment["tolerance"])
    )
    seed = int(arguments.seed if arguments.seed is not None else experiment.get("seed", 0))
    set_global_seed(seed)
    # Seed the optimizers and chains; mvnun exposes no Python seed argument.
    if arguments.seed is not None:
        for key in ("support_search", "mcmc"):
            common.setdefault(key, {})["seed"] = seed

    dataset = AirfoilDataset.load(_require(paths["dataset"], "airfoil dataset"))
    representation = AirfoilPCARepresentation.load(
        _require(paths["preprocessor"], "airfoil preprocessor")
    )
    target = _load_target(
        _require(paths["targets"], "airfoil targets"),
        target_index,
        tolerance,
    )
    constraint = LinearInequalityConstraint.load(
        _require(paths["constraints"], "airfoil constraint asset")
    )
    method = arguments.method
    pool_config = None
    method_config = (
        _load_method_config(
            project_root,
            method,
            arguments.method_config,
            experiment.get("method_config_directory", "configs/airfoil/baselines"),
        )
        if method != "guide"
        else None
    )
    if method_config is not None and arguments.seed is not None:
        if method != "cdm-s":
            method_config.setdefault("parameters", {})["seed"] = seed
        for section in ("mcmc", "diffusion"):
            if section in method_config:
                method_config[section]["seed"] = seed
    if method in {"random-search", "mcmc-bi", "abc-mcmc"}:
        training_latent = dataset.train.latent

    # All methods use the same SNGP mean; probability-based methods also use its covariance.
    forward = _build_sngp(
        common, paths, arguments.device,
        representation.transform_aoa(dataset.angles_of_attack),
    )
    sngp_label = paths["sngp_checkpoint"].stem

    if method == "guide":
        likelihood = _build_likelihood(common, forward)
        support_settings = common["support_search"]
        objective = GUIDeSupportObjective(
            likelihood,
            coverage_weight=float(support_settings.get("coverage_weight", 1.0)),
            support_covariance_jitter=float(
                support_settings.get("support_covariance_jitter", 1e-3)
            ),
            distance_statistic=str(
                support_settings.get("distance_statistic", "mahalanobis")
            ),
        )
        support_config = PSOConfig(
            **{
                key: value
                for key, value in support_settings.items()
                if key
                in {
                    "n_particles",
                    "iterations",
                    "cognitive",
                    "social",
                    "inertia",
                    "lower_bound",
                    "upper_bound",
                    "nonzero_threshold",
                    "invalid_base_penalty",
                    "invalid_violation_scale",
                    "seed",
                }
            }
        )
        support_finder = ParticleSwarmSupportFinder(
            objective,
            constraint,
            dimension=constraint.design_dimension,
            config=support_config,
        )
        output_settings = common.get("output", {})
        generator = GUIDeGenerator(
            likelihood,
            support_finder,
            constraint,
            config=GUIDeGenerationConfig(
                mcmc=MCMCConfig(**common["mcmc"]),
                unique_decimals=output_settings.get("unique_decimals"),
                n_output=output_settings.get("n_output"),
                selection=output_settings.get("selection", "linspace"),
            ),
        )
        generation = generator.run(target)

    elif method == "ga":
        generation = GeneticAlgorithmBaseline(
            forward,
            constraint,
            config=GAConfig(**method_config["parameters"]),
        ).run(target)

    elif method == "random-search":
        generation = RandomSearchBaseline(
            forward,
            constraint,
            training_latent,
            config=RandomSearchConfig(**method_config["parameters"]),
        ).run(target)

    elif method == "mcmc-bi":
        parameters = dict(method_config["parameters"])
        if parameters.get("covariance_scale") is None:
            parameters["covariance_scale"] = _covariance_scale(common["likelihood"])
        else:
            parameters["covariance_scale"] = _covariance_scale(parameters)
        baseline = MCMCBIBaseline(
            forward,
            constraint,
            config=MCMCBIConfig(
                mcmc=MCMCConfig(**method_config["mcmc"]),
                **parameters,
            ),
        )
        generation = baseline.run(
            target,
            initial_design=draw_admissible_gaussian_initial_design(
                training_latent,
                constraint,
                seed=seed,
            ),
        )

    elif method == "abc-mcmc":
        parameters = dict(method_config["parameters"])
        baseline = ABCMCMCBaseline(
            forward,
            constraint,
            config=ABCMCMCConfig(
                mcmc=MCMCConfig(**method_config["mcmc"]),
                **parameters,
            ),
        )
        generation = baseline.run(
            target,
            initial_design=draw_admissible_gaussian_initial_design(
                training_latent,
                constraint,
                seed=seed,
            ),
        )

    elif method == "cdm":
        generation = _build_cdm(
            method_config,
            paths,
            constraint,
            arguments.device,
        ).run(target)
        generation.predicted_mean = forward.predict_mean(generation.designs)
        generation.metadata["prediction_model"] = f"{sngp_label}_mean"

    elif method == "cdm-s":
        cdm_config_path = project_root / method_config.get(
            "cdm_config",
            "configs/airfoil/baselines/cdm.yaml",
        )
        pool_config = load_yaml(_require(cdm_config_path, "CDM configuration"))
        if arguments.seed is not None:
            pool_config["parameters"]["seed"] = seed
            pool_config["diffusion"]["seed"] = seed
        cdm = _build_cdm(
            pool_config,
            paths,
            constraint,
            arguments.device,
        )
        generation = CDMScreeningBaseline(
            cdm,
            _build_likelihood(common, forward),
            config=ScreeningConfig(**method_config["parameters"]),
        ).run(target)

    else:  # pragma: no cover - argparse prevents this
        raise ValueError(f"Unknown method: {method}")

    generation.metadata.setdefault("forward_model", sngp_label)
    generation.metadata.setdefault("target_source", str(paths["dataset"]))

    output_root = project_root / experiment.get("output_root", "outputs")
    output_directory = (
        Path(arguments.output).resolve()
        if arguments.output
        else output_root / method / target.target_id
    )
    save_generation_result(generation, output_directory)
    np.savez_compressed(
        output_directory / "target.npz",
        response=target.response,
        tolerance=target.tolerance,
        response_mask=target.response_mask,
    )
    (output_directory / "run_configuration.json").write_text(
        json.dumps(
            json_ready(
                {
                    "common_config": common,
                    "method_config": method_config,
                    "cdm_pool_config": pool_config,
                    "method": method,
                    "target_index": target_index,
                    "target_metadata": dict(target.metadata),
                    "tolerance": tolerance,
                    "seed": seed,
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GUIDe or an airfoil inverse-design baseline for one target."
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "guide",
            "ga",
            "cdm",
            "cdm-s",
            "mcmc-bi",
            "abc-mcmc",
            "random-search",
        ],
    )
    parser.add_argument("--config", default="configs/airfoil_demo.yaml")
    parser.add_argument("--method-config")
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="For example: cpu, cuda, or cuda:0")
    parser.add_argument("--output")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    output = run_from_args(arguments)
    print(f"Saved result bundle to {output}")


if __name__ == "__main__":
    main()
