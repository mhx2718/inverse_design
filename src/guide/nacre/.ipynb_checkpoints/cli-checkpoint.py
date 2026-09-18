"""Nacre inverse design with GUIDe, CDM, CDM-S, and GA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from guide.config import load_yaml
from guide.core.likelihood import GUIDeSupportObjective, ToleranceTubeLikelihood
from guide.core.mcmc import MCMCConfig
from guide.core.pipeline import GUIDeGenerationConfig, GUIDeGenerator
from guide.core.probability import MVNUNRectangle
from guide.core.support import ParticleSwarmSupportFinder, PSOConfig
from guide.io import save_generation_result
from guide.nacre.targets import load_nacre_target
from guide.utils import json_ready, set_global_seed

SUPPORTED_METHODS = ("guide", "ga", "cdm", "cdm-s")


def require(path, label):
    if not path.is_file():
        raise FileNotFoundError(f"Missing nacre {label}: {path}.")
    return path


def build_forward(config, paths, representation, device=None):
    from guide.nacre.forward import NacreMCDropoutForwardModel

    checkpoint = require(paths["forward_checkpoint"], "forward checkpoint")
    settings = dict(config["forward_model"])
    if device is not None:
        import os
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        elif device.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        elif device != "cuda":
            raise ValueError("Nacre device must be cpu, cuda or cuda:N.")
    config["forward_model"] = settings
    return NacreMCDropoutForwardModel.from_checkpoint(checkpoint, representation, **settings)


def build_likelihood(config, forward):
    settings = dict(config["likelihood"])
    if settings.pop("backend", "mvnun") != "mvnun":
        raise ValueError("The likelihood backend must be mvnun.")
    # The forward adapter already scales the covariance and adds jitter.
    return ToleranceTubeLikelihood(forward, MVNUNRectangle(**settings), covariance_scale=1.0, covariance_jitter=0.0)


def load_cdm(config, paths, constraint, device):
    from guide.baselines.cdm import CDMBaseline, CDMConfig
    from guide.models.diffusion import DiffusionConfig
    from guide.nacre.diffusion import (
        NacreConditionalDiffusionGenerator,
        NacreConditionalUNetConfig,
        load_nacre_diffusion_model,
    )
    model = load_nacre_diffusion_model(
        require(paths["cdm_checkpoint"], "CDM checkpoint"),
        network_config=NacreConditionalUNetConfig(**config["network"]),
        device=device, strict=True,
    )
    generator = NacreConditionalDiffusionGenerator(model, constraint, config=DiffusionConfig(**config["diffusion"]))
    return CDMBaseline(generator, config=CDMConfig(**config["parameters"]))


def seeded_settings(config, seed):
    config = dict(config)
    for section in ("parameters", "diffusion", "mcmc"):
        if section in config:
            config[section] = {**config[section], "seed": seed}
    return config


def run_from_args(arguments: argparse.Namespace) -> Path:
    method = arguments.method
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported nacre method: {method}. Supported methods: {', '.join(SUPPORTED_METHODS)}."
        )
    from guide.nacre.constraints import NacreCohesiveConstraint
    from guide.nacre.representation import NacreRepresentation

    config_path = Path(arguments.config).resolve()
    config = load_yaml(config_path)
    root = (config_path.parent / config.get("project_root", ".")).resolve()
    paths = {k: (root / v).resolve() for k, v in config["paths"].items()}
    experiment = config["experiment"]
    seed = int(arguments.seed if arguments.seed is not None else experiment.get("seed", 0))
    set_global_seed(seed)
    representation = NacreRepresentation.load(require(paths["preprocessor"], "preprocessor"))
    target_index = int(arguments.target_index if arguments.target_index is not None else experiment.get("target_index", 0))
    tolerance = arguments.tolerance if arguments.tolerance is not None else experiment.get("tolerance")
    target, strains = load_nacre_target(
        require(paths["targets"], "targets"), target_index, tolerance=tolerance,
        response_points=len(representation.physical_strains), mask=experiment.get("response_mask"),
    )
    if strains is not None:
        strains = np.asarray(strains) * float(experiment.get("target_strain_multiplier", 1.0))
    if strains is not None and (strains.shape != representation.physical_strains.shape or not np.allclose(strains, representation.physical_strains, rtol=1e-5, atol=1e-8)):
        raise ValueError("Target strain coordinates differ from the forward model grid. Convert the targets explicitly; do not silently interpolate.")
    constraint = NacreCohesiveConstraint(representation, **config.get("constraints", {}))
    method_config = None
    pool_config = None
    if method != "guide":
        path = Path(arguments.method_config).resolve() if arguments.method_config else root / experiment["method_config_directory"] / f"{method}.yaml"
        method_config = load_yaml(require(path, f"{method} configuration"))
        if method != "cdm-s":
            method_config = seeded_settings(method_config, seed)

    # Pure CDM needs no forward checkpoint.
    forward = None
    if method != "cdm":
        forward = build_forward(config, paths, representation, arguments.device)
    if method == "guide":
        likelihood = build_likelihood(config, forward)
        settings = dict(config["support_search"])
        objective_settings = settings.pop("objective", {})
        settings["seed"] = seed
        config["support_search"]["seed"] = seed
        config["mcmc"]["seed"] = seed
        objective = GUIDeSupportObjective(likelihood, **objective_settings)
        finder = ParticleSwarmSupportFinder(objective, constraint, 10, PSOConfig(**settings))
        mcmc = MCMCConfig(**{**config["mcmc"], "seed": seed})
        result = GUIDeGenerator(likelihood, finder, constraint, config=GUIDeGenerationConfig(mcmc=mcmc, **config.get("output", {}))).run(target)
    elif method == "ga":
        from guide.baselines.ga import GAConfig, GeneticAlgorithmBaseline
        result = GeneticAlgorithmBaseline(forward, constraint, config=GAConfig(**method_config["parameters"])).run(target)
    elif method == "cdm":
        result = load_cdm(method_config, paths, constraint, arguments.device).run(target)
    elif method == "cdm-s":
        from guide.baselines.screening import CDMScreeningBaseline, ScreeningConfig
        pool_config = seeded_settings(load_yaml(root / method_config["cdm_config"]), seed)
        cdm = load_cdm(pool_config, paths, constraint, arguments.device)
        result = CDMScreeningBaseline(cdm, build_likelihood(config, forward), config=ScreeningConfig(**method_config["parameters"])).run(target)
    else:
        raise ValueError(f"Unknown method: {method}")
    result.metadata.update(benchmark="nacre", seed=seed, design_coordinates="standardized", response_units="MPa", forward_settings=config["forward_model"] if forward else None, response_mask=target.response_mask.tolist())
    if method == "ga":
        result.metadata["forward_usage"] = "mean_only"
    output = Path(arguments.output).resolve() if arguments.output else root / experiment.get("output_root", "outputs/nacre") / method / target.target_id
    save_generation_result(result, output)
    physical = representation.inverse_transform_designs(result.designs)
    np.savez_compressed(output / "physical_designs.npz", designs=physical)
    np.savez_compressed(output / "target.npz", response=target.response, tolerance=target.tolerance, response_mask=target.response_mask, strains=representation.physical_strains)
    run_configuration = {"common_config": config, "method_config": method_config, "cdm_pool_config": pool_config, "target_index": target_index, "tolerance": target.tolerance, "seed": seed}
    (output / "run_configuration.json").write_text(json.dumps(json_ready(run_configuration), indent=2) + "\n")
    return output


def build_parser() -> argparse.ArgumentParser:
    from guide.cli import build_parser as build_shared_parser

    parser = build_shared_parser()
    parser.description = "GUIDe nacre inverse design with CDM, CDM-S, and GA baselines."
    for action in parser._actions:
        if action.dest == "method":
            action.choices = SUPPORTED_METHODS
        elif action.dest == "benchmark":
            action.choices = ("nacre",)
    parser.set_defaults(config="configs/nacre_demo.yaml", benchmark="nacre")
    return parser


def main():
    print(f"Saved result bundle to {run_from_args(build_parser().parse_args())}")
