"""Command-line entry point for airfoil and nacre inverse design."""
from __future__ import annotations

import argparse
from pathlib import Path

from guide.config import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GUIDe inverse design: airfoil or nacre.")
    parser.add_argument("--benchmark", choices=["airfoil", "nacre"])
    parser.add_argument("--method", required=True, choices=["guide", "ga", "cdm", "cdm-s", "mcmc-bi", "abc-mcmc", "random-search"])
    parser.add_argument("--config", default="configs/airfoil_demo.yaml")
    parser.add_argument("--method-config")
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="For example cpu or cuda:0; TensorFlow uses its runtime device policy.")
    parser.add_argument("--output", "--output-dir", dest="output")
    return parser


def run_from_args(arguments: argparse.Namespace) -> Path:
    config = load_yaml(arguments.config)
    configured = config.get("benchmark", "airfoil")
    requested = getattr(arguments, "benchmark", None)
    if requested is not None and requested != configured:
        raise ValueError(f"--benchmark {requested} conflicts with configured benchmark {configured}.")
    if configured == "airfoil":
        from guide.airfoil.cli import run_from_args as run
    elif configured == "nacre":
        from guide.nacre.cli import run_from_args as run
    else:
        raise ValueError(f"Unknown benchmark: {configured}")
    return run(arguments)


def main() -> None:
    output = run_from_args(build_parser().parse_args())
    print(f"Saved result bundle to {output}")


if __name__ == "__main__":
    main()
