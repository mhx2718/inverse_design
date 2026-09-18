#!/usr/bin/env python3
"""Run selected methods and targets for the airfoil or nacre benchmark."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np

from guide.config import load_yaml

METHODS = {
    "airfoil": ("guide", "cdm", "cdm-s", "ga", "mcmc-bi", "abc-mcmc", "random-search"),
    "nacre": ("guide", "cdm", "cdm-s", "ga"),
}


def load_target_ids(config, root, tolerance):
    """Use the same target names and validation as single-target generation."""
    target_path = root / config["paths"]["targets"]
    if config.get("benchmark", "airfoil") == "airfoil":
        from guide.airfoil.data import load_targets

        indices = config["paths"].get("target_indices")
        targets, _ = load_targets(target_path, root / indices if indices else None)
        return [f"airfoil-target-{i:03d}" for i in range(len(targets))]

    from guide.nacre.representation import NacreRepresentation
    from guide.nacre.targets import load_nacre_target

    representation = NacreRepresentation.load(root / config["paths"]["preprocessor"])
    if target_path.suffix == ".npz":
        with np.load(target_path, allow_pickle=False) as data:
            responses = data["responses"]
    else:
        responses = np.load(target_path, allow_pickle=False)
    count = 1 if responses.ndim == 1 else len(responses)
    experiment = config["experiment"]
    return [
        load_nacre_target(
            target_path, index,
            tolerance=tolerance if tolerance is not None else experiment.get("tolerance"),
            response_points=representation.response_dimension,
            mask=experiment.get("response_mask"),
        )[0].target_id
        for index in range(count)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Benchmark YAML configuration.")
    parser.add_argument("--methods", nargs="+", help="Methods to run; defaults to all methods for the benchmark.")
    parser.add_argument("--target-indices", "--target-index", dest="target_indices", nargs="+", type=int,
                        help="Target indices to run; defaults to all targets in the configured file.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--output-root", type=Path, help="Override experiment.output_root; relative to the current directory.")
    parser.add_argument("--timeout", type=float, help="Optional time limit in seconds per method and target.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them or creating outputs.")
    args = parser.parse_args()
    for name in ("timeout", "tolerance"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name} must be finite and positive.")
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    benchmark = config.get("benchmark", "airfoil")
    if benchmark not in METHODS:
        parser.error(f"Unsupported benchmark: {benchmark}")
    methods = list(dict.fromkeys(args.methods if args.methods is not None else METHODS[benchmark]))
    invalid = [method for method in methods if method not in METHODS[benchmark]]
    if invalid:
        parser.error(f"Unsupported {benchmark} methods: {', '.join(invalid)}. Choose from {', '.join(METHODS[benchmark])}.")
    root = (config_path.parent / config.get("project_root", ".")).resolve()
    target_ids = load_target_ids(config, root, args.tolerance)
    if not target_ids or len(set(target_ids)) != len(target_ids):
        parser.error("The configured target file must contain nonempty, unique target IDs.")
    indices = list(dict.fromkeys(args.target_indices if args.target_indices is not None else range(len(target_ids))))
    if any(index < 0 or index >= len(target_ids) for index in indices):
        parser.error(f"Target indices must be between 0 and {len(target_ids) - 1}.")
    output = args.output_root.resolve() if args.output_root else (
        root / config["experiment"].get("output_root", f"outputs/{benchmark}")
    ).resolve()
    print(f"{benchmark}: {len(indices)} targets x {len(methods)} methods = {len(indices) * len(methods)} runs", flush=True)
    failures = []
    for index in indices:
        for method in methods:
            run_dir = output / method / target_ids[index]
            command = [sys.executable, "-m", "guide", "--benchmark", benchmark,
                       "--config", str(config_path), "--method", method,
                       "--target-index", str(index), "--device", args.device,
                       "--output", str(run_dir)]
            for option in ("seed", "tolerance"):
                value = getattr(args, option)
                if value is not None:
                    command += [f"--{option}", str(value)]
            if args.dry_run:
                print(shlex.join(command))
                continue
            label = f"{method}/{target_ids[index]}"
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "run.log"
            print(f"Running {label}", flush=True)
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=args.timeout, check=False)
                if result.returncode:
                    failures.append(label)
                    print(f"Failed {label}; see {log_path}", flush=True)
                else:
                    print(f"Saved {run_dir}", flush=True)
            except subprocess.TimeoutExpired:
                failures.append(label)
                print(f"Timed out {label}; see {log_path}", flush=True)
    if failures:
        raise SystemExit(f"Failed runs: {', '.join(failures)}")


if __name__ == "__main__":
    main()
