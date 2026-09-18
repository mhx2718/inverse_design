# GUIDe

**Generative and Uncertainty-Informed Inverse Design for On-Demand Nonlinear Functional Responses**

Inverse design for airfoils and nacre-inspired composites using pretrained
probabilistic forward models, GUIDe sampling, and baseline methods. Data,
preprocessing parameters, model weights, and inference settings are included.

## Install

Python 3.10–3.13. Run commands from the repository root.

```bash
# Airfoil
python -m pip install -e ".[airfoil]"

# Nacre
python -m pip install -e ".[nacre]"

# Airfoil and nacre
python -m pip install -e ".[airfoil,nacre]"
```

## Run

```bash
python -m guide --config configs/airfoil_demo.yaml --method guide --target-index 0 --device cuda
python -m guide --config configs/nacre_demo.yaml --method guide --target-index 0 --device cuda

# All methods for selected targets
python scripts/run_benchmark.py --config configs/airfoil_demo.yaml --target-indices 0 --device cuda
python scripts/run_benchmark.py --config configs/nacre_demo.yaml --target-indices 0 --device cuda

# Equivalent example wrappers
bash examples/airfoil/run_benchmark.sh --target-indices 0 --device cuda
bash examples/nacre/run_benchmark.sh --target-indices 0 --device cuda
```

| Method | Selection criterion | Benchmark |
|---|---|---|
| `guide` | Response-tolerance probability | Airfoil, nacre |
| `ga` | Predictive-mean squared error | Airfoil, nacre |
| `cdm` | Conditional diffusion | Airfoil, nacre |
| `cdm-s` | CDM candidates ranked by tolerance probability | Airfoil, nacre |
| `mcmc-bi` | Predictive density at the target | Airfoil |
| `abc-mcmc` | Maximum response discrepancy | Airfoil |
| `random-search` | Predictive-mean squared error | Airfoil |

Use `--seed 1` for another generation seed and `--output outputs/my_run` for a
separate result directory. The default generation seed is 0. Use `--device cuda:0`
for GPU execution. Nacre uses a fixed bank of 30 dropout samples with its own
seed in `forward_model`.

The batch runner defaults to all targets and all supported methods. Select a
subset with `--target-indices 0 1` and `--methods guide cdm cdm-s ga`. Use
`--output-root` to choose the batch output directory, `--timeout` to limit each
run in seconds, or `--dry-run` to preview commands. Each run has a `run.log`;
failures are reported after the remaining runs finish. Both example directories
provide `run_demo.sh` for one GUIDe target and `run_benchmark.sh` for batch runs.

## Configuration and targets

`configs/airfoil_demo.yaml` and `configs/nacre_demo.yaml` use small generation
budgets. Select `configs/airfoil_paper.yaml` or `configs/nacre_paper.yaml` for
larger budgets. Set GUIDe budgets in `support_search`, `mcmc`, and `output`;
baseline settings live in `configs/airfoil` and `configs/nacre`.
Paths in YAML resolve relative to `project_root`.

Both benchmarks evaluate joint tolerance probabilities with
`scipy.stats._mvn.mvnun`. The dependency range `scipy>=1.11,<1.16` provides this
routine. Integration budgets and tolerances are configured in `likelihood`.

Airfoil provides 50 targets, indexed 0–49, with default tolerance 0.025.
Nacre provides two 100-point stress curves on a 0–4% strain grid:

| Index | Target | Tolerance |
|---|---|---|
| 0 | `brittle` | 8.3% of peak stress |
| 1 | `plateau` | 7.5% of peak stress |

All nacre response points, including the zero tails, are included. Override
the target with `--target-index` or set an absolute tolerance with `--tolerance`.

## Files and outputs

| Directory | Contents |
|---|---|
| `src/guide/` | Shared algorithms and benchmark implementations |
| `configs/` | Model, generation, and method settings |
| `data/airfoil/` | Airfoil data, targets, PCA/scalers, and constraints |
| `data/nacre/` | Nacre dataset, targets, and input scaler |
| `checkpoints/` | Airfoil SNGP/CDM and nacre LSTM/CDM weights |
| `examples/` | Run commands for each benchmark |
| `scripts/` | Shared benchmark runner |

Outputs use `outputs/demo/<benchmark>/<method>/<target>` for demo configs and
`outputs/<benchmark>/<method>/<target>` for paper configs. Both the single-target
CLI and the batch runner respect `experiment.output_root`.

Each run saves `generation.npz` (`designs`, `scores`, and optional
`predicted_mean`), `generation_manifest.json` (method and score definition),
`target.npz`, and `run_configuration.json`. Designs use standardized model
coordinates; nacre also saves `physical_designs.npz`.

Airfoil designs have 16 standardized PCA coordinates; nacre designs have 10
standardized material parameters. Each benchmark uses its bundled
`preprocessor.npz`. Nacre `dataset.npz` contains `X_train`, `X_val`, `X_test`
with shape `(N, 100, 11)` (10 design parameters followed by strain) and
`y_train`, `y_val`, `y_test` with shape `(N, 100)` in MPa. Nacre
`targets.npz` contains the response curves, strains, target IDs, tolerances,
and response masks.

The four pretrained weights are `checkpoints/airfoil_sngp.pt`,
`checkpoints/airfoil_cdm.pt`, `checkpoints/nacre_forward.h5`, and
`checkpoints/nacre_cdm.pt`.

The airfoil SNGP and nacre CDM checkpoints use lossless tensor compression.
The model loaders decode them automatically at their original numerical precision.

MIT License. See [CITATION.cff](CITATION.cff) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
