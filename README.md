# GUIDe: Generative and Uncertainty-Informed Inverse Design for On-Demand Nonlinear Functional Responses📈🧑‍🎨⚙️

🔬 Check out our full manuscript at:

Preprint: https://arxiv.org/abs/2509.05641

## 🗺 Overview

GUIDe generates designs with prescribed nonlinear response curves using probabilistic forward models and statistical inference.

- **Uncertainty-informed design:** sampling is guided by the 📊 probability of satisfying a target within a 📐 user-specified tolerance.
- **Two benchmarks:** 🛩 airfoil inverse design and 🐚 nacre-inspired composite design .
- **Ready-to-run experiments:** datasets, preprocessing settings, pretrained models, and baseline methods are included 📦.

## 🚀 Quickstart

### Installation

Use Python 3.10–3.13. Clone the repository and install both benchmarks:

```bash
python -m pip install -e ".[airfoil,nacre]"
```

To install only one benchmark, use `.[airfoil]` or `.[nacre]` instead. Dependencies and version constraints are defined in [pyproject.toml](pyproject.toml). Run the commands below from the repository root.

### Run GUIDe

```bash
# Airfoil: target 0
python -m guide --config configs/airfoil_demo.yaml --method guide --target-index 0 --device cuda

# Nacre: brittle target
python -m guide --config configs/nacre_demo.yaml --method guide --target-index 0 --device cuda
```

Replace `cuda` with `cpu` to run on the CPU. Use `cuda:0` to select a specific visible GPU.

### Run all methods

```bash
# Airfoil: all methods for target 0
python scripts/run_benchmark.py --config configs/airfoil_demo.yaml --target-indices 0 --device cuda

# Nacre: all methods for both OOD targets
python scripts/run_benchmark.py --config configs/nacre_demo.yaml --target-indices 0 1 --device cuda
```

The single-run command uses `--target-index`; the benchmark runner accepts one or more indices with `--target-indices`. To run an individual baseline, replace `--method guide` in the single-run command with a method below.

## 🪤 Methods

| Method | Approach | Benchmark |
|---|---|---|
| `guide` | Sampling guided by response-tolerance probability | Airfoil, nacre |
| `cdm` | Conditional diffusion | Airfoil, nacre |
| `cdm-s` | CDM candidates ranked by response-tolerance probability | Airfoil, nacre |
| `ga` | Genetic algorithm minimizing predictive-mean squared error | Airfoil, nacre |
| `mcmc-bi` | Bayesian inversion using predictive density at the target | Airfoil |
| `abc-mcmc` | Approximate Bayesian computation using maximum response discrepancy | Airfoil |
| `random-search` | Random candidates ranked by predictive-mean squared error | Airfoil |

## 🎛 Configuration and targets

Demo configurations use small generation budgets. Use the paper configurations for larger budgets:

| Benchmark | Demo configuration | Paper configuration |
|---|---|---|
| Airfoil | `configs/airfoil_demo.yaml` | `configs/airfoil_paper.yaml` |
| Nacre | `configs/nacre_demo.yaml` | `configs/nacre_paper.yaml` |

GUIDe generation budgets are set in `support_search`, `mcmc`, and `output`. Baseline configurations are in `configs/airfoil/` and `configs/nacre/`. Paths in YAML files resolve relative to `project_root`.

**Airfoil:** 50 targets, indexed 0–49, with a default absolute response tolerance of 0.025.

**Nacre:** two OOD stress–strain targets, each sampled at 100 strain points over 0–4% strain:

| Index | Target | Absolute stress tolerance |
|---|---|---|
| 0 | `brittle` | 8.3% of tensile stress |
| 1 | `plateau` | 7.5% of tensile stress |

All nacre response points, including the zero tails, are included. Use `--tolerance` to override the absolute tolerance; nacre stress tolerances are in MPa.

## 🗂 Files and outputs

| Directory | Contents |
|---|---|
| `src/guide/` | Shared algorithms and benchmark implementations |
| `configs/` | Model, generation, and method settings |
| `data/airfoil/` | Airfoil data, targets, PCA/scalers, and constraints |
| `data/nacre/` | Nacre data, OOD targets, and input scaler |
| `checkpoints/` | Pretrained forward and diffusion models |
| `scripts/` | Shared benchmark runner |
| `examples/` | Benchmark command wrappers |

The pretrained checkpoints are `airfoil_sngp.pt`, `airfoil_cdm.pt`, `nacre_forward.h5`, and `nacre_cdm.pt`, all in `checkpoints/`.

### Data

Airfoil designs have 16 standardized PCA coordinates; nacre designs have 10 standardized material parameters. Each benchmark includes a `preprocessor.npz` file.

Nacre `dataset.npz` contains `X_train`, `X_val`, and `X_test` with shape `(N, 100, 11)` (10 design parameters followed by strain), and `y_train`, `y_val`, and `y_test` with shape `(N, 100)` in MPa. Its `targets.npz` stores response curves, strains, target IDs, tolerances, and response masks.

### Generated results

Demo runs save to `outputs/demo/<benchmark>/<method>/<target>/`; paper configurations save to `outputs/<benchmark>/<method>/<target>/`. Change `experiment.output_root` to choose another output directory.

| Output | Contents |
|---|---|
| `generation.npz` | Standardized `designs`, method-specific `scores`, and optional `predicted_mean` |
| `physical_designs.npz` | Nacre designs in physical units |
| `generation_manifest.json` | Method and score definition |
| `target.npz` | Target used for the run |
| `run_configuration.json` | Run settings |

## 📚 Citation

If you use this repository, please cite the [GUIDe paper](https://arxiv.org/abs/2509.05641):

```bibtex
@misc{mu2025guide,
  title = {{GUIDe}: Generative and Uncertainty-Informed Inverse Design for On-Demand Nonlinear Functional Responses},
  author = {Haoxuan Dylan Mu and Mingjian Tang and Wei Gao and Wei Wayne Chen},
  year = {2025},
  eprint = {2509.05641},
  archivePrefix = {arXiv},
  primaryClass = {cs.CE},
  url = {https://arxiv.org/abs/2509.05641}
}
```

Citation metadata is also available in [CITATION.cff](CITATION.cff).

## ⚖️ License

This project is licensed under the [MIT License](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party notices.
