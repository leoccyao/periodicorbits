# Variational Continuation for Double Pendulum Periodic Orbits

This repository contains the numerical implementation accompanying the paper *Variational Continuation for Double Pendulum Periodic Orbits* by Leo Yao, Ziming Liu, and Max Tegmark. A candidate periodic orbit is represented as a truncated Fourier loop, and a loss function measures its deviation from the physical differential equations. Automatic differentiation provides the Hessian of this loss; its flat directions, identified by zero eigenvalues, guide fixed-point initialization and continuation through families of periodic solutions. Hessian scans of original and repeated loops locate orbit-family intersections and subharmonic bifurcations.

The variational loss and Hessian continuation direction do not require integrating an initial condition forward in time. Direct numerical integration is used separately to initialize isolated recurrences and to measure one-period closure error for validation and stopping diagnostics.

The scripts support end-to-end runs, while the notebooks use the same functions for step-by-step execution, visual inspection, and plotting.

## Project resources

- Paper: Leo Yao, Ziming Liu, and Max Tegmark, *Variational Continuation for Double Pendulum Periodic Orbits*. [arXiv:2609.05337](https://arxiv.org/abs/2609.05337)
- Interactive visualization: [Explore the periodic-orbit families](https://periodicorbits.pages.dev/)
- Poster: Leo Yao, Ziming Liu, and Max Tegmark, *Variational Loss Landscapes for Periodic Orbits*, NeurIPS 2024 Workshop: Machine Learning and the Physical Sciences. [View poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202024/100057.png)

## Installation

The pinned environment contains the direct numerical dependencies and the software needed to run the included notebooks:

```bash
conda env create -f environment.yml
conda activate periodicorbits
```

Run commands from the repository root so that the local `orbitlib` package and double-pendulum modules are importable.

The calculations use double-precision PyTorch tensors. The default dimensionless system parameters are

```text
m1 = m2 = l1 = l2 = g = 1
```

## Main workflow

The complete branch and bifurcation workflow has three stages:

1. Scan the Hessian at a fixed point and create a branch predictor with `fixed_point_init.py`.
2. Correct and continue that predictor with `branch_propagation.py`.
3. Scan Hessian eigenvalues along the saved branch and optionally seed a new branch with `curve_eigenvalues.py`.

Every script supports `--help` for its complete set of controls.

### 1. Initialize a branch at a fixed point

```bash
python fixed_point_init.py \
  --initial-condition 0 0 0 0 \
  --period-min 0.01 \
  --period-max 10 \
  --period-count 1000 \
  --frequency-cutoff 16 \
  --sweep-output results/fixed-point-eigenvalues \
  --case-name down_down \
  --branch-output results/down_down
```

The initial condition is the phase-space state

```text
(theta1, theta2, theta1_dot, theta2_dot)
```

in radians and radians per unit time. The script evaluates the lowest Hessian eigenvalue across the requested period range, selects its minimum unless `--candidate-index` is supplied, perturbs the fixed loop along the selected eigenvector, validates the seed by direct integration, and writes:

- restricted and full-space eigenvalue arrays plus JSON metadata under `--sweep-output`; and
- `propagation.pt` and `seed-metadata.json` under `--branch-output`.

Use `--eigenvector-index`, `--orientation-component`, and `--orientation-sign` to make mode selection and orientation explicit. The default `--step-angle-degrees 0.5` sets the initial phase-space displacement scale.

The full default sweep performs many Hessian decompositions and can take substantial time. A shorter scan can be used to check an installation, but the period resolution is part of a scientific run and should be recorded.

The companion [`fixed_point_init.ipynb`](fixed_point_init.ipynb) exposes the scan, candidate selection, seed construction, validation, and eigenvalue plots as separate cells.

### 2. Propagate the periodic-orbit branch

Inspect the saved continuation state without changing it:

```bash
python branch_propagation.py results/down_down --inspect-only
```

Run ten continuation attempts:

```bash
python branch_propagation.py results/down_down \
  --iterations 10 \
  --print-validation 200
```

Each attempt optimizes the predicted Fourier loop subject to a continuation constraint. An attempt may either accept a new branch point or increase the epoch budget or Fourier cutoff for a later retry. The JSON printed by the command reports `accepted_point` explicitly.

The branch directory contains the resumable `propagation.pt` state. Each accepted point is written to a numbered directory containing:

```text
var-model.pt
var-results.json
```

The point directory name also records the final phase-space initial condition in degrees. Running the command again resumes from the saved predictor. Relevant numerical controls include `--learning-rate`, `--val-epochs`, `--epoch-limit`, `--frequency-double-cutoff`, and `--epoch-double-cutoff`. The `--extrapolate` option enables blockwise convergence extrapolation.

The companion [`branch_propagation.ipynb`](branch_propagation.ipynb) provides state inspection, individual propagation calls, accepted-attempt summaries, and training/validation-loss plots.

### 3. Scan eigenvalues and seed a bifurcation

Compute the lowest eight full-coordinate Hessian eigenvalues for the original and period-doubled loops at every accepted branch point:

```bash
python curve_eigenvalues.py results/down_down \
  --multipliers 1 2 \
  --num-eigenvalues 8 \
  --scan-output eigenvalue-scan.pt
```

This writes the aggregate tensor file and a JSON metadata file in the branch directory. By default, per-point eigenpairs are cached as `conditions-1.pt` and `conditions-2.pt` next to each saved trainer. `--no-use-existing-sidecars` forces recomputation, and `--no-write-sidecars` disables those caches.

After inspecting the spectrum and selecting a source point and eigenvector, the same command can create a predictor for a bifurcating family:

```bash
python curve_eigenvalues.py results/down_down \
  --multipliers 1 2 \
  --scan-output eigenvalue-scan.pt \
  --seed-output results/period_doubled \
  --source-point-index 0 \
  --seed-multiplier 2 \
  --seed-eigenvector-index 2
```

The source point in this example is illustrative. A scientific run should choose `--source-point-index` and `--seed-eigenvector-index` from the inspected eigenspectrum and record that choice. The new output directory contains a `propagation.pt` that can be passed directly to `branch_propagation.py`.

The companion [`curve_eigenvalues.ipynb`](curve_eigenvalues.ipynb) plots a selected eigenvalue against branch index, energy, period, or a phase-space coordinate and supports interactive seed selection.

## Independent single-orbit optimization

`run_train_job.py` initializes one Fourier loop from a numerically integrated trajectory and optimizes it as a periodic orbit. Unlike the fixed-point interface, its initial angles are specified in degrees, and the angular velocities default to zero.

```bash
python run_train_job.py results/single_orbits 20 20 \
  --frequency-cutoff 64 \
  --stationary-starts both
```

The result is stored below `results/single_orbits` in a directory named by the supplied initial angles. Use this entry point when optimizing an isolated recurrence rather than constructing a Hessian-guided family.

## Numerical representation and diagnostics

For a state with four variables, the Fourier coefficient tensor has shape

```text
(4, 2 * frequency_cutoff + 1)
```

Each row contains a constant coefficient, cosine coefficients, and sine coefficients. The train loss is the period-normalized integral of the squared residual between the Fourier-loop derivative and the double-pendulum vector field. The validation loss is computed independently by integrating the initial state for one period and measuring its failure to close.

The Hessian is taken with respect to flattened Fourier coefficients and, when requested, the period. Low eigenvalues identify locally flat directions used for fixed-point initialization, branch continuation, and bifurcation analysis. `frequency_cutoff_mode=base`, the command-line default, evaluates repeated loops at the source cutoff; `scaled` multiplies the cutoff with the period multiplier to retain the source spectral resolution.

Saved `.pt` files use PyTorch serialization and may contain Python objects. Load only files from trusted sources.

## AI Usage

AI tools (Claude Code and OpenAI Codex) assisted with rewriting research notebooks into runnable scripts and companion notebooks. The migrated workflows were checked through automated tests and visual comparison with the original notebooks. The numerical core (orbitlib, doublepend.py, analysis.py, branch.py, and curve_prop.py) retains the original research implementation, aside from minor cleanup (comments, documentation, deprecated methods).

## Citation

```bibtex
@article{yao_variational_continuation,
  title   = {Variational Continuation for Double Pendulum Periodic Orbits},
  author  = {Yao, Leo and Liu, Ziming and Tegmark, Max},
  year    = {2026},
  eprint  = {2609.05337},
  archivePrefix = {arXiv}
}
```
