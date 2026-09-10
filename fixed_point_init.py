"""Discover a fixed-point mode and use it to initialize a continuation branch.

This module contains the computation formerly performed interactively in
``fixed_point_init.ipynb``.  Every stage is callable independently for use in
the companion notebook, while :func:`main` runs and saves the complete
workflow without notebook state.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from branch import Branch_Propagation
from orbitlib import Auto_Integrator_Module, module
from orbitlib.theta_manipulate import frequency_extend
from run_train_job import SYSTEM, find_sol_model, process_stationary_starts


torch.set_default_dtype(torch.float64)


@dataclass
class FixedPointSweep:
    """Hessian scan and fixed-loop data used to select a branch seed."""

    initial_condition: tuple[float, float, float, float]
    periods: torch.Tensor
    minimum_eigenvalues: torch.Tensor
    full_minimum_eigenvalues: torch.Tensor | None
    theta0: torch.Tensor
    grad_active: list[list[bool]]
    stationary_starts: tuple[bool, bool]


@dataclass
class BranchSeedResult:
    """A continuation seed and diagnostics describing its construction."""

    branch: Branch_Propagation
    candidate_index: int
    candidate_period: float
    minimum_eigenvalue: float
    eigenvector_index: int
    step_size: float
    seed_loss: float
    closure_error: float
    closure_error_degrees: float


def create_fixed_point_loop(
    initial_condition: Sequence[float],
    frequency_cutoff: int,
    stationary_starts: tuple[bool, bool] = (True, True),
) -> tuple[torch.Tensor, list[list[bool]]]:
    """Represent a fixed state in radians as a Fourier loop with the requested symmetry."""
    if len(initial_condition) != SYSTEM.num_variables:
        raise ValueError(
            f'initial_condition must contain {SYSTEM.num_variables} values'
        )
    if frequency_cutoff < 1:
        raise ValueError('frequency_cutoff must be at least 1')

    theta0 = frequency_extend(
        torch.tensor(initial_condition, dtype=torch.get_default_dtype()).reshape((-1, 1)),
        frequency_cutoff,
    )
    return process_stationary_starts(theta0, stationary_starts)


def fixed_point_trainer(
    theta0: torch.Tensor,
    period: float,
    grad_active: list[list[bool]],
):
    """Construct the zero-epoch trainer used to evaluate a fixed-loop Hessian."""
    model = module.Odd_Grad_T_Module(
        theta0.clone().detach(),
        torch.tensor([period], dtype=theta0.dtype),
        grad_active,
    )
    _, trainer = find_sol_model(model, epoch_limit=0)
    return trainer


def sweep_period_hessian(
    initial_condition: Sequence[float],
    periods: Sequence[float] | np.ndarray | torch.Tensor,
    frequency_cutoff: int = 16,
    stationary_starts: tuple[bool, bool] = (True, True),
    compute_full_hessian: bool = True,
) -> FixedPointSweep:
    """Evaluate the lowest restricted and full Hessian eigenvalues over periods."""
    theta0, grad_active = create_fixed_point_loop(
        initial_condition, frequency_cutoff, stationary_starts
    )
    periods_tensor = torch.as_tensor(periods, dtype=theta0.dtype).reshape(-1)
    if periods_tensor.numel() == 0:
        raise ValueError('periods must not be empty')
    if not torch.all(periods_tensor > 0):
        raise ValueError('all periods must be positive')

    minimum_eigenvalues = []
    full_minimum_eigenvalues = []

    for period in periods_tensor:
        trainer = fixed_point_trainer(theta0, period.item(), grad_active)
        checkpoint = trainer.checkpoints[-1]
        minimum_eigenvalues.append(
            checkpoint.conditions_theta(full_theta=False, include_T=False)[0][0]
        )
        if compute_full_hessian:
            full_minimum_eigenvalues.append(
                checkpoint.conditions_theta(full_theta=True, include_T=False)[0][0]
            )

    return FixedPointSweep(
        initial_condition=tuple(float(value) for value in initial_condition),
        periods=periods_tensor.detach(),
        minimum_eigenvalues=torch.stack(minimum_eigenvalues).detach(),
        full_minimum_eigenvalues=(
            torch.stack(full_minimum_eigenvalues).detach()
            if compute_full_hessian
            else None
        ),
        theta0=theta0.detach(),
        grad_active=grad_active,
        stationary_starts=stationary_starts,
    )


def select_candidate(sweep: FixedPointSweep, candidate_index: int | None = None) -> int:
    """Select an explicit scan point, or the smallest restricted eigenvalue."""
    if candidate_index is None:
        return int(torch.argmin(sweep.minimum_eigenvalues).item())
    if not 0 <= candidate_index < sweep.periods.numel():
        raise IndexError(
            f'candidate_index {candidate_index} is outside '
            f'[0, {sweep.periods.numel() - 1}]'
        )
    return candidate_index


def plot_sweep(sweep: FixedPointSweep):
    """Plot positive eigenvalues on a log axis and report masked samples.

    Nonpositive and nonfinite values become gaps, never absolute values or
    artificial floors. The original tensors remain unchanged for selection.
    """
    import warnings
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    periods = sweep.periods.detach().cpu().numpy()
    series = [('symmetry restricted', sweep.minimum_eigenvalues)]
    if sweep.full_minimum_eigenvalues is not None:
        series.append(('full loop space', sweep.full_minimum_eigenvalues))
    notes = []
    has_positive = False
    for label, tensor in series:
        values = tensor.detach().cpu().numpy()
        valid = np.isfinite(values) & (values > 0)
        has_positive |= bool(valid.any())
        ax.plot(periods, np.where(valid, values, np.nan), label=label)
        if not valid.all():
            notes.append(f'{label}: {(~valid).sum()} masked')
            samples = list(zip(periods[~valid].tolist(), values[~valid].tolist()))
            warnings.warn(
                f'{label}: nonpositive/nonfinite eigenvalues hidden on log axis; '
                f'(period, eigenvalue) = {samples}. Inspect before selecting a seed.',
                RuntimeWarning, stacklevel=2,
            )
    if not has_positive:
        ax.set_ylim(1e-16, 1)
        notes.append('No positive finite eigenvalues to display')
    ax.set_yscale('log')
    ax.set_xlabel('Period T')
    ax.set_ylabel('Minimum Hessian eigenvalue')
    if notes:
        ax.text(0.02, 0.02, '\n'.join(notes), transform=ax.transAxes,
                fontsize='small', va='bottom')
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_seed_trajectory(seed: BranchSeedResult, num_samples: int = 400):
    """Compare the Fourier predictor and its IVP trajectory over one period.

    The theta1–theta2 projection is displayed in radians with start/end markers.
    Its apparent closure is not a substitute for the four-state closure error.
    """
    import matplotlib.pyplot as plt

    if num_samples < 2:
        raise ValueError('num_samples must be at least 2')
    branch = seed.branch
    period = float(branch.T0.item())
    z0 = SYSTEM.z0(branch.theta0).detach().cpu().numpy().reshape(-1)
    integration = Auto_Integrator_Module.solve_ivp(SYSTEM, z0, (0, period))
    if not integration.success:
        raise RuntimeError(f'seed integration failed: {integration.message}')
    times = np.linspace(0, period, num_samples)
    trajectory = integration.sol(times)
    loop = SYSTEM.z(branch.theta0, torch.as_tensor(times, dtype=branch.theta0.dtype),
                    branch.T0).detach().cpu().numpy()
    closure = float(np.linalg.norm(trajectory[:, -1] - z0))
    fig, ax = plt.subplots()
    ax.plot(loop[0], loop[1], '--', label='Fourier seed')
    ax.plot(trajectory[0], trajectory[1], label='Integrated trajectory')
    ax.scatter(z0[0], z0[1], marker='o', facecolors='none', edgecolors='black',
               label='Initial state', zorder=3)
    ax.scatter(trajectory[0, -1], trajectory[1, -1], marker='x', color='red',
               label='Integrated endpoint', zorder=4)
    ax.set_xlabel(r'$\theta_1$ (rad)')
    ax.set_ylabel(r'$\theta_2$ (rad)')
    ax.set_title(f'Seed predictor: T = {period:.6g}\nFull-state closure error = {closure:.3e}')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend()
    fig.tight_layout()
    return fig, ax


def seed_branch(
    sweep: FixedPointSweep,
    results_folder: str | os.PathLike[str],
    candidate_index: int | None = None,
    eigenvector_index: int = 0,
    step_angle_degrees: float = 0.5,
    orientation_component: int = 0,
    orientation_sign: int = 1,
    branch_module=module.Const_Odd_Grad_T_Module,
    wparam: bool = True,
) -> BranchSeedResult:
    """Perturb a fixed loop along a Hessian mode and construct branch state."""
    selected_index = select_candidate(sweep, candidate_index)
    if orientation_component not in range(SYSTEM.num_variables):
        raise ValueError('orientation_component must be between 0 and 3')
    if orientation_sign not in (-1, 1):
        raise ValueError('orientation_sign must be -1 or 1')
    if step_angle_degrees <= 0:
        raise ValueError('step_angle_degrees must be positive')

    period = float(sweep.periods[selected_index].item())
    trainer = fixed_point_trainer(sweep.theta0, period, sweep.grad_active)
    checkpoint = trainer.checkpoints[-1]
    eigenvalues, conditions = checkpoint.conditions_theta(
        full_theta=False, include_T=False
    )
    if not 0 <= eigenvector_index < len(conditions):
        raise IndexError('eigenvector_index is outside the Hessian eigensystem')

    d_theta = conditions[eigenvector_index][0]
    d_z0 = trainer.system.z0(d_theta)
    if d_z0[orientation_component, 0] * orientation_sign < 0:
        d_theta = -d_theta
        d_z0 = -d_z0

    d_z0_norm = torch.linalg.norm(d_z0)
    if not torch.isfinite(d_z0_norm) or d_z0_norm.item() == 0:
        raise ValueError('selected Hessian mode has an invalid initial-state direction')

    step_angle = step_angle_degrees * np.pi / 180
    step_size = step_angle / d_z0_norm
    theta0_perturbed, _ = process_stationary_starts(
        sweep.theta0 + step_size * d_theta,
        sweep.stationary_starts,
    )
    T0 = torch.tensor((period,), dtype=sweep.theta0.dtype)

    branch = Branch_Propagation(
        branch_module,
        theta0_perturbed.detach(),
        d_z0.detach(),
        T0,
        sweep.stationary_starts,
        str(results_folder),
        step_angle=step_angle,
        wparam=wparam,
    )

    diagnostics = validate_seed(branch)
    return BranchSeedResult(
        branch=branch,
        candidate_index=selected_index,
        candidate_period=period,
        minimum_eigenvalue=float(eigenvalues[eigenvector_index].item()),
        eigenvector_index=eigenvector_index,
        step_size=float(step_size.item()),
        seed_loss=diagnostics['seed_loss'],
        closure_error=diagnostics['closure_error'],
        closure_error_degrees=diagnostics['closure_error_degrees'],
    )


def validate_seed(branch: Branch_Propagation) -> dict[str, float]:
    """Measure variational residual and one-period IVP closure of a branch seed."""
    z0 = SYSTEM.z0(branch.theta0)
    period = float(branch.T0.item())
    integration = Auto_Integrator_Module.solve_ivp(
        SYSTEM,
        z0.detach().numpy().reshape(-1),
        (0, period),
    )
    if not integration.success:
        raise RuntimeError(f'seed validation integration failed: {integration.message}')

    closure_error = float(
        np.linalg.norm(integration.sol(period) - z0.detach().numpy().reshape(-1))
    )
    return {
        'seed_loss': float(SYSTEM.loss(branch.theta0, branch.T0).item()),
        'closure_error': closure_error,
        'closure_error_degrees': closure_error * 180 / np.pi,
    }


def save_sweep(
    sweep: FixedPointSweep,
    output_directory: str | os.PathLike[str],
    case_name: str,
) -> dict[str, Path]:
    """Save scan arrays and self-describing metadata."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    restricted_path = output_directory / f'{case_name}.pt'
    full_path = output_directory / f'{case_name}-full.pt'
    metadata_path = output_directory / f'{case_name}.json'

    torch.save(sweep.minimum_eigenvalues, restricted_path)
    if sweep.full_minimum_eigenvalues is not None:
        torch.save(sweep.full_minimum_eigenvalues, full_path)

    metadata = {
        'case_name': case_name,
        'initial_condition': list(sweep.initial_condition),
        'periods': sweep.periods.tolist(),
        'frequency_cutoff': (sweep.theta0.shape[1] - 1) // 2,
        'stationary_starts': list(sweep.stationary_starts),
        'restricted_eigenvalues_file': restricted_path.name,
        'full_eigenvalues_file': (
            full_path.name if sweep.full_minimum_eigenvalues is not None else None
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    return {
        'restricted': restricted_path,
        'full': full_path if sweep.full_minimum_eigenvalues is not None else None,
        'metadata': metadata_path,
    }


def save_branch_seed(result: BranchSeedResult, overwrite: bool = False) -> Path:
    """Save continuation state and metadata, refusing accidental replacement."""
    results_folder = Path(result.branch.results_folder)
    propagation_path = results_folder / 'propagation.pt'
    if propagation_path.exists() and not overwrite:
        raise FileExistsError(
            f'{propagation_path} already exists; pass overwrite=True to replace it'
        )
    results_folder.mkdir(parents=True, exist_ok=True)
    result.branch.save_self()

    metadata = {
        'candidate_index': result.candidate_index,
        'candidate_period': result.candidate_period,
        'minimum_eigenvalue': result.minimum_eigenvalue,
        'eigenvector_index': result.eigenvector_index,
        'step_size': result.step_size,
        'step_angle_degrees': result.branch.step_angle * 180 / np.pi,
        'seed_loss': result.seed_loss,
        'closure_error': result.closure_error,
        'closure_error_degrees': result.closure_error_degrees,
        'stationary_starts': list(result.branch.stationary_starts),
        'frequency_cutoff': result.branch.frequency_cutoff,
        'module': result.branch.module.__name__,
    }
    (results_folder / 'seed-metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n'
    )
    return propagation_path


def run_workflow(
    *,
    initial_condition: Sequence[float],
    periods: Sequence[float] | np.ndarray | torch.Tensor,
    frequency_cutoff: int,
    stationary_starts: tuple[bool, bool],
    compute_full_hessian: bool,
    sweep_output: str | os.PathLike[str],
    case_name: str,
    branch_output: str | os.PathLike[str],
    candidate_index: int | None,
    eigenvector_index: int,
    step_angle_degrees: float,
    orientation_component: int,
    orientation_sign: int,
    overwrite: bool = False,
) -> tuple[FixedPointSweep, BranchSeedResult]:
    """Run the complete fixed-point scan, selection, seed, validation, and save."""
    sweep = sweep_period_hessian(
        initial_condition,
        periods,
        frequency_cutoff,
        stationary_starts,
        compute_full_hessian,
    )
    save_sweep(sweep, sweep_output, case_name)
    seed = seed_branch(
        sweep,
        branch_output,
        candidate_index,
        eigenvector_index,
        step_angle_degrees,
        orientation_component,
        orientation_sign,
    )
    save_branch_seed(seed, overwrite=overwrite)
    return sweep, seed


def stationary_starts_from_name(name: str) -> tuple[bool, bool]:
    """Convert the CLI symmetry name to the pair used by the research code."""
    return {
        'both': (True, True),
        'upper': (True, False),
        'lower': (False, True),
        'none': (False, False),
    }[name]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Sweep a fixed-point Hessian and initialize a continuation branch.'
    )
    parser.add_argument(
        '--initial-condition',
        type=float,
        nargs=4,
        default=(0, 0, 0, 0),
        help='Fixed phase-space state in radians and radians per unit time.',
    )
    parser.add_argument('--period-min', type=float, default=0.01)
    parser.add_argument('--period-max', type=float, default=10.0)
    parser.add_argument('--period-count', type=int, default=1000)
    parser.add_argument('--frequency-cutoff', type=int, default=16)
    parser.add_argument(
        '--stationary-starts',
        choices=('both', 'upper', 'lower', 'none'),
        default='both',
    )
    parser.add_argument('--skip-full-hessian', action='store_true')
    parser.add_argument('--sweep-output', default='fixed_point_evalues')
    parser.add_argument('--case-name', default='down_down')
    parser.add_argument('--branch-output', required=True)
    parser.add_argument('--candidate-index', type=int)
    parser.add_argument('--eigenvector-index', type=int, default=0)
    parser.add_argument('--step-angle-degrees', type=float, default=0.5)
    parser.add_argument('--orientation-component', type=int, default=0)
    parser.add_argument('--orientation-sign', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main(argv=None):
    """Run the complete fixed-point initialization workflow from the command line."""
    args = build_parser().parse_args(argv)
    if args.period_count < 1:
        raise ValueError('period_count must be at least 1')
    if args.period_min <= 0 or args.period_max <= 0:
        raise ValueError('period bounds must be positive')
    if args.period_max < args.period_min:
        raise ValueError('period_max must not be smaller than period_min')

    periods = np.linspace(args.period_min, args.period_max, args.period_count)
    _, seed = run_workflow(
        initial_condition=args.initial_condition,
        periods=periods,
        frequency_cutoff=args.frequency_cutoff,
        stationary_starts=stationary_starts_from_name(args.stationary_starts),
        compute_full_hessian=not args.skip_full_hessian,
        sweep_output=args.sweep_output,
        case_name=args.case_name,
        branch_output=args.branch_output,
        candidate_index=args.candidate_index,
        eigenvector_index=args.eigenvector_index,
        step_angle_degrees=args.step_angle_degrees,
        orientation_component=args.orientation_component,
        orientation_sign=args.orientation_sign,
        overwrite=args.overwrite,
    )
    summary = {
        'candidate_index': seed.candidate_index,
        'candidate_period': seed.candidate_period,
        'minimum_eigenvalue': seed.minimum_eigenvalue,
        'seed_loss': seed.seed_loss,
        'closure_error_degrees': seed.closure_error_degrees,
        'branch_file': str(Path(seed.branch.results_folder) / 'propagation.pt'),
    }
    print(json.dumps(summary, indent=2))
    return seed


if __name__ == '__main__':
    main()
