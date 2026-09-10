"""Scan branch Hessian eigenvalues and optionally seed a bifurcating branch.

This is the authoritative computational implementation of the workflow that
was formerly contained in ``curve_eigenvalues.ipynb``.  It consumes accepted
point directories produced by ``branch_propagation.py``.  The companion
notebook imports these functions to inspect and plot intermediate results.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from analysis import get_match_dir_names, load_trainer
from branch import Branch_Propagation
from fixed_point_init import validate_seed
from orbitlib import Trainer, frequency_extend, module, period_multiply
from run_train_job import SYSTEM, process_stationary_starts


torch.set_default_dtype(torch.float64)


@dataclass
class BranchEigenvalueScan:
    """Ordered Hessian spectra and branch coordinates for accepted points."""

    data_directory: str
    matches: list[str]
    multipliers: tuple[int, ...]
    num_eigenvalues: int
    frequency_cutoff_mode: str
    eigenvalues: dict[int, torch.Tensor]
    energy: torch.Tensor
    period: torch.Tensor
    initial_conditions_degrees: torch.Tensor


@dataclass
class BifurcationSeedResult:
    """A new branch predictor and the choices used to construct it."""

    branch: Branch_Propagation
    source_directory: str
    source_match: str
    source_point_index: int
    period_multiplier: int
    eigenvector_index: int
    eigenvalue: float
    first_step_multiplier: float
    step_size: float
    orientation_component: int
    orientation_sign: int
    seed_loss: float
    closure_error: float
    closure_error_degrees: float


def conditions_sidecar_path(
    data_directory: str | Path,
    match: str,
    multiplier: int,
    frequency_cutoff_mode: str = 'base',
) -> Path:
    suffix = '' if frequency_cutoff_mode == 'base' else f'-{frequency_cutoff_mode}'
    return Path(data_directory) / match / f'conditions-{multiplier}{suffix}.pt'


def hessian_frequency_cutoff(
    trainer: Trainer, multiplier: int, mode: str
) -> int:
    """Choose the Hessian representation size.

    ``base`` matches the sidecar workflow used for the paper: a multiplied loop
    is represented at the source model's cutoff.  ``scaled`` retains the source
    spectral resolution by multiplying the cutoff with the period.
    """
    base_cutoff = trainer.model.frequency_cutoff
    if mode == 'base':
        return base_cutoff
    if mode == 'scaled':
        return base_cutoff * multiplier
    raise ValueError("frequency_cutoff_mode must be 'base' or 'scaled'")


def compute_conditions(
    trainer: Trainer,
    multiplier: int,
    num_eigenvalues: int = 8,
    frequency_cutoff_mode: str = 'base',
):
    """Compute full-loop Hessian eigenpairs without retaining a checkpoint cache."""
    if multiplier < 1:
        raise ValueError('multiplier must be at least 1')
    if num_eigenvalues < 1:
        raise ValueError('num_eigenvalues must be at least 1')
    cutoff = hessian_frequency_cutoff(
        trainer, multiplier, frequency_cutoff_mode
    )
    return trainer.checkpoints[-1].conditions_theta(
        full_theta=True,
        frequency_cutoff=cutoff,
        mult_periods=multiplier,
        num_eig=num_eigenvalues,
        _cache_controls=(False, True, False),
    )


def load_or_compute_conditions(
    data_directory: str | Path,
    match: str,
    trainer: Trainer,
    multiplier: int,
    *,
    num_eigenvalues: int = 8,
    frequency_cutoff_mode: str = 'base',
    use_existing_sidecars: bool = True,
    write_sidecars: bool = True,
):
    """Load compatible sidecar eigenpairs or compute them from a saved trainer."""
    sidecar = conditions_sidecar_path(
        data_directory, match, multiplier, frequency_cutoff_mode
    )
    if use_existing_sidecars and sidecar.is_file():
        conditions = torch.load(sidecar, map_location='cpu')
        if len(conditions[0]) >= num_eigenvalues:
            return conditions[0][:num_eigenvalues], conditions[1][:num_eigenvalues]

    conditions = compute_conditions(
        trainer,
        multiplier,
        num_eigenvalues,
        frequency_cutoff_mode,
    )
    if write_sidecars:
        torch.save(conditions, sidecar)
    return conditions


def scan_branch_eigenvalues(
    data_directory: str | Path,
    *,
    multipliers: Sequence[int] = (1, 2),
    num_eigenvalues: int = 8,
    limit: int | None = None,
    frequency_cutoff_mode: str = 'base',
    use_existing_sidecars: bool = True,
    write_sidecars: bool = True,
) -> BranchEigenvalueScan:
    """Compute ordered Hessian spectra and coordinates along a saved branch."""
    data_directory = Path(data_directory)
    if not data_directory.is_dir():
        raise FileNotFoundError(f'branch directory does not exist: {data_directory}')
    multipliers = tuple(int(value) for value in multipliers)
    if not multipliers or any(value < 1 for value in multipliers):
        raise ValueError('multipliers must contain positive integers')
    if len(set(multipliers)) != len(multipliers):
        raise ValueError('multipliers must not contain duplicates')

    matches = get_match_dir_names(data_directory)
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be at least 1')
        matches = matches[:limit]
    if not matches:
        raise ValueError(
            f'no accepted branch-point directories were found in {data_directory}'
        )

    eigenvalues = {multiplier: [] for multiplier in multipliers}
    energies = []
    periods = []
    initial_conditions = []

    for match in matches:
        trainer = load_trainer(data_directory, match)
        checkpoint = trainer.checkpoints[-1]
        for multiplier in multipliers:
            values, _ = load_or_compute_conditions(
                data_directory,
                match,
                trainer,
                multiplier,
                num_eigenvalues=num_eigenvalues,
                frequency_cutoff_mode=frequency_cutoff_mode,
                use_existing_sidecars=use_existing_sidecars,
                write_sidecars=write_sidecars,
            )
            eigenvalues[multiplier].append(values.detach().cpu())

        initial_condition = checkpoint.initial_condition.detach().cpu().reshape(-1)
        energies.append(SYSTEM.energy(initial_condition.reshape((-1, 1))).item())
        periods.append(checkpoint.T.item())
        initial_conditions.append(initial_condition * 180 / torch.pi)

    return BranchEigenvalueScan(
        data_directory=str(data_directory),
        matches=matches,
        multipliers=multipliers,
        num_eigenvalues=num_eigenvalues,
        frequency_cutoff_mode=frequency_cutoff_mode,
        eigenvalues={
            multiplier: torch.stack(values)
            for multiplier, values in eigenvalues.items()
        },
        energy=torch.tensor(energies),
        period=torch.tensor(periods),
        initial_conditions_degrees=torch.stack(initial_conditions),
    )


def scan_coordinate(scan: BranchEigenvalueScan, name: str) -> torch.Tensor:
    """Return a named horizontal coordinate for diagnostics and paper plots."""
    coordinates = {
        'index': torch.arange(len(scan.matches), dtype=torch.get_default_dtype()),
        'energy': scan.energy,
        'period': scan.period,
        'theta1': scan.initial_conditions_degrees[:, 0],
        'theta2': scan.initial_conditions_degrees[:, 1],
        'theta1dot': scan.initial_conditions_degrees[:, 2],
        'theta2dot': scan.initial_conditions_degrees[:, 3],
    }
    try:
        return coordinates[name]
    except KeyError as error:
        raise ValueError(
            f'unknown coordinate {name!r}; choose from {tuple(coordinates)}'
        ) from error


def target_eigenvalues(
    scan: BranchEigenvalueScan,
    multiplier: int,
    eigenvalue_index: int = 2,
) -> torch.Tensor:
    """Select one ordered Hessian eigenvalue along the branch."""
    if multiplier not in scan.eigenvalues:
        raise KeyError(f'multiplier {multiplier} is not present in this scan')
    values = scan.eigenvalues[multiplier]
    if not 0 <= eigenvalue_index < values.shape[1]:
        raise IndexError('eigenvalue_index is outside the saved spectrum')
    return values[:, eigenvalue_index]


def minimum_in_range(
    scan: BranchEigenvalueScan,
    *,
    multiplier: int,
    eigenvalue_index: int = 2,
    start: int = 0,
    stop: int | None = None,
) -> int:
    """Return the branch index of the lowest target eigenvalue in an index range."""
    values = target_eigenvalues(scan, multiplier, eigenvalue_index)
    if stop is None:
        stop = len(values)
    if not 0 <= start < stop <= len(values):
        raise ValueError('candidate range must satisfy 0 <= start < stop <= length')
    return start + int(torch.argmin(values[start:stop]).item())


def save_scan(scan: BranchEigenvalueScan, output: str | Path) -> Path:
    """Save aggregate scan tensors plus readable provenance metadata."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'matches': scan.matches,
            'multipliers': scan.multipliers,
            'num_eigenvalues': scan.num_eigenvalues,
            'frequency_cutoff_mode': scan.frequency_cutoff_mode,
            'eigenvalues': scan.eigenvalues,
            'energy': scan.energy,
            'period': scan.period,
            'initial_conditions_degrees': scan.initial_conditions_degrees,
        },
        output,
    )
    metadata = {
        'data_directory': scan.data_directory,
        'point_count': len(scan.matches),
        'matches': scan.matches,
        'multipliers': list(scan.multipliers),
        'num_eigenvalues': scan.num_eigenvalues,
        'frequency_cutoff_mode': scan.frequency_cutoff_mode,
        'tensor_file': output.name,
    }
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return output


def load_scan(output: str | Path) -> BranchEigenvalueScan:
    """Load an aggregate scan generated by :func:`save_scan`."""
    output = Path(output)
    values = torch.load(output, map_location='cpu')
    metadata_path = output.with_suffix('.json')
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    return BranchEigenvalueScan(
        data_directory=metadata.get('data_directory', ''),
        matches=list(values['matches']),
        multipliers=tuple(values['multipliers']),
        num_eigenvalues=int(values['num_eigenvalues']),
        frequency_cutoff_mode=values['frequency_cutoff_mode'],
        eigenvalues=values['eigenvalues'],
        energy=values['energy'],
        period=values['period'],
        initial_conditions_degrees=values['initial_conditions_degrees'],
    )


def seed_bifurcation(
    scan: BranchEigenvalueScan,
    results_folder: str | Path,
    *,
    source_point_index: int,
    period_multiplier: int,
    eigenvector_index: int,
    step_angle_degrees: float = 0.5,
    first_step_multiplier: float = 10.0,
    orientation_component: int = 2,
    orientation_sign: int = 1,
    stationary_starts: tuple[bool, bool] = (False, False),
    frequency_extension_factor: int = 1,
    frequency_double_cutoff: float = 1e-20,
    epoch_double_cutoff: float = 1e-18,
    val_epochs: int = 2000,
    epoch_limit: int = 100000,
    use_existing_sidecars: bool = True,
) -> BifurcationSeedResult:
    """Perturb a multiplied source loop along a selected full-Hessian mode."""
    if not 0 <= source_point_index < len(scan.matches):
        raise IndexError('source_point_index is outside the scan')
    if period_multiplier not in scan.multipliers:
        raise ValueError('period_multiplier was not included in the scan')
    if orientation_component not in range(SYSTEM.num_variables):
        raise ValueError('orientation_component must be between 0 and 3')
    if orientation_sign not in (-1, 1):
        raise ValueError('orientation_sign must be -1 or 1')
    if step_angle_degrees <= 0 or first_step_multiplier <= 0:
        raise ValueError('step angles and multipliers must be positive')
    if frequency_extension_factor < 1:
        raise ValueError('frequency_extension_factor must be at least 1')

    match = scan.matches[source_point_index]
    trainer = load_trainer(scan.data_directory, match)
    checkpoint = trainer.checkpoints[-1]
    eigenvalues, eigenvectors = load_or_compute_conditions(
        scan.data_directory,
        match,
        trainer,
        period_multiplier,
        num_eigenvalues=scan.num_eigenvalues,
        frequency_cutoff_mode=scan.frequency_cutoff_mode,
        use_existing_sidecars=use_existing_sidecars,
        write_sidecars=True,
    )
    if not 0 <= eigenvector_index < len(eigenvectors):
        raise IndexError('eigenvector_index is outside the saved eigensystem')

    theta0, T0 = period_multiply(
        checkpoint.theta,
        checkpoint.T.item(),
        period_multiplier,
        truncate=True,
    )
    d_theta, d_T = eigenvectors[eigenvector_index]
    d_z0 = trainer.system.z0(d_theta)
    if d_z0[orientation_component, 0] * orientation_sign < 0:
        d_theta = -d_theta
        d_z0 = -d_z0
        d_T = -d_T

    d_z0_norm = torch.linalg.norm(d_z0)
    if not torch.isfinite(d_z0_norm) or d_z0_norm.item() == 0:
        raise ValueError('selected eigenvector has an invalid initial-state direction')
    step_angle = step_angle_degrees * math.pi / 180
    step_size = step_angle * first_step_multiplier / d_z0_norm
    theta0_perturbed, _ = process_stationary_starts(
        theta0 + step_size * d_theta,
        stationary_starts,
    )
    T0_perturbed = T0 + step_size * d_T

    branch = Branch_Propagation(
        module.Theta_Grad_T_Module,
        theta0_perturbed.detach(),
        d_z0.detach(),
        torch.as_tensor(T0_perturbed).detach().reshape(1),
        stationary_starts,
        str(results_folder),
        step_angle=step_angle,
        wparam=True,
        frequency_double_cutoff=frequency_double_cutoff,
        epoch_double_cutoff=epoch_double_cutoff,
        val_epochs=val_epochs,
        epoch_limit=epoch_limit,
    )
    if frequency_extension_factor > 1:
        branch.theta0 = frequency_extend(
            branch.theta0,
            branch.frequency_cutoff * frequency_extension_factor,
        )
        branch.frequency_cutoff *= frequency_extension_factor
        branch.integral_num_samples *= frequency_extension_factor

    diagnostics = validate_seed(branch)
    return BifurcationSeedResult(
        branch=branch,
        source_directory=scan.data_directory,
        source_match=match,
        source_point_index=source_point_index,
        period_multiplier=period_multiplier,
        eigenvector_index=eigenvector_index,
        eigenvalue=float(eigenvalues[eigenvector_index].item()),
        first_step_multiplier=first_step_multiplier,
        step_size=float(step_size.item()),
        orientation_component=orientation_component,
        orientation_sign=orientation_sign,
        seed_loss=diagnostics['seed_loss'],
        closure_error=diagnostics['closure_error'],
        closure_error_degrees=diagnostics['closure_error_degrees'],
    )


def save_bifurcation_seed(
    result: BifurcationSeedResult, *, overwrite: bool = False
) -> Path:
    """Save bifurcation predictor state with its complete selection provenance."""
    results_folder = Path(result.branch.results_folder)
    propagation = results_folder / 'propagation.pt'
    if propagation.exists() and not overwrite:
        raise FileExistsError(
            f'{propagation} already exists; pass overwrite=True to replace it'
        )
    results_folder.mkdir(parents=True, exist_ok=True)
    result.branch.save_self()
    metadata = {
        'source_directory': result.source_directory,
        'source_match': result.source_match,
        'source_point_index': result.source_point_index,
        'period_multiplier': result.period_multiplier,
        'eigenvector_index': result.eigenvector_index,
        'eigenvalue': result.eigenvalue,
        'first_step_multiplier': result.first_step_multiplier,
        'step_size': result.step_size,
        'step_angle_degrees': result.branch.step_angle * 180 / math.pi,
        'orientation_component': result.orientation_component,
        'orientation_sign': result.orientation_sign,
        'stationary_starts': list(result.branch.stationary_starts),
        'frequency_cutoff': result.branch.frequency_cutoff,
        'frequency_double_cutoff': result.branch.frequency_double_cutoff,
        'epoch_double_cutoff': result.branch.epoch_double_cutoff,
        'val_epochs': result.branch.val_epochs,
        'epoch_limit': result.branch.epoch_limit,
        'seed_loss': result.seed_loss,
        'closure_error': result.closure_error,
        'closure_error_degrees': result.closure_error_degrees,
    }
    (results_folder / 'seed-metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n'
    )
    return propagation


def stationary_starts_from_name(name: str) -> tuple[bool, bool]:
    return {
        'both': (True, True),
        'upper': (True, False),
        'lower': (False, True),
        'none': (False, False),
    }[name]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Scan branch Hessians and optionally seed a new branch.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('data_directory')
    parser.add_argument('--multipliers', type=int, nargs='+', default=(1, 2))
    parser.add_argument('--num-eigenvalues', type=int, default=8)
    parser.add_argument('--limit', type=int)
    parser.add_argument(
        '--frequency-cutoff-mode', choices=('base', 'scaled'), default='base'
    )
    parser.add_argument(
        '--use-existing-sidecars',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--write-sidecars', action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument('--scan-output', default='eigenvalue-scan.pt')

    seed = parser.add_argument_group('optional bifurcation seed')
    seed.add_argument('--seed-output')
    seed.add_argument('--source-point-index', type=int)
    seed.add_argument('--seed-multiplier', type=int, default=2)
    seed.add_argument('--seed-eigenvector-index', type=int, default=2)
    seed.add_argument('--step-angle-degrees', type=float, default=0.5)
    seed.add_argument('--first-step-multiplier', type=float, default=10.0)
    seed.add_argument('--orientation-component', type=int, default=2)
    seed.add_argument('--orientation-sign', type=int, choices=(-1, 1), default=1)
    seed.add_argument(
        '--stationary-starts',
        choices=('both', 'upper', 'lower', 'none'),
        default='none',
    )
    seed.add_argument('--frequency-extension-factor', type=int, default=1)
    seed.add_argument('--frequency-double-cutoff', type=float, default=1e-20)
    seed.add_argument('--epoch-double-cutoff', type=float, default=1e-18)
    seed.add_argument('--val-epochs', type=int, default=2000)
    seed.add_argument('--epoch-limit', type=int, default=100000)
    seed.add_argument('--overwrite', action='store_true')
    return parser


def main(argv=None):
    """Run an eigenvalue scan and optionally create a selected branch predictor."""
    args = build_parser().parse_args(argv)
    scan = scan_branch_eigenvalues(
        args.data_directory,
        multipliers=args.multipliers,
        num_eigenvalues=args.num_eigenvalues,
        limit=args.limit,
        frequency_cutoff_mode=args.frequency_cutoff_mode,
        use_existing_sidecars=args.use_existing_sidecars,
        write_sidecars=args.write_sidecars,
    )
    scan_output = Path(args.scan_output)
    if not scan_output.is_absolute():
        scan_output = Path(args.data_directory) / scan_output
    save_scan(scan, scan_output)

    summary = {
        'points': len(scan.matches),
        'multipliers': list(scan.multipliers),
        'num_eigenvalues': scan.num_eigenvalues,
        'scan_file': str(scan_output),
    }
    if args.seed_output is not None:
        if args.source_point_index is None:
            raise ValueError('--source-point-index is required with --seed-output')
        seed = seed_bifurcation(
            scan,
            args.seed_output,
            source_point_index=args.source_point_index,
            period_multiplier=args.seed_multiplier,
            eigenvector_index=args.seed_eigenvector_index,
            step_angle_degrees=args.step_angle_degrees,
            first_step_multiplier=args.first_step_multiplier,
            orientation_component=args.orientation_component,
            orientation_sign=args.orientation_sign,
            stationary_starts=stationary_starts_from_name(args.stationary_starts),
            frequency_extension_factor=args.frequency_extension_factor,
            frequency_double_cutoff=args.frequency_double_cutoff,
            epoch_double_cutoff=args.epoch_double_cutoff,
            val_epochs=args.val_epochs,
            epoch_limit=args.epoch_limit,
            use_existing_sidecars=args.use_existing_sidecars,
        )
        save_bifurcation_seed(seed, overwrite=args.overwrite)
        summary['seed'] = {
            'source_match': seed.source_match,
            'eigenvalue': seed.eigenvalue,
            'seed_loss': seed.seed_loss,
            'closure_error_degrees': seed.closure_error_degrees,
            'branch_file': str(Path(args.seed_output) / 'propagation.pt'),
        }

    print(json.dumps(summary, indent=2))
    return scan


if __name__ == '__main__':
    main()
