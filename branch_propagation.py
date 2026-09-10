"""Resume and propagate a saved double-pendulum continuation branch.

The computational workflow formerly driven from ``branch_propagation.ipynb``
is implemented here once.  The companion notebook imports these functions for
interactive monitoring and plots; this module's :func:`main` provides the
noninteractive end-to-end entry point.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from branch import Branch_Propagation
from orbitlib import Trainer, param


torch.set_default_dtype(torch.float64)


@dataclass
class PropagationAttempt:
    """Summary of one corrector attempt and its resulting branch state."""

    attempt: int
    accepted_point: bool
    point_index_before: int
    point_index_after: int
    epoch: int
    train_loss: float
    validation_loss: float
    period: float
    frequency_cutoff: int
    val_epochs: int
    epoch_limit: int
    total_time: float


def propagation_path(results_folder: str | Path) -> Path:
    """Resolve either a results directory or an explicit propagation file."""
    path = Path(results_folder)
    return path if path.name == 'propagation.pt' else path / 'propagation.pt'


def load_branch(
    results_folder: str | Path,
    *,
    map_location: str | torch.device = 'cpu',
) -> Branch_Propagation:
    """Load branch state and direct future output to the supplied directory."""
    path = propagation_path(results_folder)
    if not path.is_file():
        raise FileNotFoundError(f'branch state does not exist: {path}')
    branch = torch.load(path, map_location=map_location)
    if not isinstance(branch, Branch_Propagation):
        raise TypeError(f'{path} does not contain Branch_Propagation state')
    branch.results_folder = str(path.parent)
    return branch


def branch_summary(branch: Branch_Propagation) -> dict:
    """Return the resumable continuation parameters most useful for inspection."""
    return {
        'results_folder': branch.results_folder,
        'module': branch.module.__name__,
        'num_trainers': branch.num_trainers,
        'frequency_cutoff': branch.frequency_cutoff,
        'integral_num_samples': branch.integral_num_samples,
        'stationary_starts': list(branch.stationary_starts),
        'step_angle_degrees': float(branch.step_angle * 180 / math.pi),
        'weighted_parameterization': branch.wparam,
        'val_epochs': branch.val_epochs,
        'epoch_limit': branch.epoch_limit,
        'frequency_double_cutoff': branch.frequency_double_cutoff,
        'epoch_double_cutoff': branch.epoch_double_cutoff,
        'total_time': branch.total_time,
        'period_predictor': float(branch.T0.item()),
        'initial_direction': branch.d_z0.detach().cpu().reshape(-1).tolist(),
    }


def optimizer_callback(
    learning_rate: float = 1e-3,
    step_size_min: float = 1e-100,
    step_size_max: float = 1e-1,
) -> Callable[[torch.nn.Module], torch.optim.Optimizer]:
    """Build the Rprop factory expected by ``Branch_Propagation``."""
    if learning_rate <= 0:
        raise ValueError('learning_rate must be positive')
    if not 0 < step_size_min < step_size_max:
        raise ValueError('step sizes must satisfy 0 < min < max')

    def create_optimizer(model):
        return torch.optim.Rprop(
            model.parameters(),
            lr=learning_rate,
            step_sizes=(step_size_min, step_size_max),
        )

    return create_optimizer


def parameterization_class(name: str):
    """Resolve the continuation constraint implementation by CLI-safe name."""
    choices = {
        'weighted': param.Weighted_Multi_Theta_Sum_Param,
        'factor-weighted': param.Factor_Weighted_Multi_Theta_Sum_Param,
    }
    try:
        return choices[name]
    except KeyError as error:
        raise ValueError(
            f'unknown parameterization {name!r}; choose from {tuple(choices)}'
        ) from error


def summarize_attempt(
    attempt: int,
    point_index_before: int,
    branch: Branch_Propagation,
    trainer: Trainer,
) -> PropagationAttempt:
    checkpoint = trainer.checkpoints[-1]
    return PropagationAttempt(
        attempt=attempt,
        accepted_point=branch.num_trainers > point_index_before,
        point_index_before=point_index_before,
        point_index_after=branch.num_trainers,
        epoch=checkpoint.epoch,
        train_loss=float(checkpoint.train_loss.item()),
        validation_loss=float(checkpoint.val_loss),
        period=float(checkpoint.T.item()),
        frequency_cutoff=branch.frequency_cutoff,
        val_epochs=branch.val_epochs,
        epoch_limit=branch.epoch_limit,
        total_time=float(branch.total_time),
    )


def propagate_branch(
    branch: Branch_Propagation,
    *,
    iterations: int = 1,
    add_mode: bool = False,
    perpendicularize: bool = False,
    override_param_index: int | None = None,
    parameterization: str = 'weighted',
    learning_rate: float = 1e-3,
    step_size_min: float = 1e-100,
    step_size_max: float = 1e-1,
    extrapolate: bool = False,
    epoch_doubling: bool = True,
    frequency_doubling: bool = True,
    val_per: int = 200,
    print_validation: int | None = None,
    print_extrapolation: bool = False,
    extrapolate_epochs: int = 10000,
    extrapolate_cutoff: float = 1.1,
    extrapolate_count: int = 1,
    extrapolate_checkpoints: int = 25,
    save_after_each: bool = True,
) -> tuple[list[PropagationAttempt], Trainer]:
    """Run continuation attempts, saving resumable branch state after each one.

    An attempt need not accept a new point: the legacy implementation may
    instead increase its epoch budget or Fourier cutoff and require a retry.
    ``accepted_point`` in each returned summary makes that distinction explicit.
    """
    if iterations < 1:
        raise ValueError('iterations must be at least 1')
    if val_per < 1:
        raise ValueError('val_per must be at least 1')

    optim_callback = optimizer_callback(
        learning_rate, step_size_min, step_size_max
    )
    param_class = parameterization_class(parameterization)
    attempts = []
    trainer = None

    for attempt_number in range(1, iterations + 1):
        point_index_before = branch.num_trainers
        optional_kwargs = {}
        if print_validation is not None:
            optional_kwargs['print_validation'] = print_validation
        if extrapolate:
            optional_kwargs.update(
                print_extrapolation=print_extrapolation,
                extrapolate_epochs=extrapolate_epochs,
                extrapolate_cutoff=extrapolate_cutoff,
                extrapolate_count=extrapolate_count,
                extrapolate_checkpoints=extrapolate_checkpoints,
            )

        trainer = branch.run_iteration(
            add_mode=add_mode,
            perpendicularize=perpendicularize,
            override_param_index=override_param_index,
            optim_callback=optim_callback,
            param_class=param_class,
            extrapolate=extrapolate,
            epoch_doubling=epoch_doubling,
            frequency_doubling=frequency_doubling,
            val_per=val_per,
            **optional_kwargs,
        )
        attempts.append(
            summarize_attempt(attempt_number, point_index_before, branch, trainer)
        )
        if save_after_each:
            branch.save_self()

    return attempts, trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Resume Hessian-guided continuation from propagation.pt.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'results_folder',
        help='Branch directory containing propagation.pt, or the file itself.',
    )
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--add-mode', action='store_true')
    parser.add_argument('--perpendicularize', action='store_true')
    parser.add_argument('--override-param-index', type=int)
    parser.add_argument(
        '--parameterization',
        choices=('weighted', 'factor-weighted'),
        default='weighted',
    )
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--step-size-min', type=float, default=1e-100)
    parser.add_argument('--step-size-max', type=float, default=1e-1)
    parser.add_argument('--extrapolate', action='store_true')
    parser.add_argument(
        '--epoch-doubling', action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        '--frequency-doubling', action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument('--val-per', type=int, default=200)
    parser.add_argument('--val-epochs', type=int)
    parser.add_argument('--epoch-limit', type=int)
    parser.add_argument('--frequency-double-cutoff', type=float)
    parser.add_argument('--epoch-double-cutoff', type=float)
    parser.add_argument('--print-validation', type=int)
    parser.add_argument('--print-extrapolation', action='store_true')
    parser.add_argument('--extrapolate-epochs', type=int, default=10000)
    parser.add_argument('--extrapolate-cutoff', type=float, default=1.1)
    parser.add_argument('--extrapolate-count', type=int, default=1)
    parser.add_argument('--extrapolate-checkpoints', type=int, default=25)
    parser.add_argument(
        '--inspect-only',
        action='store_true',
        help='Print saved branch state without running continuation.',
    )
    return parser


def apply_branch_overrides(branch: Branch_Propagation, args) -> None:
    """Apply optional saved-state overrides requested on the command line."""
    overrides = {
        'val_epochs': args.val_epochs,
        'epoch_limit': args.epoch_limit,
        'frequency_double_cutoff': args.frequency_double_cutoff,
        'epoch_double_cutoff': args.epoch_double_cutoff,
    }
    for attribute, value in overrides.items():
        if value is not None:
            setattr(branch, attribute, value)


def main(argv=None):
    """Load, optionally inspect, propagate, and persist one branch."""
    args = build_parser().parse_args(argv)
    branch = load_branch(args.results_folder)
    apply_branch_overrides(branch, args)

    if args.inspect_only:
        print(json.dumps(branch_summary(branch), indent=2))
        return branch

    attempts, _ = propagate_branch(
        branch,
        iterations=args.iterations,
        add_mode=args.add_mode,
        perpendicularize=args.perpendicularize,
        override_param_index=args.override_param_index,
        parameterization=args.parameterization,
        learning_rate=args.learning_rate,
        step_size_min=args.step_size_min,
        step_size_max=args.step_size_max,
        extrapolate=args.extrapolate,
        epoch_doubling=args.epoch_doubling,
        frequency_doubling=args.frequency_doubling,
        val_per=args.val_per,
        print_validation=args.print_validation,
        print_extrapolation=args.print_extrapolation,
        extrapolate_epochs=args.extrapolate_epochs,
        extrapolate_cutoff=args.extrapolate_cutoff,
        extrapolate_count=args.extrapolate_count,
        extrapolate_checkpoints=args.extrapolate_checkpoints,
    )
    print(json.dumps([asdict(attempt) for attempt in attempts], indent=2))
    return branch


if __name__ == '__main__':
    main()
