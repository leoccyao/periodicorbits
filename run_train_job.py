import argparse
import time
import torch
import torch.nn.utils.parametrize as parametrize
import numpy as np
from orbitlib import *
from doublepend import *
import json
import time
import os, sys
from functools import wraps
import warnings

torch.set_default_dtype(torch.float64)

SYSTEM = Double_Pendulum_Orbit()

def find_sol(iv, regularizer_callback=None, stationary_starts: tuple[bool, bool]=(True, True), frequency_cutoff=64, **kwargs):
    """Find a periodic orbit from an initial state specified in degrees.

    ``regularizer_callback``, when supplied, is called with ``SYSTEM`` and
    ``iv`` and must return a regularizer accepted by :class:`Trainer`.
    """
    theta0, T0 = get_initial_conditions(iv, frequency_cutoff)

    theta0, grad_active = process_stationary_starts(theta0, stationary_starts)

    regularizer = None
    if regularizer_callback:
        regularizer = regularizer_callback(SYSTEM, iv) 

    return find_sol_theta(theta0, T0, grad_active, regularizer=regularizer, **kwargs)


def get_initial_conditions(iv, frequency_cutoff, num_minima=1):
    """Construct Fourier coefficients and a period from an initial state.

    ``iv`` is in degrees and may contain two angles or the complete
    four-component state (angles followed by angular velocities).
    """
    if len(iv) == 2:
        init = [*iv, 0, 0]
    elif len(iv) == 4:
        init = iv
    else:
        raise AssertionError(f'iv must be length 2 or 4, got length {len(iv)} for iv {iv}')

    z0 = np.array(init) * np.pi / 180
    integrator = Auto_Integrator_Module(SYSTEM, z0)

    for _ in range(num_minima):
        minimization_result = integrator.auto_minimize_distance()

    theta0 = integrator.get_theta(frequency_cutoff)
    T0 = integrator.get_T()

    return theta0, T0


def process_stationary_starts(theta: torch.Tensor, stationary_starts: tuple[bool, bool]):
    """Apply stationary-start symmetry constraints in place.

    Return the modified coefficient tensor and the cosine/sine trainability
    mask used by :class:`orbitlib.module.Theta_Grad_T_Module`.
    """
    grad_active = [[True, True] for _ in range(4)]

    frequency_cutoff = (theta.shape[1] - 1) // 2

    if stationary_starts[0]:
        theta[0][frequency_cutoff+1:] = 0
        theta[2][:frequency_cutoff+1] = 0
        grad_active[0] = [True, False]
        grad_active[2] = [False, True]

    if stationary_starts[1]:
        theta[1][frequency_cutoff+1:] = 0
        theta[3][:frequency_cutoff+1] = 0
        grad_active[1] = [True, False]
        grad_active[3] = [False, True]

    return theta, grad_active


def find_sol_theta(theta0, T0, grad_active, **kwargs):
    """Find a solution from Fourier coefficients, period, and gradient mask."""
    model = module.Theta_Grad_T_Module(theta0, T0, grad_active)

    return find_sol_model(model, **kwargs)


def get_default_optim(model, learning_rate=1e-3, step_sizes=(1e-100, 1e-1)):
    return torch.optim.Rprop(model.parameters(), lr=learning_rate, step_sizes=step_sizes)


def find_sol_model(model, learning_rate=1e-3, step_sizes=(1e-100, 1e-1), **kwargs):
    """Find a solution from a constructed model using the default optimizer."""

    optimizer = get_default_optim(model, learning_rate, step_sizes)

    return find_sol_model_optimizer(model, optimizer, **kwargs)


def find_sol_model_optimizer(model, optimizer, regularizer=None, val_per=200, val_epochs=1000, epoch_limit=100000, **kwargs):
    """Find a solution from a constructed model and optimizer."""
    if 'integral_num_samples' in kwargs:
        kwargs.pop('integral_num_samples')
        warnings.warn('integral_num_samples no longer has any effect, removing from passed arguments')

    trainer = Trainer(SYSTEM, model, optimizer, regularizer)

    train_target = target.With_Val_Requirement(
        target.With_Val_Limit(
            target.Val_Decrease_Epochs(val_epochs)
        )
    )
    return trainer.train(train_target, epoch_limit=epoch_limit, val_per=val_per, **kwargs), trainer


def find_sol_model_optimizer_extrapolate(
        model, optim_callback: Callable[[Theta_T_Abstract_Module], torch.optim.Optimizer],
        param_callback: Callable[[Theta_T_Abstract_Module], tuple[str, torch.nn.Module]]=None,
        deparam_callback: Callable[[Theta_T_Abstract_Module], str]=None,
        regularizer=None, val_per=200, extrapolate_epochs=10000, extrapolate_cutoff=1.1,
        extrapolate_count=1, extrapolate_checkpoints=25, epoch_limit=100000,
        print_extrapolation=False, **kwargs
):
    """Train in blocks and extrapolate along the observed convergence direction.

    Training stops after ``extrapolate_count`` consecutive extrapolations fail
    to improve validation loss by the factor ``extrapolate_cutoff``. Optional
    callbacks install and remove a model parametrization around each block.
    """

    trainer = Trainer(SYSTEM, model, optim_callback(model), regularizer)

    failed_extrapolations = 0

    for _ in range(epoch_limit // extrapolate_epochs):
        if param_callback is not None:
            tensor_name, parametrization = param_callback(trainer.model)
            parametrize.register_parametrization(trainer.model, tensor_name, parametrization)

        trainer.train(target.Train_Target(), epoch_limit=extrapolate_epochs, val_per=val_per, **kwargs)

        if deparam_callback is not None:
            tensor_name = deparam_callback(trainer.model)
            parametrize.remove_parametrizations(trainer.model, tensor_name)

        trainer.extrapolate_convergence(extrapolate_checkpoints)
        trainer.optimizer = optim_callback(trainer.model)

        extrapolated_val = trainer.checkpoints[-1].val_loss
        previous_val = trainer.checkpoints[-2].val_loss

        if print_extrapolation:
            print(extrapolated_val, previous_val)

        if extrapolated_val * extrapolate_cutoff > previous_val:
            failed_extrapolations += 1
            if failed_extrapolations >= extrapolate_count:
                return True, trainer
        else:
            failed_extrapolations = 0

    return False, trainer


def training_time(func):
    """Decorate a function so its elapsed wall time is appended to its result."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        run_time = time.time() - start

        if type(result) == tuple:
            return *result, run_time
        else:
            return result, run_time
    
    return wrapper


@training_time
def find_sol_timed(iv, **kwargs):
    """Timed form of :func:`find_sol`."""
    return find_sol(iv, **kwargs)


@training_time
def find_sol_theta_timed(theta0, T0, grad_active, **kwargs):
    """Timed form of :func:`find_sol_theta`."""
    return find_sol_theta(theta0, T0, grad_active, **kwargs)


@training_time
def find_sol_model_timed(model, **kwargs):
    """Timed form of :func:`find_sol_model`."""
    return find_sol_model(model, **kwargs)


@training_time
def find_sol_model_optimizer_timed(model, optimizer, **kwargs):
    """Timed form of :func:`find_sol_model_optimizer`."""
    return find_sol_model_optimizer(model, optimizer, **kwargs)


@training_time
def find_sol_model_optimizer_extrapolate_timed(model, optim_callback, **kwargs):
    """Timed form of :func:`find_sol_model_optimizer_extrapolate`."""
    return find_sol_model_optimizer_extrapolate(model, optim_callback, **kwargs)


def find_sol_wrapper(folder, iv, custom_subdir=None, **kwargs):
    """Run and save an orbit unless an existing result file is readable.

    Return ``True`` after computing a result and ``False`` when the output
    directory already contains readable result metadata.
    """
    dir_prefix = f'{folder}/{str(iv)}'
    if custom_subdir:
        dir_prefix += f'/{custom_subdir}'

    os.makedirs(dir_prefix, exist_ok=True)

    if os.path.exists(f'{dir_prefix}/var-results.json'):
        try:
            with open(f'{dir_prefix}/var-results.json') as results_file:
                result_from_file = json.load(results_file)
            return False
        except UnicodeDecodeError:
            pass
        except json.JSONDecodeError:
            pass
    
    result, trainer, training_time = find_sol_timed(iv, **kwargs)

    save_data(result, trainer, training_time, dir_prefix)

    return True


def eval_checkpoint_dict(trainer: Trainer):
    """Evaluates the last checkpoint and returns both it and the dictionary."""
    last_checkpoint = trainer.checkpoints[-1]

    checkpoint_attrs = ['epoch', 'T', 'initial_condition', 'train_loss', 'reg_loss', 'total_loss', 'val_loss', 'eval_T', 'eval_loss']
    checkpoint_dict: dict[str, float | list] = {}
    
    for key in checkpoint_attrs:
        value = getattr(last_checkpoint, key)
        if type(value) == torch.Tensor or type(value) == torch.nn.Parameter:
            value: np.ndarray = value.detach().numpy()

            if value.size == 1:
                value: float = value.item()
            else:
                value: list = value.tolist()

        checkpoint_dict[key] = value
    
    return last_checkpoint, checkpoint_dict


def save_data(result, trainer: Trainer, training_time, dir_prefix):
    """Save a trainer and summary metadata, then return its last checkpoint."""
    last_checkpoint, checkpoint_dict = eval_checkpoint_dict(trainer)

    results_dict = {
        'result': bool(result),
        'time': training_time,
        'checkpoint': checkpoint_dict,
    }

    os.makedirs(dir_prefix, exist_ok=True)
    torch.save(trainer, f'{dir_prefix}/var-model.pt')
    with open(f'{dir_prefix}/var-results.json', 'w') as results_file:
        json.dump(results_dict, results_file)

    return last_checkpoint

def main(argv=None):
    """Command-line entry point for finding and saving one periodic orbit."""
    parser = argparse.ArgumentParser(
        description=(
            'Initialize a Fourier loop from a double-pendulum trajectory, '
            'optimize it as a periodic orbit, and save the result.'
        )
    )
    parser.add_argument('output_folder', help='Directory in which results are stored.')
    parser.add_argument('theta1', type=float, help='Initial upper-arm angle in degrees.')
    parser.add_argument('theta2', type=float, help='Initial lower-arm angle in degrees.')
    parser.add_argument(
        'subdir', nargs='?', default=None,
        help='Optional subdirectory below the initial-condition directory.',
    )
    parser.add_argument('--frequency-cutoff', type=int, default=64)
    parser.add_argument('--val-epochs', type=int, default=1000)
    parser.add_argument('--epoch-limit', type=int, default=100000)
    parser.add_argument('--val-per', type=int, default=200)
    parser.add_argument(
        '--stationary-starts',
        choices=('both', 'upper', 'lower', 'none'),
        default='both',
        help='Which pendulum arms are constrained to start at a stationary phase.',
    )

    args = parser.parse_args(argv)
    stationary_starts = {
        'both': (True, True),
        'upper': (True, False),
        'lower': (False, True),
        'none': (False, False),
    }[args.stationary_starts]

    return find_sol_wrapper(
        args.output_folder,
        (args.theta1, args.theta2),
        args.subdir,
        stationary_starts=stationary_starts,
        frequency_cutoff=args.frequency_cutoff,
        val_epochs=args.val_epochs,
        epoch_limit=args.epoch_limit,
        val_per=args.val_per,
    )


if __name__ == '__main__':
    main()
