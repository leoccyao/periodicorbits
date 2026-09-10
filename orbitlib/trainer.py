import torch
import numpy as np
from typing import Generator, Optional, NoReturn
from functools import cached_property

from .orbit import *
from .integrator import *
from .module.module import *
from .reg.regularizer import Regularizer
from .hessian import *
from .theta_manipulate import *
from .caching import cached_method, Cached_Methods

from .train import train_target

class Trainer(object):
    """Optimize a periodic-orbit model and retain diagnostic checkpoints."""

    def __init__(self, system: Orbit_Interface, model: Theta_T_Abstract_Module, optimizer: torch.optim.Optimizer, regularizer: Optional[Regularizer]=None, epoch: int=0, first_checkpoint=True):
        self.system = system
        self.model = model
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.epoch = epoch
        self.checkpoints: list[Training_Progress] = []
        if first_checkpoint:
            self.checkpoints.append(self.progress())

    def train_epochs(self, epochs: int, print_per: Optional[int]=None, checkpoint=True):
        """Train for ``epochs`` and return the resulting progress snapshot.

        ``print_per`` is retained for compatibility with existing workflows but
        does not currently produce output. Set ``checkpoint`` to append the
        returned snapshot to :attr:`checkpoints`.
        """
        for _ in range(epochs):

            def closure():
                self.optimizer.zero_grad()
                theta, T = self.model()
                res = self.system.loss(theta, T)
                if self.regularizer:
                    res += self.regularizer(theta, T)
                
                res.backward()
                return res

            epoch_num = self.epoch

            self.optimizer.step(closure)

            self.epoch += 1
        
        progress = self.progress()
        if checkpoint:
            self.checkpoints.append(progress)
        return progress


    def train(self, target: 'train_target.Train_Target', epoch_limit: int, val_per: int, print_validation: Optional[int]=None, print_per: Optional[int]=None, checkpoint=True):
        """Train in ``val_per`` blocks until ``target`` succeeds or the limit is reached.

        The effective maximum is ``val_per * (epoch_limit // val_per)`` epochs.
        ``print_per`` is forwarded to :meth:`train_epochs` for compatibility;
        only ``print_validation`` currently emits progress output.
        """
        training_checker = target()
        next(training_checker)
        for _ in range(epoch_limit // val_per):
            progress = self.train_epochs(val_per, print_per, checkpoint)

            if print_validation and self.epoch % print_validation == 0:
                print(f'{self.epoch} {progress.train_loss} {progress.val_loss}')
            if (check_res := training_checker.send((progress, self))):
                return check_res
        return False

    def extrapolate_convergence(self, checkpoint_lookback: int, use_last_checkpoint=True, checkpoint=True, remove_old_checkpoint=False):
        """Extrapolate along the recent convergence direction.

        ``checkpoint_lookback`` selects the earlier snapshot defining the
        direction. This operation does not increment the epoch count.
        """
        
        current_progress = self.checkpoints[-1] if use_last_checkpoint else self.progress()
        lookback_progress = self.checkpoints[-checkpoint_lookback - (1 if use_last_checkpoint else 0)]

        cur_theta, cur_T = current_progress.theta, current_progress.T
        back_theta, back_T = lookback_progress.theta, lookback_progress.T

        dtheta = cur_theta - back_theta
        dT = cur_T - back_T

        def extrapolate(alpha):
            return cur_theta + alpha * dtheta, cur_T + alpha * dT

        def loss(alpha):
            theta, T = extrapolate(alpha)
            return self.system.loss(theta.detach(), T.detach())
        
        result = scipy.optimize.minimize_scalar(loss, bracket=(0, 1))

        new_theta, new_T = extrapolate(result.x)

        self.model = self.model.create_instance(new_theta.detach(), new_T.detach())

        if remove_old_checkpoint:
            self.checkpoints.pop()

        progress = self.progress()
        if checkpoint:
            self.checkpoints.append(progress)
        return progress


    def progress(self):
        """Return a progress snapshot without appending it to checkpoints."""
        return Training_Progress(self)


    def validate(self, num_orbits=1):
        """Return the terminal state mismatch after ``num_orbits`` periods."""

        progress = self.progress()
        T, initial_condition = progress.T, progress.initial_condition

        z0_eval: np.ndarray = initial_condition.detach().numpy()
        tspan_eval = (0, T.item() * num_orbits)

        result_eval = Auto_Integrator_Module.solve_ivp(self.system, z0_eval.reshape(-1), tspan_eval)

        return np.linalg.norm(result_eval.sol(tspan_eval[1]) - z0_eval.reshape(-1))

    def get_checkpoints_data(self, key: str):
        """Return one attribute from every saved checkpoint."""
        return [getattr(epoch, key) for epoch in self.checkpoints]

class Training_Progress(Cached_Methods):
    """Frozen model snapshot with lazily evaluated diagnostics."""
    EVAL_TSPAN_FACTOR = 1.5

    def __init__(self, trainer: Trainer):
        super().__init__()

        self.system = trainer.system
        self.regularizer = trainer.regularizer

        self.epoch = trainer.epoch

        self.model = trainer.model
        theta, T = self.model.forward()
        self.theta = theta.clone()
        self.T = T.clone()

    @cached_property
    def initial_condition(self):
        return self.system.z0(self.theta)

    @cached_property
    def train_loss(self):
        return self.system.loss(self.theta, self.T)
    
    @cached_property
    def reg_loss(self):
        return self.regularizer(self.theta, self.T) if self.regularizer else torch.zeros(1,)

    @cached_property
    def total_loss(self):
        return self.train_loss + self.reg_loss

    @cached_property
    def _integrator_stats(self) -> tuple[float, float, float]:
        """Compute and cache validation and nearby-recurrence diagnostics."""
        z0_eval: np.ndarray = self.initial_condition.detach().numpy()

        integrator = Auto_Integrator_Module(self.system, z0_eval.reshape(-1))
        tspan = (0, self.EVAL_TSPAN_FACTOR * self.T.item())        

        result = integrator.manual_minimize_distance(tspan, self.T.item())
        eval_T, eval_loss = result.x, result.fun
        val_loss = integrator.parameter_space_distance(self.T.item())
        
        return val_loss, eval_T, eval_loss

    @cached_property
    def val_loss(self):
        return self._integrator_stats[0]
    
    @cached_property
    def eval_T(self):
        return self._integrator_stats[1]
    
    @cached_property
    def eval_loss(self):
        return self._integrator_stats[2]
    
    def _transformed_model(self, frequency_cutoff, reset_module, mult_periods):
        return self.model.transform_instance(
            lambda theta, T: (
                period_multiply(theta, T, mult_periods, new_freq_cutoff=frequency_cutoff)
            ),
            Theta_T_Module(self.theta, self.T) if reset_module else self.model
        )

    @cached_method
    def _hessian(self, full_theta, include_T, create_graph, frequency_cutoff, mult_periods, /, _cache_controls):
        transformed_model = self._transformed_model(frequency_cutoff, full_theta, mult_periods)

        return hessian(self.system, transformed_model, full_theta, include_T, create_graph)


    def hessian(self, full_theta=False, include_T=True, create_graph=False, frequency_cutoff=None, mult_periods=1, _cache_controls=None):
        if frequency_cutoff is None:
            frequency_cutoff = self.model.frequency_cutoff * mult_periods

        return self._hessian(full_theta, include_T, create_graph, frequency_cutoff, mult_periods, _cache_controls=_cache_controls)


    @cached_method
    def _conditions_theta(self, full_theta, include_T, create_graph, frequency_cutoff, mult_periods, num_eig, /, _cache_controls):
        transformed_model = self._transformed_model(frequency_cutoff, full_theta, mult_periods)

        hess = self._hessian(full_theta, include_T, create_graph, frequency_cutoff, mult_periods, _cache_controls=_cache_controls)
        return conditions_from_hess(hess, transformed_model, full_theta, include_T, num_eig)


    def conditions_theta(self, full_theta=False, include_T=True, create_graph=False, frequency_cutoff=None, mult_periods=1, num_eig=None, _cache_controls=None):
        if frequency_cutoff is None:
            frequency_cutoff = self.model.frequency_cutoff * mult_periods

        return self._conditions_theta(full_theta, include_T, create_graph, frequency_cutoff, mult_periods, num_eig, _cache_controls=_cache_controls)


    @cached_method
    def _conditions_phase(self, full_theta, include_T, create_graph, frequency_cutoff, mult_periods, num_eig, /, _cache_controls):
        eigv, eigs_packed = self._conditions_theta(full_theta, include_T, create_graph, frequency_cutoff, mult_periods, num_eig, _cache_controls=_cache_controls)
        return phase_from_conditions_theta(self.system, eigv, eigs_packed)


    def conditions_phase(self, full_theta=False, include_T=True, create_graph=False, frequency_cutoff=None, mult_periods=1, num_eig=None, _cache_controls=None):
        if frequency_cutoff is None:
            frequency_cutoff = self.model.frequency_cutoff * mult_periods

        return self._conditions_phase(full_theta, include_T, create_graph, frequency_cutoff, mult_periods, num_eig, _cache_controls=_cache_controls)


Training_Checker = Generator[bool, tuple[Training_Progress, Trainer], NoReturn]
