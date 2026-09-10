import torch
import torch.nn.utils.parametrize as parametrize

import numpy as np
from orbitlib import *
from doublepend import *
from run_train_job import *
from analysis import *
from curve_prop import *
import time
import datetime
from contextlib import contextmanager

torch.set_default_dtype(torch.float64)

def tprint(*args, **kwargs):
    print(datetime.datetime.now().strftime('%H:%M:%S'), *args, **kwargs)

@contextmanager
def modification_logging(sleep=3, message='parameters', enabled=True):
    if enabled:
        tprint(f'Preparing to modify {message} - do not interrupt')
        time.sleep(sleep)
        tprint(f'Modifying {message} in progress')

    try:
        yield None
    finally:
        if enabled:
            tprint(f'Finished modifying {message}')


class Branch_Propagation(object):    
    """Stateful predictor-corrector propagation along a periodic-orbit branch."""
    def __init__(self, module: type[Theta_T_Abstract_Module],
                 theta0: torch.Tensor, d_z0: torch.Tensor, T0: torch.Tensor, stationary_starts: tuple[bool, bool], results_folder: str,
                 step_angle: float = 1 * np.pi / 180, wparam: bool=True,
                 frequency_double_cutoff=1e-20, epoch_double_cutoff=1e-18,
                 val_epochs=2000, epoch_limit=100000) -> None:
        self.module = module

        self.theta0 = theta0
        self.d_z0 = d_z0
        self.T0 = T0

        self.results_folder = results_folder
        self.frequency_cutoff = (theta0.shape[1] - 1) // 2
        self.integral_num_samples = self.frequency_cutoff * SYSTEM.INTEGRAL_NUM_SAMPLES_FACTOR
        self.frequency_double_cutoff = frequency_double_cutoff

        self.epoch_double_cutoff = epoch_double_cutoff
        self.val_epochs = val_epochs
        self.epoch_limit = epoch_limit

        self.stationary_starts = stationary_starts
        self.theta0, self.grad_active = process_stationary_starts(self.theta0, self.stationary_starts)

        self.step_angle = step_angle
        self.wparam = wparam

        self.num_trainers = 0
        self.total_time = 0


    @classmethod
    def from_iv(cls, module: type[Theta_T_Abstract_Module],
                 starting_point: list[float], iv_deviation: list[float], stationary_starts: tuple[bool, bool], results_folder: str,
                 step_angle: float = 1 * np.pi / 180, wparam: bool=True,
                 num_minima=1, frequency_cutoff=16, frequency_double_cutoff=1e-20, epoch_double_cutoff=1e-18,
                 val_epochs=2000, epoch_limit=100000) -> None:
        
        d_z0 = torch.tensor(iv_deviation).reshape((4, 1)) * np.pi / 180
        iv = [sum(x) for x in zip(starting_point, iv_deviation)]
        theta0, T0 = get_initial_conditions(iv, frequency_cutoff, num_minima)
        
        return cls(module, theta0, d_z0, T0, stationary_starts, results_folder,
                   step_angle, wparam,
                   frequency_double_cutoff, epoch_double_cutoff,
                   val_epochs, epoch_limit)


    def run_iteration(
            self, add_mode=False, perpendicularize=False, override_param_index=None,
            optim_callback=get_default_optim, param_class = param.Weighted_Multi_Theta_Sum_Param,
            extrapolate=False, epoch_doubling=True, frequency_doubling=True, **kwargs
    ) -> Trainer:
        """Attempt one corrected branch step and return its trainer.

        When adaptive epoch or frequency doubling is triggered, the attempted
        result is archived and the propagation state is prepared for a retry;
        ``num_trainers`` is not advanced. Disabling both safeguards preserves
        the research workflow's behavior of accepting the resulting trainer.
        """
        startTime = time.time()

        model = self.module(self.theta0.clone().detach(), self.T0.clone().detach(), self.grad_active)

        if self.wparam:
            if override_param_index is not None:
                constraint_map = [(
                    [(override_param_index, 0, 1)], self.theta0[override_param_index, 0:self.frequency_cutoff+1].sum().detach()
                )]

            else:
                weighted_orig_theta = sum(
                    self.d_z0[i, 0] * self.theta0[i, 0:self.frequency_cutoff+1].sum().detach()
                    for i in range(4)
                )

                param_list = [
                    (i, 0, self.d_z0[i, 0]) for i in range(4)
                ]

                constraint_map = [(
                    param_list,
                    weighted_orig_theta
                )]
        
        if extrapolate:
            param_callback = lambda model: (
                    'theta_enabled', param_class(
                    model, constraint_map,
                    add_mode=add_mode
                )
            )
            deparam_callback = lambda model: 'theta_enabled'

            result, trainer, training_time = find_sol_model_optimizer_extrapolate_timed(
                model, optim_callback,
                param_callback = param_callback if self.wparam else None,
                deparam_callback = deparam_callback if self.wparam else None,
                epoch_limit=self.epoch_limit,
                **kwargs
            )

            model = trainer.model

        else:
            optimizer = optim_callback(model)

            if self.wparam:
                parametrize.register_parametrization(model, 'theta_enabled', param_class(
                    model, constraint_map,
                    add_mode=add_mode
                ))

            result, trainer, training_time = find_sol_model_optimizer_timed(
                model, optimizer, val_epochs=self.val_epochs, epoch_limit=self.epoch_limit, **kwargs
            )

            if self.wparam:
                parametrize.remove_parametrizations(model, 'theta_enabled')
        
        last_checkpoint = trainer.checkpoints[-1]
 
        if epoch_doubling and (not result or last_checkpoint.train_loss > self.epoch_double_cutoff):
            with modification_logging(enabled=False):
                dir_prefix = f'{self.results_folder}/{[self.val_epochs, self.epoch_limit]}'
                save_data(result, trainer, training_time, dir_prefix)
                torch.save(self, f'{dir_prefix}/propagation.pt')

                self.val_epochs *= 2
                self.epoch_limit *= 2

            tprint(self.num_trainers, int(self.total_time), (self.val_epochs, self.epoch_limit, trainer.checkpoints[-1].train_loss.item(), trainer.checkpoints[-1].val_loss))
            return trainer


        if frequency_doubling and (last_checkpoint.train_loss > self.frequency_double_cutoff):
            with modification_logging(enabled=False):
                dir_prefix = f'{self.results_folder}/{[self.frequency_cutoff, self.integral_num_samples]}'
                save_data(result, trainer, training_time, dir_prefix)
                torch.save(self, f'{dir_prefix}/propagation.pt')

                self.theta0 = frequency_multiply(self.theta0, 2)
                self.frequency_cutoff *= 2
                self.integral_num_samples *= 2

                self.total_time += (time.time() - startTime)

            tprint(self.num_trainers, int(self.total_time), (self.frequency_cutoff, self.integral_num_samples, trainer.checkpoints[-1].train_loss.item(), trainer.checkpoints[-1].val_loss))
            return trainer

        final_cond_iv = trainer.checkpoints[-1].initial_condition.detach().numpy().ravel() * 180 / np.pi
        dir_prefix = f'{self.results_folder}/{self.num_trainers} {[round(i, 5) for i in final_cond_iv]}'

        eigenvals, conditions = last_checkpoint.conditions_theta(num_eig=8)

        with modification_logging(enabled=False):
            save_data(result, trainer, training_time, dir_prefix)
            torch.save(self, f'{dir_prefix}/propagation.pt')
            self.num_trainers += 1

            self.update_self(trainer, conditions, perpendicularize=perpendicularize)

            self.total_time += (time.time() - startTime)

        tprint(self.num_trainers, int(self.total_time), trainer.checkpoints[-1].train_loss.item(), trainer.checkpoints[-1].val_loss, eigenvals[0].item(), final_cond_iv)  
        return trainer

    def update_self(self, trainer: Trainer, conditions, perpendicularize=False):
        """Advance the predictor from a corrected orbit and Hessian condition.

        Set ``perpendicularize`` when the time-translation direction is mixed
        into the first two returned condition vectors.
        """

        if perpendicularize:
            t1, T1 = conditions[0]
            t2, T2 = conditions[1]
            d_theta, dT = propagation_theta(t1, t2, T1, T2)
        else:
            d_theta, dT = conditions[0]
        
        last_checkpoint = trainer.checkpoints[-1]
        d_z0 = trainer.system.z0(d_theta)

        if (self.d_z0.T @ d_z0 < 0):
            d_theta = -d_theta
            dT = -dT
            d_z0 = -d_z0

        self.d_z0 = d_z0.clone().detach()

        d_z0_norm = torch.linalg.norm(d_z0)
        step_size = self.step_angle / d_z0_norm

        self.theta0 = last_checkpoint.theta + step_size * d_theta
        self.theta0, _ = process_stationary_starts(self.theta0, self.stationary_starts)

        self.T0 = last_checkpoint.T + step_size * dT

        self.theta0 = self.theta0.clone().detach()
        self.T0 = self.T0.clone().detach()

    def save_self(self):
        """Serialize the current propagation state in ``results_folder``."""
        with modification_logging(message='object on disk', enabled=False):
            torch.save(self, f'{self.results_folder}/propagation.pt')
