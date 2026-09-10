import math
from typing import Optional
import torch
import numpy as np
import scipy.integrate, scipy.optimize, scipy.fft

from .orbit import Orbit_Interface

class Auto_Integrator_Module(object):
    """Generate a Fourier-loop guess from numerical trajectory recurrence.

    Initialize with a system and state, call :meth:`auto_minimize_distance` to
    locate a recurrence, then retrieve the estimated period and coefficients
    with :meth:`get_T` and :meth:`get_theta`.
    """
    INITIAL_SEARCH_FACTOR = 16
    SEARCH_INCREASE_FACTOR = 2
    DEFAULT_BOUNDS_DELTA = 0.01
    IVP_RTOL = 2.220446049250313e-14
    IVP_ATOL = 1e-18
    MIN_TOL = 1e-15
    DEFAULT_NUM_SAMPLES = 101
    

    def __init__(self, system: Orbit_Interface, z0: np.ndarray, tmin: float=1, tstep: float=1) -> None:
        self.system = system
        self.z0 = z0
        self.result = None
        self.tspan: Optional[tuple] = None
        self.T: Optional[float] = None
        self.tmin = tmin
        self.tstep = tstep
    
    @classmethod
    def solve_ivp(cls, system: Orbit_Interface, z0: np.ndarray, tspan: tuple):
        """Integrate ``system`` from ``z0`` over ``tspan`` with dense output."""
        f_np = lambda t, z : system.f(torch.from_numpy(z)).detach().numpy()
        result = scipy.integrate.solve_ivp(f_np, tspan, z0, dense_output=True, vectorized=True, method='DOP853', rtol=cls.IVP_RTOL, atol=cls.IVP_ATOL)
        return result
    
    def parameter_space_distance(self, t: float):
        """Return the distance between the trajectory at ``t`` and ``z0``."""
        return np.linalg.norm(self.result.sol(t) - self.z0)

    def parameter_space_distance_vec(self, t_vector: np.ndarray) -> np.ndarray:
        """Return recurrence distances for all times in ``t_vector``."""
        a = self.result.sol(t_vector) - self.z0.reshape((-1, 1))
        return np.linalg.norm(a, axis=0)

    def get_plot_vectors(self, num_samples: int=DEFAULT_NUM_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
        """Return time and distance arrays suitable for plotting."""

        t_vector = np.linspace(*self.tspan, num_samples)
        return t_vector, self.parameter_space_distance_vec(t_vector)

    def auto_minimize_distance(self):
        """Find and store the next recurrence-distance minimum using Brent's method."""
        if self.tspan is None:
            self.tspan = (0, self.INITIAL_SEARCH_FACTOR * self.tmin)
            self.result = self.solve_ivp(self.system, self.z0, self.tspan)

        while self.tmin < self.tspan[1]:
            tmax = self.tmin + self.tstep
            result = scipy.optimize.minimize_scalar(self.parameter_space_distance, bracket=(self.tmin, tmax), tol=self.MIN_TOL, method='brent')
            if result.x > self.tmin:
                self.T = result.x
                self.tmin = math.ceil(result.x / self.tstep) * self.tstep
                return result
            self.tmin = tmax

        self.tspan = (0, self.tspan[1] * self.SEARCH_INCREASE_FACTOR)
        self.result = self.solve_ivp(self.system, self.z0, self.tspan)
        return self.auto_minimize_distance()

    def manual_minimize_distance(self, tspan, target, rel_bounds=(1 - DEFAULT_BOUNDS_DELTA, 1 + DEFAULT_BOUNDS_DELTA)):
        """Find a recurrence minimum bracketed relative to an expected period."""
        self.tspan = tspan
        self.result = self.solve_ivp(self.system, self.z0, tspan)

        return scipy.optimize.minimize_scalar(self.parameter_space_distance, 
                bracket=(rel_bounds[0] * target, rel_bounds[1] * target), tol=self.MIN_TOL, method='brent')


    def get_T(self):
        """Return the estimated period as a tensor in the default dtype."""
        return torch.tensor([self.T]).type(torch.get_default_dtype())
 
    def get_theta(self, frequency_cutoff: int):
        """Estimate Fourier coefficients through ``frequency_cutoff``."""
        
        fft_span = (0, self.T)
        fft_times = np.linspace(*fft_span, 2 * frequency_cutoff + 1)
        fft_vector = self.result.sol(fft_times)
        fft_result = scipy.fft.rfft(fft_vector, norm='forward')

        theta_result = np.hstack((np.real(fft_result)[:, 0:1], 2 * np.real(fft_result)[:, 1:], -2 * np.imag(fft_result)[:, 1:]))
        return torch.from_numpy(theta_result).type(torch.get_default_dtype())
