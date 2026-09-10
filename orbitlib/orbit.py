import torch

class Orbit_Interface(object):
    """Base interface for Fourier-represented periodic-orbit systems."""

    num_variables = 0
    INTEGRAL_NUM_SAMPLES_FACTOR = 4

    def __init__(self) -> None:
        assert self.num_variables != 0, "must override num_variables"

    def check_theta_shape(self, func):
        """Decorate an instance method with coefficient-shape assertions."""
        def decorator(self, theta: torch.Tensor, *args, **kwargs):
            assert len(theta.shape) == 2, "incorrect parameter vector size"
            assert theta.shape[0] == self.num_variables, "incorrect number of rows"

            return func(self, theta, *args, **kwargs)
        return decorator


    def check_z_shape(self, func):
        """Decorate an instance method with state-shape assertions."""
        def decorator(self, z: torch.Tensor, *args, **kwargs):
            assert len(z.shape) == 2, "incorrect number of input dimensions"
            assert z.shape[0] == self.num_variables, "incorrect input shape"

            return func(self, z, *args, **kwargs)
        return decorator

    def z(self, theta: torch.Tensor, t: torch.Tensor, T: float) -> torch.Tensor:
        """Evaluate a Fourier loop at sample times.

        ``theta`` has shape ``(num_variables, 2 * cutoff + 1)`` and ``t`` has
        shape ``(num_samples,)``. The result has shape
        ``(num_variables, num_samples)``.
        """

        frequency_cutoff = (theta.shape[1] - 1) // 2
        num_samples = t.shape[0]
        const = theta[:, 0:1]
        a_params = theta[:, 1:frequency_cutoff+1]
        b_params = theta[:, frequency_cutoff+1:]

        angle_coeffs = 2 * torch.pi * t.reshape((num_samples, 1))/ T
        angles = angle_coeffs @ torch.arange(1, frequency_cutoff+1).reshape((1, frequency_cutoff)).type(torch.get_default_dtype())
        cos_coeffs = torch.cos(angles).T
        sin_coeffs = torch.sin(angles).T

        a_term = a_params @ cos_coeffs
        b_term = b_params @ sin_coeffs

        return const + a_term + b_term


    def z0(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate a Fourier loop at time zero.

        Return a state array with shape ``(num_variables, 1)``.
        """
        frequency_cutoff = (theta.shape[1] - 1) // 2
        const = theta[:, 0:1]
        a_params = theta[:, 1:frequency_cutoff+1]

        a_term = a_params.sum(axis=1).reshape((-1, 1))

        return const + a_term


    def dz(self, theta: torch.Tensor, t: torch.Tensor, T: float) -> torch.Tensor:
        """Evaluate the time derivative of a Fourier loop.

        Inputs follow :meth:`z`; the result has shape
        ``(num_variables, num_samples)``.
        """

        frequency_cutoff = (theta.shape[1] - 1) // 2
        num_samples = t.shape[0]
        a_params = theta[:, 1:frequency_cutoff+1]
        b_params = theta[:, frequency_cutoff+1:]

        angle_coeffs = 2 * torch.pi * t.reshape((num_samples, 1)) / T
        angles = angle_coeffs @ torch.arange(1., frequency_cutoff+1).reshape((1, frequency_cutoff))
        der_coeffs_1d = torch.arange(1, frequency_cutoff+1) * (2 * torch.pi / T)
        der_coeffs = der_coeffs_1d.reshape((1, frequency_cutoff)).expand((num_samples, frequency_cutoff))

        cos_coeffs = (torch.cos(angles) * der_coeffs).T
        sin_coeffs = (torch.sin(angles) * der_coeffs).T

        a_term = -a_params @ sin_coeffs
        b_term = b_params @ cos_coeffs

        return a_term + b_term

    def f(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate the system vector field; subclasses must implement this."""

        raise NotImplementedError

    def diff_squared_norm(self, theta: torch.Tensor, t: torch.Tensor, T: float) -> torch.Tensor:
        """Return the pointwise squared residual ``||dz/dt - f(z)||^2``."""

        zdot = self.dz(theta, t, T)
        fz = self.f(self.z(theta, t, T))
        diff = zdot - fz
        return torch.sum(diff * diff, axis=0)


    def loss(self, theta: torch.Tensor, T: float | torch.Tensor):
        """Integrate the squared orbit residual over one normalized period."""

        frequency_cutoff = (theta.shape[1] - 1) // 2
        integral_num_samples = frequency_cutoff * self.INTEGRAL_NUM_SAMPLES_FACTOR

        T_float: float = T.item() if hasattr(T, 'item') else T

        t_vector = torch.linspace(0, T_float, integral_num_samples + 1)
        diff_vector = self.diff_squared_norm(theta, t_vector, T)
        
        result = torch.trapezoid(diff_vector, x=t_vector)
        return result / T_float
