import torch
from typing import Callable, Optional

class Theta_T_Abstract_Module(torch.nn.Module):
    """Base module for flattening and reconstructing orbit parameters."""
    def __init__(self, theta: torch.Tensor, T: torch.Tensor) -> None:
        super().__init__()
        self.num_variables = theta.shape[0]
        self.frequency_cutoff = (theta.shape[1] - 1) // 2

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    def transform_instance(self, transform: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]], base_instance: 'Optional[Theta_T_Abstract_Module]'=None):
        """Transform ``(theta, T)`` and create a matching module instance.

        Supply ``base_instance`` to select a different output module class.
        """
        transformed = transform(*self.forward())
        if base_instance is None:
            base_instance = self

        return base_instance.create_instance(*transformed)
    
    def create_instance(self, theta: torch.Tensor, T: torch.Tensor):
        """Create an instance of this module class from ``theta`` and ``T``."""
        return type(self)(theta, T)

    def flatten(self, full_theta=False, include_T=True):
        """Flatten trainable (or all) coefficients, optionally prepending ``T``."""
        theta, T = self.forward()
        if full_theta:
            flat_theta = self.flatten_full_theta(theta)
        else:
            flat_theta = self._flatten_theta()

        return torch.cat((T, flat_theta)) if include_T else flat_theta

    def _flatten_theta(self):
        """Flatten only the trainable parameters of theta."""
        raise NotImplementedError

    @classmethod
    def flatten_full_theta(cls, theta: torch.Tensor):
        """Flatten a complete coefficient tensor in variable-major order."""
        return torch.cat(torch.split(theta, 1), 1).reshape(-1)

    @classmethod
    def flatten_full_theta_T(cls, theta: torch.Tensor, T):
        flat_theta = cls.flatten_full_theta(theta)
        return torch.cat((T, flat_theta))

    def repack(self, flat_theta: torch.Tensor, full_theta=False, include_T=True):
        """Reconstruct ``(theta, T)`` from a flattened parameter tensor."""
        T = flat_theta[0] if include_T else None
        flat_theta_mod = flat_theta[1:] if include_T else flat_theta
        
        if full_theta:
            theta = self.repack_full_theta(flat_theta_mod)
        else:
            theta = self._repack_theta(flat_theta_mod)
        
        return theta, T

    def _repack_theta(self, flat_theta: torch.Tensor):
        """Reconstruct theta from the trainable parameters."""
        raise NotImplementedError

    def repack_full_theta(self, flat_theta: torch.Tensor):
        """Reconstruct a complete coefficient tensor for this system size."""
        return self.repack_full_theta_vars(flat_theta, self.num_variables)

    @classmethod
    def repack_full_theta_vars(cls, flat_theta: torch.Tensor, num_variables: int):
        """Reconstruct a complete coefficient tensor with ``num_variables`` rows."""
        return torch.stack(torch.split(flat_theta, flat_theta.shape[0] // num_variables, 0), 0)


class Theta_T_Module(Theta_T_Abstract_Module):
    """Basic implementation of Theta_T_Abstract_Module that holds theta and T."""
    def __init__(self, theta: torch.Tensor, T: torch.Tensor) -> None:
        super().__init__(theta, T)

        self.theta = torch.nn.Parameter(theta)
        self.T = torch.nn.Parameter(T)

    def forward(self):
        return self.theta, self.T

    def _flatten_theta(self):
        return self.flatten_full_theta(self.theta)

    def _repack_theta(self, flat_theta):
        return self.repack_full_theta(flat_theta)


class Theta_Grad_T_Module(Theta_T_Abstract_Module):
    """Store trainable and fixed cosine/sine blocks separately.

    ``grad`` enables each variable's cosine and sine blocks independently.
    Subclasses may override the compression and expansion hooks to impose
    Fourier symmetries.
    """

    def __init__(self, theta: torch.Tensor, T: torch.Tensor, grad: list[tuple[bool, bool]]=None, zero_no_grad=False) -> None:
        super().__init__(theta, T)

        assert not hasattr(super(), '_compress_cosines')
        assert not hasattr(super(), '_compress_sines')
        assert not hasattr(super(), '_expand_cosines')
        assert not hasattr(super(), '_expand_sines')

        self._grad = grad
        self._zero_no_grad = zero_no_grad

        if grad is None:
            grad = [(True, True) for _ in range(self.num_variables)]

        theta_list = [None for _ in range(2 * self.num_variables)]
        grad_list = [False for _ in range(2 * self.num_variables)]

        lengths = {
            True: 0,
            False: 0,
        }

        theta_split = torch.vsplit(theta, self.num_variables)
        for y in range(self.num_variables):
            theta_split_1d = theta_split[y].reshape((2*self.frequency_cutoff+1,))

            cosines = self._compress_cosines(theta_split_1d[:self.frequency_cutoff+1])
            sines = self._compress_sines(theta_split_1d[self.frequency_cutoff+1:])

            if zero_no_grad:
                if not grad[y][0]:
                    cosines = torch.zeros(cosines.shape)
                if not grad[y][1]:
                    sines = torch.zeros(sines.shape)

            theta_list[2 * y] = cosines
            theta_list[2 * y + 1] = sines

            grad_list[2 * y] = grad[y][0]
            grad_list[2 * y + 1] = grad[y][1]

            lengths[grad_list[2 * y]] += cosines.shape[0]
            lengths[grad_list[2 * y + 1]] += sines.shape[0]

        thetas_collected = {
            True: torch.zeros((lengths[True],)),
            False: torch.zeros((lengths[False],)),
        }

        offsets = {
            True: 0,
            False: 0,
        }

        part_mappings: list[tuple[bool, int, int]] = []

        for grad_active, theta_part in zip(grad_list, theta_list):
            start_offset = offsets[grad_active]
            end_offset = start_offset + theta_part.shape[0]
            offsets[grad_active] = end_offset

            thetas_collected[grad_active][start_offset:end_offset] = theta_part

            part_mappings.append((grad_active, start_offset, end_offset))

        self.part_mappings = part_mappings

        self.theta_enabled = torch.nn.Parameter(thetas_collected[True])
        self.theta_disabled = thetas_collected[False]
        self.T = torch.nn.Parameter(T)


    def create_instance(self, theta, T):
        return type(self)(theta, T, self._grad, self._zero_no_grad)


    def _compress_cosines(self, cosines):
        return cosines
    
    def _compress_sines(self, sines):
        return sines
    
    def _expand_cosines(self, cosines):
        return cosines
    
    def _expand_sines(self, sines):
        return sines


    def _get_part(self, index):
        grad_active, start_offset, end_offset = self.part_mappings[index]

        if grad_active:
            return self.theta_enabled[start_offset:end_offset]
        else:
            return self.theta_disabled[start_offset:end_offset]


    def _get_theta_list(self):
        return [self._get_part(i) for i in range(2 * self.num_variables)]


    def _stack_theta(self, theta_list: list[torch.Tensor]):
        return torch.vstack([
            torch.concat((
                self._expand_cosines(theta_list[2 * i]),
                self._expand_sines(theta_list[2 * i + 1]),
            ))
            for i in range(self.num_variables)
        ])

    def forward(self):
        return self._stack_theta(self._get_theta_list()), self.T

    def _flatten_theta(self):
        return self.theta_enabled


    def _repack_theta(self, flat_theta: torch.Tensor):
        theta_parts = []
        for index in range(2 * self.num_variables):
            grad_active, start_offset, end_offset = self.part_mappings[index]

            if grad_active:
                theta_parts.append(flat_theta[start_offset:end_offset])
            else:
                theta_parts.append(self.theta_disabled[start_offset:end_offset])

        return self._stack_theta(theta_parts)


class Odd_Grad_T_Module(Theta_Grad_T_Module):
    """Retain only odd-frequency cosine and sine modes."""

    def _compress_cosines(self, cosines):
        cosines = super()._compress_cosines(cosines)
        return cosines[1::2]
    
    def _compress_sines(self, sines):
        sines = super()._compress_sines(sines)
        return sines[::2]
    
    def _expand_cosines(self, cosines):
        cosines = super()._expand_cosines(cosines)

        cosines_expanded = torch.zeros((self.frequency_cutoff + 1,))
        cosines_expanded[1::2] = cosines
        return cosines_expanded
    
    def _expand_sines(self, sines):
        sines = super()._expand_sines(sines)
        
        sines_expanded = torch.zeros((self.frequency_cutoff,))
        sines_expanded[::2] = sines
        return sines_expanded


class Const_Prepend_T_Module(Theta_Grad_T_Module):
    """Retain the constant mode in a cooperatively compressed module.

    This mixin is intended for multiple inheritance with another compression
    module and cannot be instantiated directly.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert not type(self).mro()[1] == Theta_Grad_T_Module, "cannot instantiate Const_Prepend_T_Module"

    def _compress_cosines(self, cosines):
        cosines_super = super()._compress_cosines(cosines)
        return torch.cat((cosines[0:1], cosines_super))
    
    def _compress_sines(self, sines):
        sines_super = super()._compress_sines(sines)
        return sines_super
    
    def _expand_cosines(self, cosines):
        cosines_super = super()._expand_cosines(cosines[1:])

        cosines_super[0] = cosines[0]
        return cosines_super
    
    def _expand_sines(self, sines):
        sines_super = super()._expand_sines(sines)

        return sines_super


class Normalize_T_Module(Theta_Grad_T_Module):
    """Train unit-scaled parameters multiplied by their initial factors."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.initial_factors = self.theta_enabled.data
        self.theta_enabled = torch.nn.Parameter(torch.ones(self.initial_factors.shape))
    

    def _get_part(self, index):
        grad_active, start_offset, end_offset = self.part_mappings[index]

        if grad_active:
            return self.theta_enabled[start_offset:end_offset] * self.initial_factors[start_offset:end_offset]
        else:
            return self.theta_disabled[start_offset:end_offset]

    def _flatten_theta(self):
        return self.theta_enabled * self.initial_factors


class Const_Odd_Grad_T_Module(Const_Prepend_T_Module, Odd_Grad_T_Module):
    """Odd-frequency coefficients with the constant cosine mode retained."""


class Norm_Odd_Grad_T_Module(Normalize_T_Module, Odd_Grad_T_Module):
    """Unit-scaled form of :class:`Odd_Grad_T_Module`."""


class Norm_Const_Odd_Grad_T_Module(Normalize_T_Module, Const_Prepend_T_Module, Odd_Grad_T_Module):
    """Unit-scaled form of :class:`Const_Odd_Grad_T_Module`."""
