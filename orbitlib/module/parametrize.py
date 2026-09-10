import torch
import numpy as np
from .module import Normalize_T_Module, Theta_Grad_T_Module


class Hyperbolic_Param(torch.nn.Module):
    """Map unconstrained parameters through a scaled hyperbolic sine.

    ``prefactor`` selects the magnitude range over which the map is
    approximately linear.
    """

    def __init__(self, prefactor: float=1) -> None:
        super().__init__()
        self.prefactor = prefactor


    def forward(self, theta: torch.Tensor):
        return self.prefactor * torch.sinh(theta)

    
    def right_inverse(self, theta: torch.Tensor):
        return torch.asinh(theta / self.prefactor)


class Weighted_Multi_Theta_Sum_Param(torch.nn.Module):
    """Constrain weighted sums across trainable coefficient blocks.

    Each constraint is ``(variables, value)``, where each variable entry is
    ``(variable_index, cosine_or_sine, weight)``. A block may appear in at most
    one constraint.
    """
    def __init__(self, module: Theta_Grad_T_Module, constraint_map: list[tuple[list[tuple[int, int, float]], float]], strict_grad=True, add_mode=False) -> None:
        super().__init__()

        self.add_mode = add_mode

        constraint_intervals: list[tuple[float, list[tuple[float, int, int]]]] = []

        constrained_variables: list[bool] = [False for _ in range(2 * module.num_variables)]

        for variables, constraint_value in constraint_map:
            offsets = []

            for var_index, cos_sin, weight in variables:
                if weight != 0:
                    index = 2 * var_index + cos_sin
                    
                    if constrained_variables[index]:
                        raise Exception('Variable already constrained')

                    constrained_variables[index] = True

                    grad_active, start_offset, end_offset = module.part_mappings[index]

                    if not grad_active:
                        if strict_grad:
                            raise Exception('Attempting to constrain an inactive gradient')
                        else:
                            continue

                    offsets.append((weight, start_offset, end_offset))
            
            constraint_intervals.append((constraint_value, offsets))

        self.part_mappings = module.part_mappings
        self.constraint_intervals = constraint_intervals
    
    @staticmethod
    def get_constraint_map_from_arr(constraints):
        """Convert per-variable cosine/sine constraints to a constraint map."""
        constraint_map = []

        for variable_index, constraint_pair in enumerate(constraints):
            for index, value in enumerate(constraint_pair):
                if value is not None:
                    constraint_map.append(([(variable_index, index, 1)], value))

        return constraint_map


    def forward(self, theta_enabled: torch.Tensor):
        return (
            self._forward_add_mode(theta_enabled) if self.add_mode
            else self._forward_mult_mode(theta_enabled)
        )


    def _forward_mult_mode(self, theta_enabled: torch.Tensor):
        theta_constrained = theta_enabled.clone()

        for constraint_value, offsets_list in self.constraint_intervals:
            constraint_sum = 0

            for weight, start_offset, end_offset in offsets_list:
                constraint_sum += weight * theta_constrained[start_offset:end_offset].sum()

            for weight, start_offset, end_offset in offsets_list:
                theta_constrained[start_offset:end_offset] *= constraint_value / constraint_sum

        return theta_constrained


    def _forward_add_mode(self, theta_enabled: torch.Tensor):
        theta_constrained = theta_enabled.clone()

        for constraint_value, offsets_list in self.constraint_intervals:
            constraint_sum = 0

            max_weight_index = np.argmax(np.abs([w for w, _, _ in offsets_list]))

            for weight, start_offset, end_offset in offsets_list:
                constraint_sum += weight * theta_constrained[start_offset:end_offset].sum()

            weight, start_offset, end_offset = offsets_list[max_weight_index]

            constraint_sum -= weight * theta_constrained[start_offset]
            theta_constrained[start_offset] = (constraint_value - constraint_sum) / weight

        return theta_constrained


class Factor_Weighted_Multi_Theta_Sum_Param(Weighted_Multi_Theta_Sum_Param):
    """Weighted-sum constraints for :class:`Normalize_T_Module`."""

    def __init__(self, module: Normalize_T_Module, constraint_map: list[tuple[list[tuple[int, int, float]], float]], strict_grad=True, add_mode=False) -> None:
        super().__init__(module, constraint_map, strict_grad, add_mode)

        self.initial_factors = module.initial_factors

    def _forward_mult_mode(self, theta_enabled: torch.Tensor):
        theta_constrained = theta_enabled.clone()

        for constraint_value, offsets_list in self.constraint_intervals:
            constraint_sum = 0

            for weight, start_offset, end_offset in offsets_list:
                constraint_sum += weight * (theta_constrained[start_offset:end_offset] * self.initial_factors[start_offset:end_offset]).sum()

            for weight, start_offset, end_offset in offsets_list:
                theta_constrained[start_offset:end_offset] *= constraint_value / constraint_sum

        return theta_constrained


    def _forward_add_mode(self, theta_enabled: torch.Tensor):
        raise NotImplementedError('Add mode is not currently supported')
