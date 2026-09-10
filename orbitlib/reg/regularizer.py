import torch
from typing import TypeVar, Generic
from ..orbit import Orbit_Interface

System = TypeVar('System', bound=Orbit_Interface, covariant=True)


class Regularizer(Generic[System]):
    """Base callable interface for system-specific loss regularizers."""
    def __init__(self, system: System) -> None:
        self.system = system

    def __call__(self, theta, T) -> torch.Tensor:
        """Return the scalar regularization for model output ``(theta, T)``."""
        raise NotImplementedError


class T_Regularizer(Regularizer):
    """Quadratically penalize deviation from a target period."""
    def __init__(self, system, target_T: float, weight: float) -> None:
        super().__init__(system)
        self.target_T = target_T
        self.weight = weight
    
    def __call__(self, theta, T):
        return self.weight * (T - self.target_T) ** 2
