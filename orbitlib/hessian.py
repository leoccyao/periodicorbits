import torch
from .orbit import Orbit_Interface
from .module.module import Theta_T_Abstract_Module


def hessian(system: Orbit_Interface, model: Theta_T_Abstract_Module, full_theta=False, include_T=True, create_graph=False):
    """Compute the orbit-loss Hessian in the model's flattened coordinates.

    ``include_T`` includes period as the leading coordinate; ``full_theta``
    includes fixed as well as trainable Fourier coefficients.
    """
    def repack(flat_theta):
        theta, T = model.repack(flat_theta, full_theta, include_T)
        if T is None:
            T = model.forward()[1].item()
        
        return theta, T


    loss = lambda flat_theta: system.loss(*repack(flat_theta))

    return torch.autograd.functional.hessian(loss, model.flatten(full_theta, include_T), create_graph)


def conditions_from_hess(hess: torch.Tensor, model: Theta_T_Abstract_Module, full_theta=False, include_T=True, num_eig=None):
    """Diagonalize a Hessian and repack its eigenvectors as ``(theta, T)``."""
    eigv, vr = torch.linalg.eigh(hess)

    return eigv[:num_eig], [model.repack(eig, full_theta, include_T) for eig in vr.T[:num_eig]]


def conditions_theta(system: Orbit_Interface, model: Theta_T_Abstract_Module, full_theta=False, include_T=True, num_eig=None, create_graph=False):
    """Compute Hessian eigenvalues and coefficient-space conditions."""
    hess = hessian(system, model, full_theta, include_T, create_graph)
    return conditions_from_hess(hess, model, full_theta, include_T, num_eig)


def phase_from_conditions_theta(system: Orbit_Interface, eigv, eigs_packed):
    """Map coefficient-space conditions to their initial-state variations."""
    return eigv, [(system.z0(theta), T) for theta, T in eigs_packed]


def conditions_phase(system: Orbit_Interface, model: Theta_T_Abstract_Module, full_theta=False, include_T=True, num_eig=None, create_graph=False):
    """Compute Hessian eigenvalues and initial-state-space conditions."""

    eigv, eigs_packed = conditions_theta(system, model, full_theta, include_T, num_eig, create_graph)
    return phase_from_conditions_theta(system, eigv, eigs_packed)
