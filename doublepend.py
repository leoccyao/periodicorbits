import torch
from orbitlib.reg.regularizer import Regularizer
from orbitlib import Orbit_Interface

class Double_Pendulum_Orbit(Orbit_Interface):
    """Dimensionless planar double-pendulum vector field and observables."""

    num_variables = 4

    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=1.0) -> None:
        super().__init__()
        self.m1, self.m2, self.l1, self.l2, self.g = m1, m2, l1, l2, g

    def f(self, z):
        """Evaluate ``(theta1_dot, theta2_dot, theta1_ddot, theta2_ddot)``."""
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        sin, cos = torch.sin, torch.cos

        t1 = z[0, :]
        t2 = z[1, :]
        t1dot = z[2, :]
        t2dot = z[3, :]

        dt = t1 - t2

        t1ddot = (
            (
                - g * (m1 + m2/2) * sin(t1)
                - g * m2 * sin(t1 - 2 * t2)/2
                - l1 * m2 * t1dot ** 2 * sin(2 * dt)/2
                - l2 * m2 * t2dot ** 2 * sin(dt)
            )/l1
        )

        t2ddot = (
            (
                - g * (m1 + m2) * sin(t2)/2
                + g * (m1 + m2) * sin(2 * t1 - t2)/2
                + l1 * (m1 + m2) * t1dot ** 2 * sin(dt)
                + l2 * m2 * t2dot ** 2 * sin(2 * dt)/2
            )/l2
        )

        denom = (m1 - m2 * cos(dt) ** 2 + m2)
        t1ddot /= denom
        t2ddot /= denom

        return torch.vstack((t1dot, t2dot, t1ddot, t2ddot))

    def x_y_coordinates(self, z):
        """Map state samples to ``(x1, y1, x2, y2)`` Cartesian coordinates."""
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        sin, cos = torch.sin, torch.cos

        t1 = z[0, :]
        t2 = z[1, :]

        x1 = l1 * sin(t1)
        y1 = -l1 * cos(t1)
        x2 = l2 * sin(t2) + x1
        y2 = -l2 * cos(t2) + y1

        return torch.vstack((x1, y1, x2, y2))

    def energy(self, z):
        """Return total mechanical energy for each sampled state."""
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        sin, cos = torch.sin, torch.cos

        t1 = z[0, :]
        t2 = z[1, :]
        t1dot = z[2, :]
        t2dot = z[3, :]

        dt = t1 - t2

        U1 = - m1 * g * (l1 * cos(t1))
        U2 = - m2 * g * (l1 * cos(t1) + l2 * cos(t2))

        v1 = l1 * t1dot
        v2 = l2 * t2dot

        T1 = m1 * v1 ** 2 / 2
        T2 = m2 * (v1 ** 2 + v2 ** 2 + 2 * v1 * v2 * cos(dt)) / 2

        return U1 + U2 + T1 + T2
    

Double_Pendulum_Regularizer = Regularizer[Double_Pendulum_Orbit]

class Energy_Regularizer(Double_Pendulum_Regularizer):
    """Quadratically penalize deviation from a target energy."""
    def __init__(self, system, target_energy: float, weight: float) -> None:
        super().__init__(system)
        self.target_energy = target_energy
        self.weight = weight
    
    def __call__(self, theta, T) -> torch.Tensor:
        energy = self.system.energy(self.system.z0(theta))
        return self.weight * (energy - self.target_energy) ** 2


class Upper_Angle_Regularizer(Double_Pendulum_Regularizer):
    """Quadratically penalize the initial upper-arm angle."""
    def __init__(self, system, target_angle: float, weight: float) -> None:
        super().__init__(system)
        self.target_angle = target_angle
        self.weight = weight
    
    def __call__(self, theta, T) -> torch.Tensor:
        upper_angle = self.system.z0(theta)[0, 0]
        return self.weight * (upper_angle - self.target_angle) ** 2


class Lower_Angle_Regularizer(Double_Pendulum_Regularizer):
    """Quadratically penalize the initial lower-arm angle."""
    def __init__(self, system, target_angle: float, weight: float) -> None:
        super().__init__(system)
        self.target_angle = target_angle
        self.weight = weight
    
    def __call__(self, theta, T) -> torch.Tensor:
        lower_angle = self.system.z0(theta)[1, 0]
        return self.weight * (lower_angle - self.target_angle) ** 2
