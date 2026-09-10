import numpy as np
import scipy.optimize

def time_evolution(v1, v2, T1, T2):
    """Return the zero-period-component combination of two conditions."""
    return v1 * T2 - v2 * T1


def perpendicularize(v1, v2, T1, T2):
    """Remove the time-evolution component from a symmetric combination."""
    v = time_evolution(v1, v2, T1, T2)
    vperp = v1 * T1 + v2 * T2
    vperp -= (vperp.T @ v) * v / np.linalg.norm(v) ** 2
    return vperp


def T_propagation(v1, v2, T1, T2, vperp):
    """Recover the period component associated with ``vperp``.

    The input vectors need not be perpendicular. The two-dimensional leading
    coordinate block is used to solve for their linear-combination weights.
    """
    A = np.concatenate([v1[0:2, :], v2[0:2, :]], axis=1)
    b = vperp[0:2, 0]
    sol = np.linalg.solve(A, b)
    return sol[0] * T1 + sol[1] * T2


def propagation(v1, v2, T1, T2):
    """Produce the propagation vector and T component."""
    vperp = perpendicularize(v1, v2, T1, T2)
    vperp_norm = vperp / np.linalg.norm(vperp)
    return vperp_norm, T_propagation(v1, v2, T1, T2, vperp_norm)


def time_evolution_theta(t1, t2, T1, T2):
    """Return the zero-period-component combination in coefficient space."""
    return t1 * T2 - t2 * T1


def perpendicularize_theta(t1, t2, T1, T2):
    """Remove the time-evolution component in coefficient space."""
    t = time_evolution_theta(t1, t2, T1, T2)
    vperp = t1 * T1 + t2 * T2
    vperp -= (vperp.ravel() @ t.ravel()) * t / np.linalg.norm(t.ravel()) ** 2
    return vperp


def T_propagation_theta(t1, t2, T1, T2, tperp):
    """Recover the period component associated with ``tperp``.

    A least-squares solve obtains the combination weights because the input
    coefficient vectors need not be perpendicular.
    """
    def diff_func(a):
        diff_vec = a[0] * t1 + a[1] * t2 - tperp
        return np.linalg.norm(diff_vec.ravel()) ** 2

    result = scipy.optimize.least_squares(diff_func, x0=np.ones(2))
    x = result.x

    return x[0] * T1 + x[1] * T2


def propagation_theta(t1, t2, T1, T2):
    """Produce the normalized propagation theta vector and T component."""
    tperp = perpendicularize_theta(t1, t2, T1, T2)
    tperp_norm = tperp / np.linalg.norm(tperp.ravel())
    return tperp_norm, T_propagation_theta(t1, t2, T1, T2, tperp_norm)
