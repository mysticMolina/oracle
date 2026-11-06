"""Jacob Matthew Molina. Inception: 2026-11-05.

CPU reference for MATLAB diracFullMatterStep's upper/lower Schur solve.
K is the upper-row/lower-column block of the spatial Dirac Hamiltonian on free
grid points. This module ports the algebraic Cayley map and its original
block residual. Spatial assembly lives in operators.py; midpoint current
and Maxwell feedback are exposed by full_matter_step below.
"""

import numpy as np
from scipy import sparse

from .linear import LinearPolicy, schur_operator, solve_system
from .core import dirac_matrices
from .operators import free_dirac_block


def _schur_step(psi, free, K, tau, mass, c, hbar, tolerance, solver_policy=None, solve_log=None):
    """Advance [M,4] spinors through the shifted full-Dirac Cayley map.

    The free rows use MATLAB column-major component stacking. Frozen rows
    receive only the restored global mass phase, matching the source.
    Returns physical spinor, pre-phase rotated spinor, and block residual.
    """
    a = np.asarray(psi, dtype=np.complex128)
    f = np.asarray(free, dtype=bool)
    if a.ndim != 2 or a.shape[1] != 4 or f.shape != (a.shape[0],):
        raise ValueError("psi must be [M,4] and free must be a length-M mask")
    if tau == 0 or mass <= 0 or c <= 0 or hbar <= 0 or tolerance <= 0:
        raise ValueError("invalid physical or solver parameter")
    nf = int(f.sum())
    if nf == 0 or K.shape != (2 * nf, 2 * nf):
        raise ValueError("K must be [2*number_free,2*number_free]")
    hop = K if sparse.issparse(K) else np.asarray(K, dtype=np.complex128)
    adj = hop.conj().T
    u = a[f, :2].reshape(-1, order="F")
    v = a[f, 2:].reshape(-1, order="F")
    scale = tau / (2 * hbar)
    rest = mass * c * c
    d = 1 - 2j * scale * rest
    ru = u - 1j * scale * (hop @ v)
    rv = v - 1j * scale * (adj @ u - 2 * rest * v)
    rhs = ru - 1j * scale * (hop @ rv) / d
    def explicit_left():
        identity = (sparse.eye(2*nf, format="csr", dtype=np.complex128)
                    if sparse.issparse(hop) else np.eye(2*nf, dtype=np.complex128))
        return identity + (scale**2/d)*(hop @ adj)

    def original_residual(upper):
        lower = (rv - 1j*scale*(adj @ upper))/d
        residual = np.linalg.norm(np.r_[upper + 1j*scale*(hop @ lower) - ru,
                                        d*lower + 1j*scale*(adj @ upper) - rv])
        return float(residual/max(np.linalg.norm(np.r_[ru, rv]), np.finfo(float).tiny))

    policy = LinearPolicy() if solver_policy is None else solver_policy
    operator = schur_operator(hop, scale**2/d)
    un, report = solve_system(operator, rhs, tolerance, policy, direct_matrix=explicit_left,
                             acceptance=original_residual, matrix_free=True, solve_log=solve_log)
    vn = (rv - 1j*scale*(adj @ un))/d
    residual = report.original_residual
    rotated = a.copy()
    rotated[f, :2] = un.reshape((nf, 2), order="F")
    rotated[f, 2:] = vn.reshape((nf, 2), order="F")
    return rotated * np.exp(-1j * tau * rest / hbar), rotated, float(residual)


def full_cayley_schur_step(psi, free, K, tau, mass, c, hbar, tolerance=1e-10, *,
                           solver_policy=None, solve_log=None):
    """Advance matter and return physical spinor plus original-block residual."""
    physical, _, residual = _schur_step(psi, free, K, tau, mass, c, hbar, tolerance, solver_policy, solve_log)
    return physical, residual


def dirac_current(psi, charge, c):
    """Collocated real q*c*psi^dagger*alpha*psi for one [M,4] spinor."""
    a = np.asarray(psi, dtype=complex)
    if a.ndim != 2 or a.shape[1] != 4 or not np.all(np.isfinite(a)):
        raise ValueError("psi must be a finite [M,4] spinor")
    if not np.isfinite(charge) or not np.isfinite(c) or c <= 0:
        raise ValueError("charge must be finite and c positive")
    alpha, _ = dirac_matrices()
    return charge * c * np.column_stack([
        np.real(np.sum(a.conj() * (a @ matrix.T), axis=1)) for matrix in alpha])


def charge_density(psi, charge, background_charge=None):
    """Signed single-species nodal charge, without an implicit neutralizer."""
    a = np.asarray(psi, dtype=complex)
    if a.ndim != 2 or a.shape[1] != 4 or not np.all(np.isfinite(a)) or not np.isfinite(charge):
        raise ValueError("expected finite [M,4] psi and charge")
    rho = charge * np.sum(np.abs(a)**2, axis=1)
    if background_charge is not None:
        background = np.asarray(background_charge)
        if background.shape != rho.shape or not np.isrealobj(background) or not np.all(np.isfinite(background)):
            raise ValueError("background charge must be a finite real length-M array")
        rho = rho + background
    return rho


def full_matter_step(psi, vector_potential, electric_momentum, operators,
                     tau, charge, mass, c, hbar, feedback=False, tolerance=1e-10, *,
                     solver_policy=None, solve_log=None):
    """Source full-Cayley map, rotated-midpoint current and continuity screen.

    The current uses the midpoint BEFORE restoring the global rest phase.
    The continuity screen uses the centered source divergence; it is not
    automatically zero, nor the adjoint forward Maxwell Gauss operator.
    """
    y = np.asarray(electric_momentum)
    if y.shape != (len(operators.free), 3) or not np.isrealobj(y) or not np.all(np.isfinite(y)):
        raise ValueError("electric_momentum must be finite real [M,3]")
    k = free_dirac_block(vector_potential, operators, charge, c, hbar)
    physical, rotated, residual = _schur_step(psi, operators.free, k, tau, mass, c, hbar, tolerance, solver_policy, solve_log)
    midpoint = (np.asarray(psi) + rotated) / 2
    current = dirac_current(midpoint, charge, c)
    continuity = charge * (np.sum(np.abs(rotated)**2, axis=1)
                            - np.sum(np.abs(psi)**2, axis=1)) / tau
    continuity += operators.divergence @ current.ravel(order="F")
    next_y = y + tau * current * operators.free[:, None] if feedback else y.copy()
    return physical, next_y, {"residual": residual, "current": current,
                              "continuity": continuity}
