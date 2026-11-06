"""Jacob Matthew Molina. Inception: 2026-11-05.

ORACLE's two distinct spatial discretizations in x-fastest ordering.
Matter uses centered differences with frozen rows. Compatible Maxwell
uses forward differences and an adjoint electric divergence. These are
source parity operators, not a new choice of stencil or boundary closure.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .core import dirac_matrices


@dataclass(frozen=True)
class CenteredOperators:
    derivatives: tuple
    gradient: sparse.csr_matrix
    divergence: sparse.csr_matrix
    curl: sparse.csr_matrix
    free: np.ndarray


@dataclass(frozen=True)
class CompatibleOperators:
    gradient: sparse.csr_matrix
    curl: sparse.csr_matrix
    magnetic_divergence: sparse.csr_matrix
    free: np.ndarray
    free_curl: sparse.csr_matrix
    curl_curl: sparse.csr_matrix
    free_gradient: sparse.csr_matrix


def _geometry(grid, spacing):
    n = np.asarray(grid)
    h = np.asarray(spacing)
    if n.shape != (3,) or h.shape != (3,) or not np.isrealobj(n) or not np.isrealobj(h):
        raise ValueError("grid and spacing must have three real entries")
    if not np.all(np.isfinite(n)) or not np.all(n == np.floor(n)):
        raise ValueError("grid entries must be finite integers")
    if not np.all((n == 1) | (n >= 3)) or not np.any(n > 1):
        raise ValueError("active axes require at least three nodes")
    if not np.all(np.isfinite(h)) or not np.all(h > 0):
        raise ValueError("spacing must be finite and positive")
    return tuple(n.astype(int)), h.astype(float)


def _tensor(z, y, x):
    return sparse.kron(z, sparse.kron(y, x, format="csr"), format="csr")


def _curl(dx, dy, dz):
    zero = sparse.csr_matrix(dx.shape)
    return sparse.bmat([[zero, -dz, dy], [dz, zero, -dx],
                        [-dy, dx, zero]], format="csr")


def centered_operators(grid, spacing):
    """Port diracMaxwellDifferentialOperators, including boundary columns.

    A singleton coordinate is inactive and contributes no wall. A boundary
    row is zero in every derivative; its column still feeds its neighbor.
    Consequently only the free-free derivative block is skew-adjoint.
    """
    n, h = _geometry(grid, spacing)
    d, p = [], []
    for size, step in zip(n, h):
        if size == 1:
            d.append(sparse.csr_matrix((1, 1)))
            p.append(sparse.eye(1, format="csr"))
        else:
            mask = np.ones(size)
            mask[[0, -1]] = 0
            project = sparse.diags(mask, format="csr")
            diff = sparse.diags([-np.ones(size - 1), np.ones(size - 1)],
                                [-1, 1], shape=(size, size), format="csr") / (2 * step)
            d.append(project @ diff)
            p.append(project)
    dx = _tensor(p[2], p[1], d[0])
    dy = _tensor(p[2], d[1], p[0])
    dz = _tensor(d[2], p[1], p[0])
    free = _tensor(p[2], p[1], p[0]).diagonal().astype(bool)
    return CenteredOperators((dx, dy, dz), sparse.vstack([dx, dy, dz], format="csr"),
                             sparse.hstack([dx, dy, dz], format="csr"),
                             _curl(dx, dy, dz), free)


def compatible_operators(grid, spacing):
    """Port diracCompatibleMaxwell geometry; no solve or Mur closure yet.

    Electric divergence is -gradient.T, while magnetic_divergence uses
    forward derivatives. The vector free mask is the source PEC mask,
    not the Mur mask. Mur must later replace the field boundary evolution.
    """
    n, h = _geometry(grid, spacing)
    f, identity = [], []
    for size, step in zip(n, h):
        diagonal = -np.ones(size)
        diagonal[-1] = 0
        f.append(sparse.diags([diagonal, np.ones(size - 1)], [0, 1],
                             shape=(size, size), format="csr") / step)
        identity.append(sparse.eye(size, format="csr"))
    dx = _tensor(identity[2], identity[1], f[0])
    dy = _tensor(identity[2], f[1], identity[0])
    dz = _tensor(f[2], identity[1], identity[0])
    coordinates = np.indices(n).reshape((3, -1), order="F")
    face = [(coordinates[k] == 0) | (coordinates[k] == size - 1)
            if size > 1 else np.zeros(np.prod(n), dtype=bool)
            for k, size in enumerate(n)]
    free = np.r_[~(face[1] | face[2]), ~(face[0] | face[2]), ~(face[0] | face[1])]
    gradient = sparse.vstack([dx, dy, dz], format="csr")
    curl = _curl(dx, dy, dz)
    cf = curl[:, free]
    scalar_free = ~(face[0] | face[1] | face[2])
    return CompatibleOperators(gradient, curl, sparse.hstack([dx, dy, dz], format="csr"),
                               free, cf, (cf.T @ cf).tocsr(), gradient[:, scalar_free])


def spatial_hamiltonian(vector_potential, operators, charge, c, hbar):
    """Assemble c alpha dot (-i hbar grad - q A), without mass.

    Input A is [M,3]; spinor blocks are component-major. Real A makes
    the free-free block Hermitian. Frozen boundary rows are exactly zero.
    """
    a = np.asarray(vector_potential)
    m = len(operators.free)
    if a.shape != (m, 3) or not np.isrealobj(a) or not np.all(np.isfinite(a)):
        raise ValueError("vector_potential must be a finite real [M,3] array")
    if not np.isfinite(charge) or not np.isfinite(c) or not np.isfinite(hbar) or c <= 0 or hbar <= 0:
        raise ValueError("charge must be finite; c and hbar must be finite and positive")
    alpha, _ = dirac_matrices()
    result = sparse.csr_matrix((4 * m, 4 * m), dtype=complex)
    for k in range(3):
        momentum = -1j * hbar * operators.derivatives[k]
        momentum = momentum - charge * sparse.diags(a[:, k] * operators.free, format="csr")
        result = result + c * sparse.kron(alpha[k], momentum, format="csr")
    return result


def free_dirac_block(vector_potential, operators, charge, c, hbar):
    """Return the upper/lower K block consumed by full_cayley_schur_step."""
    h = spatial_hamiltonian(vector_potential, operators, charge, c, hbar)
    m = len(operators.free)
    indices = np.flatnonzero(operators.free)
    upper = np.r_[indices, indices + m]
    return h[upper, :][:, upper + 2 * m].tocsr()
