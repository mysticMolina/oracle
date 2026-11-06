"""Jacob Matthew Molina. Inception: 2026-11-05.

First source-bound 2D ORACLE port slice: the spinor representation,
public-to-solver packing, and exact rest-energy map. The MATLAB public
array is [Ny,Nx,Nz,component]; its packed solver rows put x fastest.
These functions use an explicit array namespace so NumPy can establish
parity before a GPU backend is selected and measured.
"""

import numpy as np


def dirac_matrices(xp=np):
    """Return alpha_x/y/z and beta in MATLAB's standard Dirac basis."""
    sx = xp.asarray([[0, 1], [1, 0]], dtype=xp.complex128)
    sy = xp.asarray([[0, -1j], [1j, 0]], dtype=xp.complex128)
    sz = xp.asarray([[1, 0], [0, -1]], dtype=xp.complex128)
    zero = xp.zeros((2, 2), dtype=xp.complex128)
    alpha = tuple(xp.block([[zero, s], [s, zero]]) for s in (sx, sy, sz))
    beta = xp.diag(xp.asarray([1, 1, -1, -1], dtype=xp.complex128))
    return alpha, beta


def pack_field(field, xp=np):
    """Map [Ny,Nx,Nz,C] to [Nx*Ny*Nz,C], matching MATLAB permute+reshape."""
    a = xp.asarray(field)
    if a.ndim != 4:
        raise ValueError("field must have [Ny,Nx,Nz,component] axes")
    return a.transpose(1, 0, 2, 3).reshape((-1, a.shape[3]), order="F")


def unpack_field(packed, grid, xp=np):
    """Invert pack_field for grid=(Nx,Ny,Nz), without conjugating phases."""
    a = xp.asarray(packed)
    nx, ny, nz = map(int, grid)
    if a.ndim != 2 or nx < 1 or ny < 1 or nz < 1 or a.shape[0] != nx * ny * nz:
        raise ValueError("packed rows do not match the positive grid dimensions")
    return a.reshape((nx, ny, nz, a.shape[1]), order="F").transpose(1, 0, 2, 3)


def rest_energy_step(psi, tau, mass, c, hbar, xp=np):
    """Apply exp[-i beta m c^2 tau/hbar] to the last spinor axis."""
    a = xp.asarray(psi)
    if a.shape[-1] != 4 or mass <= 0 or c <= 0 or hbar <= 0:
        raise ValueError("expected a four-component spinor and positive physical constants")
    signs = xp.asarray([1, 1, -1, -1], dtype=xp.float64)
    phase = xp.exp((-1j * mass * c * c * tau / hbar) * signs)
    return a * phase
