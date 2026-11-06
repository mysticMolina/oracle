"""Jacob Matthew Molina. Inception: 2026-11-05.

CPU planar Mur-1 and longitudinal reference, translated from ORACLE's
compatible-forward MATLAB path. Y is canonical electric momentum, not E.
The initial harmonic gradient remains fixed while grounded charge follows
rho(t). Mur-2, pair occupations and GPU solvers are not implemented here.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .operators import centered_operators, compatible_operators
from .linear import LinearPolicy, mur_operator, solve_system


@dataclass(frozen=True)
class MurHistory:
    initial_longitudinal_y: np.ndarray
    initial_harmonic_y: np.ndarray
    previous_trace_y: np.ndarray | None = None


@dataclass(frozen=True)
class MurContext:
    grid: tuple
    spacing: np.ndarray
    dt: float
    c: float
    eps0: float
    tolerance: float
    closure: str
    volume: float
    matter: object
    field: object
    rows: np.ndarray
    adjacent: np.ndarray
    inactive: np.ndarray
    nu: np.ndarray
    r: np.ndarray
    trace: sparse.csr_matrix
    stiffness: sparse.csr_matrix
    left: sparse.csr_matrix
    solve: object
    poisson_solve: object
    fft_size: tuple
    green_fft: np.ndarray
    solver_policy: LinearPolicy


def mur_context(grid, spacing, dt, c, eps0=1., longitudinal="grounded", tolerance=1e-10, *,
                solver_policy=None):
    """Build the source planar Mur-1 rows, midpoint solve and Poisson map.

    Upper padded forward edges are inactive. At tangential corners the
    inward neighbor crosses each incident face and uses the diagonal h.
    Only Nx,Ny>=3 and Nz=1 are enabled in this validated port stage.
    """
    policy = LinearPolicy() if solver_policy is None else solver_policy
    if not isinstance(policy, LinearPolicy):
        raise ValueError("solver_policy must be LinearPolicy")
    matter = centered_operators(grid, spacing)
    n = tuple(int(k) for k in grid)
    h = np.asarray(spacing, dtype=float)
    if n[0] < 3 or n[1] < 3 or n[2] != 1:
        raise ValueError("this Mur port supports the xy plane with Nx,Ny>=3, Nz=1")
    if not np.all(np.isfinite([dt, c, eps0, tolerance])) or min(dt, c, eps0, tolerance) <= 0:
        raise ValueError("dt, c, eps0 and tolerance must be finite and positive")
    if np.max(c * dt / h[:2]) > 1 + 1e-12:
        raise ValueError("Mur-1 Courant number exceeds one")
    if longitudinal not in ("grounded", "free-space"):
        raise ValueError("longitudinal must be grounded or free-space")
    field = compatible_operators(n, h)
    m = int(np.prod(n))
    coordinates = np.indices(n).reshape((3, -1), order="F").T
    face = np.zeros((m, 3), dtype=bool)
    positive = np.zeros_like(face)
    for axis in range(2):
        face[:, axis] = (coordinates[:, axis] == 0) | (coordinates[:, axis] == n[axis] - 1)
        positive[:, axis] = coordinates[:, axis] == n[axis] - 1
    inactive = positive.ravel(order="F")
    rows = np.flatnonzero(~field.free & ~inactive)
    adjacent = np.zeros_like(rows)
    nu = np.zeros(len(rows))
    stride = np.array([1, n[0], n[0] * n[1]])
    for j, row in enumerate(rows):
        node, component = row % m, row // m
        normal = face[node].copy()
        normal[component] = False
        if not np.any(normal):
            raise ValueError("Mur row lacks a tangential face")
        direction = np.where(positive[node], -1, 1)
        adjacent[j] = row + np.sum(stride[normal] * direction[normal])
        nu[j] = c * dt / np.linalg.norm(h[normal])
    r = (nu - 1) / (nu + 1)
    index = np.arange(len(rows))
    trace = sparse.csr_matrix((np.r_[np.ones(len(rows)), -r],
                               (np.r_[index, index], np.r_[rows, adjacent])), shape=(len(rows), 3 * m))
    stiffness = (field.curl.T @ field.curl).tocsr()
    left = (sparse.eye(3 * m, format="csr") + dt**2 * c**2 * stiffness / 4).tolil()
    left[rows, :] = trace
    frozen = np.flatnonzero(inactive)
    left[frozen, :] = sparse.csr_matrix((np.ones(len(frozen)), (np.arange(len(frozen)), frozen)),
                                       shape=(len(frozen), 3 * m))
    left = left.tocsr()
    poisson = (field.free_gradient.T @ field.free_gradient).tocsc()
    offsets = [np.arange(-(size - 1), size) * step for size, step in zip(n, h)]
    x, y, z = np.meshgrid(*offsets, indexing="ij")
    radius = np.sqrt(x*x + y*y + z*z)
    volume = float(np.prod(h[:2]))
    radius[radius == 0] = np.sqrt(volume)
    kernel = -np.log(radius / np.sqrt(volume)) * volume / (2 * np.pi)
    fft_size = tuple(int(2**np.ceil(np.log2(3 * size - 2))) for size in n)
    green_fft = np.fft.fftn(kernel, s=fft_size, axes=(0, 1, 2))
    overlap = field.free_gradient[rows, :]
    if overlap.nnz and np.linalg.norm(overlap.data) != 0:
        raise ValueError("Mur rows overlap grounded Gauss-gradient support")
    return MurContext(n, h, dt, c, eps0, tolerance, longitudinal, volume, matter, field,
                      rows, adjacent, inactive, nu, r, trace, stiffness, left,
                      splu(left.tocsc()) if policy.method == "cpu-direct" else None,
                      splu(poisson), fft_size, green_fft, policy)


def _real_array(values, shape, name):
    array = np.asarray(values)
    if array.shape != shape or not np.isrealobj(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite real array of shape {shape}")
    return array.astype(float, copy=False)


def longitudinal_field(rho, ctx):
    """Compatible Y_L=G phi, with G_f.T Y_L=rho on interior nodes."""
    m = len(ctx.matter.free)
    charge = _real_array(rho, (m,), "rho")
    if not np.any(charge):
        return np.zeros(3 * m)
    phi = np.zeros(m)
    if ctx.closure == "free-space":
        transformed = np.fft.fftn(charge.reshape(ctx.grid, order="F"),
                                  s=ctx.fft_size, axes=(0, 1, 2))
        potential = np.fft.ifftn(transformed * ctx.green_fft).real
        crop = tuple(slice(size - 1, 2 * size - 1) for size in ctx.grid)
        phi[~ctx.matter.free] = potential[crop].ravel(order="F")[~ctx.matter.free]
    boundary = ctx.field.gradient @ phi
    interior = ctx.poisson_solve.solve(charge[ctx.matter.free] - ctx.field.free_gradient.T @ boundary)
    return boundary + ctx.field.free_gradient @ interior


def initialize_mur(rho, ctx, history=None):
    """Create the fixed initial harmonic reference once; reject broken history."""
    if history is not None:
        m = len(ctx.matter.free)
        _real_array(history.initial_longitudinal_y, (3*m,), "initial longitudinal Y")
        _real_array(history.initial_harmonic_y, (3*m,), "initial harmonic Y")
        if history.previous_trace_y is not None:
            _real_array(history.previous_trace_y, (3*m,), "previous trace Y")
        return history
    initial = longitudinal_field(rho, ctx)
    grounded = ctx.field.free_gradient @ ctx.poisson_solve.solve(np.asarray(rho)[ctx.matter.free])
    return MurHistory(initial, initial - grounded)


def mur_reference_field(rho, ctx, history):
    """Grounded charge response plus the unchanged initial harmonic gradient."""
    history = initialize_mur(rho, ctx, history)
    charge = _real_array(rho, (len(ctx.matter.free),), "rho")
    return ctx.field.free_gradient @ ctx.poisson_solve.solve(charge[ctx.matter.free]) + history.initial_harmonic_y


def gauss_project(electric_momentum, rho, ctx):
    """Project the adjoint forward constraint and record actual electric work."""
    m = len(ctx.matter.free)
    old = _real_array(electric_momentum, (m, 3), "electric momentum")
    charge = _real_array(rho, (m,), "rho")
    defect = ctx.field.free_gradient.T @ old.ravel(order="F") - charge[ctx.matter.free]
    scale = np.linalg.norm(charge[ctx.matter.free])
    before = np.linalg.norm(defect) / scale if scale > 0 else np.linalg.norm(defect)
    delta = -ctx.field.free_gradient @ ctx.poisson_solve.solve(defect)
    new = old + delta.reshape((m, 3), order="F")
    energy = ctx.volume * np.sum((new-old) * (new+old)) / (2 * ctx.eps0)
    return new, {"before": float(before), "energy": float(energy),
                 "correctionNorm": float(np.linalg.norm(delta))}


def mur_field_step(vector_potential, electric_momentum, rho, ctx, history=None, background_a=None, *,
                   solve_log=None):
    """Advance source Mur-1 Maxwell midpoint map, returning fields/history/info."""
    m = len(ctx.matter.free)
    a_total = _real_array(vector_potential, (m, 3), "vector potential")
    y = _real_array(electric_momentum, (m, 3), "electric momentum")
    background = np.zeros((m, 3)) if background_a is None else _real_array(background_a, (m, 3), "background A")
    a = (a_total-background).ravel(order="F")
    old = y.ravel(order="F")
    if np.linalg.norm(a[ctx.inactive]) + np.linalg.norm(old[ctx.inactive]) >= 1e-11:
        raise ValueError("padded forward-grid edges must have zero perturbation")
    history = initialize_mur(rho, ctx, history)
    longitudinal = mur_reference_field(rho, ctx, history)
    u = old - longitudinal
    rhs = old - ctx.dt**2 * ctx.c**2 * (ctx.stiffness @ old) / 4
    rhs -= ctx.dt * ctx.eps0 * ctx.c**2 * (ctx.stiffness @ a)
    rhs[ctx.rows] = u[ctx.adjacent] - ctx.r * u[ctx.rows] + ctx.trace @ longitudinal
    rhs[ctx.inactive] = 0
    operator = mur_operator(ctx.field.curl, ctx.trace, ctx.rows, ctx.inactive, ctx.dt**2*ctx.c**2/4)
    acceptance = lambda vector: np.linalg.norm(ctx.left @ vector-rhs)/max(np.linalg.norm(rhs), np.finfo(float).tiny)
    new, report = solve_system(operator, rhs, ctx.tolerance, ctx.solver_policy,
        direct_matrix=ctx.left, direct_solver=None if ctx.solve is None else ctx.solve.solve,
        acceptance=acceptance, matrix_free=True, solve_log=solve_log)
    residual = report.original_residual
    an = a + ctx.dt * (old + new) / (2 * ctx.eps0)
    midpoint = (old + new) / 2
    force = (new-old) / ctx.dt + ctx.eps0 * ctx.c**2 * (ctx.stiffness @ ((a+an)/2))
    work_scale = ctx.volume * ctx.dt / ctx.eps0
    work = work_scale * np.dot(midpoint[ctx.rows], force[ctx.rows])
    reference_work = work_scale * np.dot(longitudinal[ctx.rows], force[ctx.rows])
    info = {"residual": float(residual), "boundaryWork": float(work),
            "boundaryReferenceWork": float(reference_work),
            "boundaryWaveWork": float(work-reference_work),
            "gaussChange": float(np.linalg.norm(ctx.field.free_gradient.T @ (new-old)))}
    next_history = MurHistory(history.initial_longitudinal_y, history.initial_harmonic_y, u.copy())
    return an.reshape((m, 3), order="F") + background, new.reshape((m, 3), order="F"), next_history, info
