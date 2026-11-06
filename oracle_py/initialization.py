"""Jacob Matthew Molina. Inception: 2026-11-05.

Source matched-planar positive-energy coherent Landau preparation, with
its fixed 0..80 truncation, zero matter faces, and grid normalization.
Only the planar Mur-1 full-Cayley CPU route is enabled in this port slice.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .diagnostics import diagnostics
from .evolution import State
from .matter import charge_density, dirac_current
from .linear import policy_from_config
from .maxwell import gauss_project, longitudinal_field, mur_context


@dataclass(frozen=True)
class MatchedProblem:
    configuration: dict
    context: object
    initial: State
    origin: np.ndarray
    background_a: np.ndarray
    background_b: np.ndarray
    background_charge: np.ndarray
    resolution_receipt: dict
    initial_gauss_projection: dict


def _positive(value, name, zero=False):
    if not np.isscalar(value) or not np.isreal(value) or not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return float(value)


def matched_problem(options):
    """Port +oracle/matchedProblem using unchanged physical/grid options.

    Unused case metadata remains in configuration; it never silently selects
    a different method or changes the source normalization/truncation.
    """
    config = deepcopy(options)
    n = np.asarray(config.get("grid", 193))
    if n.ndim == 0:
        n = np.array([n.item(), n.item(), 1])
    if n.shape != (3,) or not np.all(np.isfinite(n)) or not np.all(n == np.floor(n)) or n[2] != 1 or np.any(n[:2] < 3):
        raise ValueError("matched grid must be Nx,Ny>=3 and Nz=1")
    n = tuple(n.astype(int))
    c = _positive(config.get("c", 10.), "c")
    b = _positive(config.get("B", 10.), "B")
    half = _positive(config.get("half", 4.), "half")
    radius = _positive(config.get("radius", 1.5), "radius", zero=True)
    number = _positive(config.get("norm", .1), "norm")
    dt = _positive(config.get("dt", .0005), "dt")
    steps = config.get("steps", 10)
    if not np.isscalar(steps) or not np.isreal(steps) or not np.isfinite(steps) or steps < 0 or steps != np.floor(steps):
        raise ValueError("steps must be a finite nonnegative integer")
    boundary = config.get("boundary", {})
    if boundary.get("type") != "mur1":
        raise ValueError("the Python matched route currently requires explicit Mur-1")
    policy = policy_from_config(config)
    spacing = np.r_[2*half/(np.array(n[:2])-1), 1.]
    ell = 1/np.sqrt(b)
    actual = ell/spacing[:2]
    receipt = {"magneticLength": ell, "cellsPerMagneticLength": actual,
               "stepOverMagneticLength": 1/actual, "enforced": False}
    if "resolution" in config:
        rule = config["resolution"]
        if not isinstance(rule, dict) or "targetCellsPerMagneticLength" not in rule:
            raise ValueError("resolution requires targetCellsPerMagneticLength")
        target = np.asarray(rule["targetCellsPerMagneticLength"], dtype=float)
        if target.ndim == 0:
            target = np.repeat(target, 2)
        tolerance = rule.get("relativeTolerance", .02)
        if target.shape != (2,) or not np.all(np.isfinite(target)) or np.any(target <= 0):
            raise ValueError("resolution target must contain one or two finite positive values")
        if not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance < 0 or tolerance >= 1:
            raise ValueError("resolution tolerance must be in [0,1)")
        receipt.update(targetCellsPerMagneticLength=target, relativeTolerance=float(tolerance), enforced=True)
        if np.any(np.abs(actual-target)/target > tolerance):
            raise ValueError("grid violates requested magnetic-length resolution")
    ctx = mur_context(n, spacing, dt, c, 1/(4*np.pi), boundary.get("longitudinal", "free-space"), 1e-11,
                      solver_policy=policy)
    x, y = np.meshgrid(np.linspace(-half, half, n[0]), np.linspace(-half, half, n[1]), indexing="ij")
    m = int(np.prod(n))
    background_a = np.column_stack([-b*y.ravel(order="F")/2, b*x.ravel(order="F")/2, np.zeros(m)])
    background_b = np.column_stack([np.zeros((m, 2)), b*np.ones(m)])
    phi = np.exp(-(x*x+y*y)*b/4)*np.sqrt(b/(2*np.pi))
    w = 1j*(x+1j*y)*np.sqrt(b/2)
    previous = np.zeros_like(phi)
    nbar = b*radius**2/2
    coefficient = complex(np.exp(-nbar/2))
    psi = np.zeros((m, 4), dtype=complex)
    for k in range(81):
        energy = np.sqrt(c**4+2*c*c*b*k)
        upper = np.sqrt((energy+c*c)/(2*energy))
        lower = np.sqrt((energy-c*c)/(2*energy))
        psi[:, 1] += coefficient*upper*phi.ravel(order="F")
        if k > 0:
            psi[:, 2] += coefficient*lower*previous.ravel(order="F")
        previous = phi
        phi = phi*w/np.sqrt(k+1)
        coefficient *= (-1j*np.sqrt(nbar))/np.sqrt(k+1)
    psi[~ctx.matter.free] = 0
    integral = np.sum(np.abs(psi)**2)*ctx.volume
    if not np.isfinite(integral) or integral <= 0:
        raise ValueError("truncated matched state has no finite positive grid norm")
    psi *= np.sqrt(number/integral)
    background_charge = np.zeros(m)
    rho = charge_density(psi, -1., background_charge)
    y0 = longitudinal_field(rho, ctx).reshape((m, 3), order="F")
    y0, info = gauss_project(y0, rho, ctx)
    return MatchedProblem(config, ctx, State(psi, background_a.copy(), y0), np.array([-half, -half, 0.]),
                          background_a, background_b, background_charge, receipt, info)


def prepare_matched_campaign(options):
    """Apply the source run_coherent launch beta and packet-collar guards.

    This prepares one state and a receipt. It does not run, schedule, or
    establish production acceptance. Output cadence/restart belong elsewhere.
    """
    problem = matched_problem(options)
    p = problem
    current = dirac_current(p.initial.psi, -1., p.context.c)
    number = np.sum(np.abs(p.initial.psi)**2)*p.context.volume
    velocity = np.sum(current[:, :2], axis=0)*p.context.volume/(-number)
    beta = np.linalg.norm(velocity)/p.context.c
    target = options.get("beta_target")
    if target is None or not np.isfinite(target) or abs(beta-target) >= 1e-8:
        raise ValueError("initial current does not match the case beta_target")
    d = diagnostics(p.initial, p.context, p.origin, background_charge=p.background_charge,
                    background_a=p.background_a, background_b=p.background_b)
    if d["boundaryMassFraction"] >= 1e-6:
        raise ValueError("initial packet reaches the source campaign collar guard")
    receipt = {"profile": "matched-planar", "betaRequested": float(target),
               "betaMeasured": float(beta), "velocity": velocity,
               "nbar": options["B"]*options["radius"]**2/2, "nmax": 80,
               "radiusParameter": options["radius"], "initialDiagnostic": d,
               "referencePeriod": options["reference_period"]}
    return problem, receipt
