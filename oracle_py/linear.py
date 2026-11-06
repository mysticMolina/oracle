"""Jacob Matthew Molina. Inception: 2026-11-05.

Explicit CPU direct/Krylov boundary for the accepted MATLAB linear maps.
GMRES uses the source restart40,20cycles and inner tolerance min(tol/10,1e-10).
The physical caller supplies its ORIGINAL-system residual acceptance check.
No array here is transferred to a GPU or distributed across ranks.
"""

from dataclasses import asdict, dataclass
import inspect

import numpy as np
from scipy import sparse
from scipy.sparse import linalg


@dataclass(frozen=True)
class LinearPolicy:
    method: str = "cpu-direct"
    restart: int = 40
    max_cycles: int = 20
    allow_direct_fallback: bool = False

    def __post_init__(self):
        if self.method not in ("cpu-direct", "cpu-krylov"):
            raise ValueError("implemented linear backends are cpu-direct and cpu-krylov")
        for value in (self.restart, self.max_cycles):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("GMRES restart and cycle count must be positive integers")
        if not isinstance(self.allow_direct_fallback, bool):
            raise ValueError("direct fallback requires an explicit boolean")
        if self.method == "cpu-direct" and self.allow_direct_fallback:
            raise ValueError("direct fallback is meaningful only for cpu-krylov")


def backend_metadata(policy):
    return {"linearBackend": policy.method, "device": "cpu",
            "restart": policy.restart, "maxCycles": policy.max_cycles,
            "allowDirectFallback": policy.allow_direct_fallback,
            "poissonBackend": "cpu-direct", "distributed": False,
            "operatorStorage": "assembled sparse spatial operators",
            "solveApplication": "composed matrix-free" if policy.method == "cpu-krylov" else "assembled direct"}


def policy_from_config(config):
    """Numerical backend is explicit case metadata; default remains direct."""
    return LinearPolicy(method=config.get("linearBackend", "cpu-direct"),
                        allow_direct_fallback=config.get("allowDirectFallback", False))


@dataclass(frozen=True)
class SolveReport:
    requested_method: str
    used_method: str
    device: str
    matrix_free: bool
    iterations: int
    restart: int
    max_cycles: int
    krylov_flag: int | None
    inner_tolerance: float
    linear_residual: float
    original_residual: float
    accepted: bool
    fallback_reason: str | None
    residual_history: tuple

    def to_dict(self):
        record = asdict(self)
        # Nonfinite failure evidence must remain serializable, not a JSON NaN.
        for key in ("linear_residual", "original_residual"):
            if not np.isfinite(record[key]):
                record[key] = None
        record["residual_history"] = [float(v) if np.isfinite(v) else None for v in self.residual_history]
        return record


class LinearSolveError(ArithmeticError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def solve_system(operator, rhs, tolerance, policy=None, *, direct_matrix=None,
                 direct_solver=None, acceptance=None, matrix_free=False, solve_log=None):
    """Solve one system, retaining direct reference and explicit fallback.

    `operator` provides matvec. `direct_matrix` may be a lazy factory, so a
    Schur product need not be assembled for a successful matrix-free solve.
    `acceptance(x)` returns the source's original-system relative residual.
    GMRES failure never silently falls back; callers must request that policy.
    """
    policy = LinearPolicy() if policy is None else policy
    if not isinstance(policy, LinearPolicy):
        raise ValueError("policy must be LinearPolicy")
    b = np.asarray(rhs)
    if b.ndim != 1 or not np.all(np.isfinite(b)) or operator.shape != (b.size, b.size):
        raise ValueError("operator and finite vector RHS shapes must agree")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("solve tolerance must be finite and positive")
    inner = min(tolerance/10, 1e-10)
    denominator = max(np.linalg.norm(b), np.finfo(float).tiny)
    history, flag, used, fallback = [], None, policy.method, None

    def direct():
        if direct_solver is not None:
            return direct_solver(b)
        matrix = direct_matrix() if callable(direct_matrix) else direct_matrix
        if matrix is None:
            matrix = operator
        if isinstance(matrix, linalg.LinearOperator):
            raise ValueError("direct solve requires an explicit matrix/factory")
        return linalg.spsolve(matrix, b) if sparse.issparse(matrix) else np.linalg.solve(matrix, b)

    def residuals(x):
        if not np.all(np.isfinite(x)):
            return np.inf, np.inf
        linear = float(np.linalg.norm(operator @ x-b)/denominator)
        original = float(acceptance(x)) if acceptance is not None else linear
        return linear, original

    if policy.method == "cpu-direct":
        x = direct()
    else:
        # SciPy1.10 spells the relative tolerance tol; newer versions use rtol.
        options = {"restart": policy.restart, "maxiter": policy.max_cycles,
                   "atol": 0., "callback": lambda value: history.append(float(value)),
                   "callback_type": "pr_norm"}
        options["rtol" if "rtol" in inspect.signature(linalg.gmres).parameters else "tol"] = inner
        x, flag = linalg.gmres(operator, b, **options)
    linear_residual, original_residual = residuals(x)
    accepted = bool(np.isfinite(original_residual) and original_residual <= tolerance
                    and np.isfinite(linear_residual) and (flag in (None, 0)))
    if not accepted and policy.method == "cpu-krylov" and policy.allow_direct_fallback:
        fallback = f"GMRES flag {flag}; original residual {original_residual:.6g}"
        x = direct()
        used = "cpu-direct"
        linear_residual, original_residual = residuals(x)
        accepted = bool(np.isfinite(original_residual) and original_residual <= tolerance
                        and np.isfinite(linear_residual))
    report = SolveReport(policy.method, used, "cpu", bool(matrix_free and used == "cpu-krylov"), len(history),
        policy.restart, policy.max_cycles, None if flag is None else int(flag), inner,
        linear_residual, original_residual, accepted, fallback, tuple(history))
    if solve_log is not None:
        solve_log.append(report)
    if not accepted:
        raise LinearSolveError(f"{policy.method} failed: GMRES flag {flag}, original residual {original_residual:.6g}", report)
    return np.asarray(x), report


def schur_operator(hop, coefficient):
    """Apply I+coefficient*K*K^H without constructing the Schur product."""
    adjoint = hop.conj().T
    return linalg.LinearOperator(hop.shape, dtype=np.result_type(hop.dtype, coefficient),
        matvec=lambda vector: vector + coefficient*(hop @ (adjoint @ vector)),
        rmatvec=lambda vector: vector + np.conj(coefficient)*(hop @ (adjoint @ vector)))


def mur_operator(curl, trace, rows, inactive, coefficient):
    """Apply source Mur row replacement after I+coefficient*C^T*C."""
    def apply(vector):
        result = np.asarray(vector + coefficient*(curl.T @ (curl @ vector))).copy()
        result[rows] = trace @ vector
        result[inactive] = vector[inactive]
        return result
    return linalg.LinearOperator((curl.shape[1], curl.shape[1]), matvec=apply, dtype=curl.dtype)
