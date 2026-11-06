"""Jacob Matthew Molina. Inception: 2026-11-05.

Single-species planar full-Cayley/Mur-1 symmetric composition. This map
owns no campaign I/O or restart schema; callers retain fields and history.
"""

from dataclasses import dataclass

import numpy as np

from .matter import charge_density, full_matter_step
from .maxwell import MurHistory, gauss_project, initialize_mur, mur_field_step


@dataclass(frozen=True)
class State:
    psi: np.ndarray
    a: np.ndarray
    y: np.ndarray
    mur: MurHistory | None = None


def coupled_mur_step(state, ctx, charge=-1., mass=1., hbar=1., mode="both",
                     project_gauss=True, background_charge=None, background_a=None, *, solve_log=None):
    """Source matter(dt/2), fields(dt), matter(dt/2), with mode controls.

    Both matter maps see their current A. Gauss projection is applied
    after each matter half-step only in coupled mode; its signed electric
    energy change remains explicit and is not claimed symplectic.
    """
    if mode not in ("QM", "EM", "both"):
        raise ValueError("mode must be QM, EM, or both")
    rho = charge_density(state.psi, charge, background_charge)
    if state.psi.shape != (len(ctx.matter.free), 4):
        raise ValueError("single-species state must contain [M,4] psi")
    if not np.all(np.isfinite([mass, hbar])) or min(mass, hbar) <= 0:
        raise ValueError("mass and hbar must be finite and positive")
    history = initialize_mur(rho, ctx, state.mur)
    psi, a, y = state.psi.copy(), state.a.copy(), state.y.copy()
    info = {key: 0. for key in ("matterResidual", "emResidual", "continuityResidual",
            "gaussBeforeProjection", "projectionEnergy", "projectionNorm", "boundaryWork",
            "boundaryWaveWork", "boundaryReferenceWork", "maxwellGaussChange")}
    for half in range(2):
        if mode != "EM":
            psi, y, matter = full_matter_step(psi, a, y, ctx.matter, ctx.dt / 2,
                                             charge, mass, ctx.c, hbar,
                                             feedback=(mode == "both"), tolerance=ctx.tolerance,
                                             solver_policy=ctx.solver_policy, solve_log=solve_log)
            info["matterResidual"] = max(info["matterResidual"], matter["residual"])
            info["continuityResidual"] = max(info["continuityResidual"],
                                              float(np.max(np.abs(matter["continuity"][ctx.matter.free]))))
            if project_gauss and mode == "both":
                rho = charge_density(psi, charge, background_charge)
                y, projection = gauss_project(y, rho, ctx)
                info["gaussBeforeProjection"] = max(info["gaussBeforeProjection"], projection["before"])
                info["projectionEnergy"] += projection["energy"]
                info["projectionNorm"] += projection["correctionNorm"]
        if half == 0 and mode != "QM":
            rho = charge_density(psi, charge, background_charge)
            a, y, history, field = mur_field_step(a, y, rho, ctx, history, background_a, solve_log=solve_log)
            info["emResidual"] = field["residual"]
            for key in ("boundaryWork", "boundaryWaveWork", "boundaryReferenceWork"):
                info[key] = field[key]
            info["maxwellGaussChange"] = field["gaussChange"]
    return State(psi, a, y, history), info
