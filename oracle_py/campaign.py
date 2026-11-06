"""Jacob Matthew Molina. Inception: 2026-11-05.

CPU single-species ORACLE campaign transactions and native MAT products.
All four source controls advance to one accepted time. Runtime stop_after
bounds work without changing the requested physical duration in case JSON.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import scipy
from scipy.io import savemat

from .checkpoint import (TOTAL_KEYS, atomic_file, atomic_json, file_sha256,
                         load_checkpoint, run_lock, save_checkpoint, validate_states)
from .core import unpack_field
from .diagnostics import boundary_probes, diagnostics
from .evolution import State, coupled_mur_step
from .initialization import prepare_matched_campaign
from .matter import charge_density
from .linear import backend_metadata
from .maxwell import initialize_mur


INFO_KEYS = ("matterResidual", "emResidual", "continuityResidual", "gaussBeforeProjection",
             "projectionEnergy", "projectionNorm", "boundaryWork", "boundaryWaveWork",
             "boundaryReferenceWork", "maxwellGaussChange", "continuityHalfStepOverPeakCharge",
             "chargeChangeLinfOverPeakCharge", "linearSolveCount", "krylovIterations",
             "directFallbackCount", "maxLinearResidual", "maxOriginalSolveResidual")


def runtime_record():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}


def source_record():
    """Hash installed numerical/runtime Python source, independent of Git dirt."""
    package = Path(__file__).resolve().parent
    files = {path.name: file_sha256(path) for path in sorted(package.glob("*.py"))}
    identity = hashlib.sha256(json.dumps({"files": files, "runtime": runtime_record()}, sort_keys=True).encode()).hexdigest()
    return identity, files


def _identity(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _public(state, grid):
    return {"psi": unpack_field(state.psi, grid), "A": unpack_field(state.a, grid), "Y": unpack_field(state.y, grid)}


def _snapshot(folder, states, step, problem, artifacts):
    name = f"state_{step:09d}.mat"
    target = folder/name
    if name in artifacts:
        if not target.is_file() or file_sha256(target) != artifacts[name]:
            raise ValueError("Existing committed snapshot changed")
        return
    if target.exists() or target.is_symlink():
        raise ValueError("Snapshot exists outside committed receipt")
    cells = np.empty((1, 4), dtype=object)
    for branch, state in enumerate(states):
        cells[0, branch] = _public(state, problem.context.grid)
    snapshot = dict(cells[0, 0], schemaVersion=3, step=step, time=step*problem.context.dt)
    atomic_file(target, lambda stream: savemat(stream, {"snapshot": snapshot, "allStates": cells}, do_compression=True))
    artifacts[name] = file_sha256(target)


def _forensic_candidate(folder, states, step, p, info, row, error, source, config_identity):
    """Preserve rejected arrays separately from canonical accepted snapshots."""
    name = f"failed_candidate_{step:09d}.mat"
    target = folder/name
    # An interrupted stopped-checkpoint commit may leave prior attempt evidence.
    # Keep it immutable and reserve a distinct name under the campaign writer lock.
    attempt = 1
    while target.exists() or target.is_symlink():
        name = f"failed_candidate_{step:09d}_attempt_{attempt:04d}.mat"
        target = folder/name
        attempt += 1
    cells = np.empty((1, 4), dtype=object)
    for branch, state in enumerate(states):
        expected = len(p.context.matter.free)
        valid_shape = (state.psi.shape == (expected, 4) and
                       state.a.shape == (expected, 3) and state.y.shape == (expected, 3))
        public = _public(state, p.context.grid) if valid_shape else {
            "psi": state.psi, "A": state.a, "Y": state.y, "rawPackedInvalidShape": True}
        history = state.mur
        public["mur"] = {} if history is None else {
            "initialLongitudinalY": history.initial_longitudinal_y,
            "initialHarmonicY": history.initial_harmonic_y,
            "previousTraceY": np.array([]) if history.previous_trace_y is None else history.previous_trace_y}
        cells[0, branch] = public
    record = {"accepted": False, "step": step, "time": step*p.context.dt,
              "reason": error, "source": source, "config": config_identity, "info": info}
    if row is not None:
        record["diagnostics"] = row
    atomic_file(target, lambda stream: savemat(stream, {"candidate": record, "allStates": cells},
                                              do_compression=True, long_field_names=True))
    return {"file": name, "sha256": file_sha256(target), "accepted": False}


def _fresh_artifacts(folder, p, source, files, config_identity):
    cfg, ctx = p.configuration, p.context
    every = max(1, int(np.floor(cfg["steps"]/cfg["saved_intervals"]+.5)))
    sim = {"schemaVersion": 2, "formulation": "bare-dirac", "name": cfg["name"], "mode": "both",
           "grid": np.array(ctx.grid), "spacing": ctx.spacing, "origin": p.origin,
           "physics": {"q": -1., "m": 1., "hbar": 1., "c": ctx.c, "eps0": ctx.eps0},
           "operatorProfile": "compatible-forward", "matterIntegrator": "full-cayley",
           "projectGauss": True, "boundary": cfg["boundary"], "dt": ctx.dt, "steps": cfg["steps"],
           "startTime": 0., "solveTolerance": ctx.tolerance,
           "output": {"name": cfg["name"], "root": str(folder.parent), "every": every, "enabled": True},
           "initial": _public(p.initial, ctx.grid), "backgroundA": unpack_field(p.background_a, ctx.grid),
           "backgroundB": unpack_field(p.background_b, ctx.grid),
           "backgroundCharge": unpack_field(p.background_charge[:, None], ctx.grid), "caseDefinition": cfg}
    meta = {"solverVariant": "oracle_python_planar_v1", "B": cfg["B"], "source": source,
            "config": config_identity, "units": "atomic", "operatorProfile": "compatible-forward",
            "boundaryCondition": "mur1", "longitudinal": ctx.closure,
            "boundaryReference": "grounded_gauss_plus_fixed_initial_harmonic",
            "boundaryValidation": "source-bound local parity; production convergence pending",
            "linearSolver": backend_metadata(ctx.solver_policy)}
    atomic_json(folder/"configuration.json", {"configuration": cfg, "sourceIdentity": source,
                 "configurationIdentity": config_identity, "sourceFiles": files, "runtime": runtime_record(),
                 "linearSolver": backend_metadata(ctx.solver_policy)})
    atomic_file(folder/"configuration.mat", lambda stream: savemat(stream, {"sim": sim, "meta": meta}, do_compression=True))
    package = Path(__file__).resolve().parent
    def archive(stream):
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for name in files:
                output.write(package/name, arcname="oracle_py/"+name)
    atomic_file(folder/"source_snapshot.zip", archive)
    return {name: file_sha256(folder/name) for name in ("configuration.json", "configuration.mat", "source_snapshot.zip")}


def _states(p):
    initial = p.initial
    candidates = (initial,
        State(initial.psi.copy(), p.background_a.copy(), np.zeros_like(initial.y)),
        State(initial.psi.copy(), np.zeros_like(initial.a), np.zeros_like(initial.y)),
        State(np.zeros_like(initial.psi), p.background_a.copy(), np.zeros_like(initial.y)))
    return tuple(State(s.psi.copy(), s.a.copy(), s.y.copy(),
        initialize_mur(charge_density(s.psi, -1., p.background_charge), p.context)) for s in candidates)


def _diagnostics(state, p):
    return diagnostics(state, p.context, p.origin, background_charge=p.background_charge,
                       background_a=p.background_a, background_b=p.background_b)


def _row(states, step, p, totals, info, initial):
    validate_states(states, len(p.context.matter.free))
    row = _diagnostics(states[0], p)
    scale = max(abs(initial["totalEnergy"]), np.finfo(float).tiny)
    row.update(step=step, time=step*p.context.dt, cumulativeProjectionEnergy=totals["projection_work"],
               energyMinusProjection=row["totalEnergy"]-totals["projection_work"],
               cumulativeBoundaryWork=totals["boundary_work"], cumulativeWaveBoundaryWork=totals["wave_work"],
               cumulativeReferenceBoundaryWork=totals["reference_work"], boundaryInjectionFraction=totals["wave_work"]/scale,
               energyBalance=row["totalEnergy"]-totals["projection_work"]-totals["boundary_work"])
    row.update(info)
    row["prescribedKinetic"] = _diagnostics(states[1], p)["kineticPerNorm"]
    row["freeKinetic"] = _diagnostics(states[2], p)["kineticPerNorm"]
    row.update(boundary_probes(states[0], p.context, p.background_a, p.background_b, p.background_charge))
    if not all(np.isfinite(value) for value in row.values()):
        raise ArithmeticError("Nonfinite scalar diagnostics")
    return row


def _screen(row, initial, config):
    criteria = config.get("screen", {})
    boundary = config["boundary"]
    energy_scale = max(abs(initial["totalEnergy"]), np.finfo(float).tiny)
    gauss = row["gaussRelative"] if row["gaussRelativeDefined"] else row["gaussAbsolute"]
    limits = [(gauss <= 1e-8, "Gauss residual"),
              (row["highKFraction"] <= criteria.get("maxHighKFraction", np.inf), "high-k fraction"),
              (abs(row["energyBalance"]-initial["totalEnergy"])/energy_scale <= criteria.get("maxRelativeEnergyBalance", np.inf), "adjusted energy budget"),
              (abs(row["norm"]/initial["norm"]-1) <= 1e-8, "norm drift"),
              (row["boundaryInjectionFraction"] <= boundary.get("maxInjectedWaveFraction", np.inf), "wave boundary work"),
              (row["boundaryMassFraction"] <= boundary.get("maxPacketBoundaryFraction", 1e-6), "packet collar")]
    for passed, name in limits:
        if not passed:
            raise ArithmeticError(f"Campaign screen exceeded: {name}")


def _write_row(stream, row, header=False):
    writer = csv.DictWriter(stream, fieldnames=list(row))
    if header:
        writer.writeheader()
    writer.writerow({key: format(value, ".17g") for key, value in row.items()})


def run_campaign(config, folder, *, resume=False, stop_after=None, checkpoint_every=25, budget_seconds=None):
    """Run bounded synchronized controls, preserving the original target.

    Restart validates before rollback; stopped scientific screens require a
    new reviewed configuration. A numerical or scientific-screen failure keeps
    the previous accepted state and marks the run stopped; complete rejected
    candidates are retained separately as forensic MAT files. No Slurm is invoked.
    """
    cfg = json.loads(Path(config).read_text()) if isinstance(config, (str, Path)) else config
    for key in ("steps", "saved_intervals"):
        if isinstance(cfg.get(key), bool) or not isinstance(cfg.get(key), (int, float)) or not np.isfinite(cfg[key]) or cfg[key] <= 0 or int(cfg[key]) != cfg[key]:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(cfg.get("name"), str) or not cfg["name"].isidentifier():
        raise ValueError("Case name must be a path-free identifier")
    if stop_after is not None and (isinstance(stop_after, bool) or not isinstance(stop_after, int) or stop_after < 0):
        raise ValueError("stop_after must be a nonnegative integer")
    if isinstance(checkpoint_every, bool) or not isinstance(checkpoint_every, int) or checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be a positive integer")
    if budget_seconds is not None and (not np.isfinite(budget_seconds) or budget_seconds <= 0):
        raise ValueError("budget_seconds must be finite and positive")
    if "target_time" in cfg and not np.isclose(cfg["steps"]*cfg["dt"], cfg["target_time"], rtol=1e-12, atol=0):
        raise ValueError("dt and steps disagree with target_time")
    if "periods" in cfg and not np.isclose(cfg["steps"]*cfg["dt"], cfg["periods"]*cfg["reference_period"], rtol=1e-12, atol=0):
        raise ValueError("Requested periods disagree with physical endpoint")
    source, files = source_record()
    config_id = _identity(cfg)
    folder = Path(folder)
    with run_lock(folder):
        checkpoint = folder/"checkpoint.npz"
        restored = None
        if checkpoint.exists():
            if not resume:
                raise ValueError("Existing checkpoint requires explicit resume")
            restored = load_checkpoint(folder, config_id, source, recover=False)
        elif resume or any(path.name != ".production.lock" for path in folder.iterdir()):
            raise ValueError("Output exists without a committed checkpoint, or resume source is missing")
        p, preparation = prepare_matched_campaign(cfg)
        if restored is None:
            states = _states(p)
            initial = _diagnostics(states[0], p)
            step, totals, info, status = 0, {key: 0. for key in TOTAL_KEYS}, {key: 0. for key in INFO_KEYS}, "partial"
            artifacts = _fresh_artifacts(folder, p, source, files, config_id)
            row = _row(states, step, p, totals, info, initial)
            _screen(row, initial, cfg)
            with (folder/"coupled.csv").open("w", encoding="utf-8", newline="") as stream:
                _write_row(stream, row, header=True)
            _snapshot(folder, states, step, p, artifacts)
            save_checkpoint(folder, states, step, totals, info, initial, status, config_id, source, artifacts)
        else:
            states, metadata = restored
            if states[0].psi.shape != p.initial.psi.shape or metadata["step"] > cfg["steps"]:
                raise ValueError("Checkpoint grid or step is incompatible")
            if set(metadata["info"]) != set(INFO_KEYS):
                raise ValueError("Checkpoint diagnostic schema differs")
            # All domain-specific checks pass before any journal rollback.
            states, metadata = load_checkpoint(folder, config_id, source)
            step, totals, info = metadata["step"], metadata["totals"], metadata["info"]
            initial, artifacts, status = metadata["initial"], metadata["artifacts"], metadata["status"]
            row = _row(states, step, p, totals, info, initial)
        timer, durations, error = time.monotonic(), [], ""
        attempted_step, rejected = step, None
        candidate_states, candidate_row, next_info = None, None, None
        solve_log = []
        endpoint = min(cfg["steps"], cfg["steps"] if stop_after is None else stop_after)
        every = max(1, int(np.floor(cfg["steps"]/cfg["saved_intervals"]+.5)))
        try:
            with (folder/"coupled.csv").open("a", encoding="utf-8", newline="") as stream:
                while step < endpoint:
                    reserve = 2*max(durations[-6:]) if durations else 0
                    if budget_seconds is not None and time.monotonic()-timer+reserve >= budget_seconds:
                        break
                    tick = time.monotonic()
                    attempted_step = step + 1
                    candidate_states, candidate_row, next_info = None, None, None
                    candidates, solve_log = [], []
                    for branch, state in enumerate(states):
                        mode = "both" if branch == 0 else "QM" if branch < 3 else "EM"
                        candidate, branch_info = coupled_mur_step(state, p.context, mode=mode,
                            background_charge=p.background_charge, background_a=p.background_a, solve_log=solve_log)
                        candidates.append(candidate)
                        if branch == 0:
                            next_info = branch_info
                    candidate_states = tuple(candidates)
                    validate_states(candidate_states, len(p.context.matter.free))
                    old_rho = charge_density(states[0].psi, -1., p.background_charge)
                    new_rho = charge_density(candidates[0].psi, -1., p.background_charge)
                    peak = max(float(np.max(np.abs(old_rho))), np.finfo(float).tiny)
                    next_info["continuityHalfStepOverPeakCharge"] = next_info["continuityResidual"]*(p.context.dt/2)/peak
                    next_info["chargeChangeLinfOverPeakCharge"] = float(np.max(np.abs(new_rho-old_rho)))/peak
                    next_info.update(linearSolveCount=len(solve_log),
                        krylovIterations=sum(report.iterations for report in solve_log),
                        directFallbackCount=sum(report.fallback_reason is not None for report in solve_log),
                        maxLinearResidual=max(report.linear_residual for report in solve_log),
                        maxOriginalSolveResidual=max(report.original_residual for report in solve_log))
                    candidate_totals = totals.copy()
                    for key, name in [("projection_work", "projectionEnergy"), ("boundary_work", "boundaryWork"),
                                      ("wave_work", "boundaryWaveWork"), ("reference_work", "boundaryReferenceWork")]:
                        candidate_totals[key] += next_info[name]
                    candidate_row = _row(candidate_states, attempted_step, p, candidate_totals, next_info, initial)
                    _screen(candidate_row, initial, cfg)
                    # Scientific acceptance precedes publication of state, work or CSV.
                    states, step, totals, info, row = candidate_states, attempted_step, candidate_totals, next_info, candidate_row
                    _write_row(stream, row)
                    if step % every == 0 or step == cfg["steps"]:
                        _snapshot(folder, states, step, p, artifacts)
                    if step % checkpoint_every == 0:
                        stream.flush()
                        save_checkpoint(folder, states, step, totals, info, initial, "partial", config_id, source, artifacts)
                    durations.append(time.monotonic()-tick)
            status = "completed" if step == cfg["steps"] else "partial"
        except (ArithmeticError, ValueError) as exception:
            status, error = "stopped", str(exception)
            if candidate_states is not None:
                rejected = _forensic_candidate(folder, candidate_states, attempted_step, p,
                    next_info, candidate_row, error, source, config_id)
        _snapshot(folder, states, step, p, artifacts)
        save_checkpoint(folder, states, step, totals, info, initial, status, config_id, source, artifacts)
        result = {"status": status, "step": step, "lastAcceptedStep": step, "lastAttemptedStep": attempted_step,
                  "rejectedCandidate": rejected, "stepsRequested": cfg["steps"], "time": step*p.context.dt,
                  "source": source, "config": config_id, "outputFolder": str(folder.resolve()), "error": error,
                  "linearSolver": backend_metadata(p.context.solver_policy),
                  "lastAttemptedLinearSolves": [report.to_dict() for report in solve_log],
                  "elapsedSeconds": time.monotonic()-timer, "medianStepSeconds": float(np.median(durations)) if durations else 0.,
                  "diagnostics": row, "cumulativeProjectionEnergy": totals["projection_work"],
                  "cumulativeBoundaryWork": totals["boundary_work"], "cumulativeWaveBoundaryWork": totals["wave_work"],
                  "cumulativeReferenceBoundaryWork": totals["reference_work"]}
        atomic_json(folder/"status.json", result)
        metadata = {"status": status, "lastCompletedStep": step, "schemaVersion": 3,
                    "arrayOrder": "[Ny,Nx,Nz,component]", "source": source, "implementation": "python-cpu"}
        atomic_file(folder/"metadata.mat", lambda stream: savemat(stream, {"metadata": metadata}))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--budget-seconds", type=float)
    args = parser.parse_args()
    result = run_campaign(args.config, args.output, resume=args.resume, stop_after=args.stop_after,
                          checkpoint_every=args.checkpoint_every, budget_seconds=args.budget_seconds)
    print(json.dumps(result, indent=2))
    if result["status"] == "stopped":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
