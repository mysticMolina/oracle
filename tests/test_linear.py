"""Jacob Matthew Molina. Inception: 2026-11-05.

Direct and explicit CPU Krylov maps against independently assembled operators.
"""

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse
from scipy.io import loadmat

from oracle_py.evolution import State, coupled_mur_step
from oracle_py.linear import (LinearPolicy, LinearSolveError, mur_operator,
                              schur_operator, solve_system, policy_from_config)
from oracle_py.maxwell import mur_context

FIXTURES = Path(__file__).resolve().parents[1]/"fixtures"


class LinearBackends(unittest.TestCase):
    def test_composed_schur_and_mur_match_assembled_actions(self):
        rng = np.random.default_rng(738)
        k = sparse.csr_matrix(rng.normal(size=(9, 9))+1j*rng.normal(size=(9, 9)))
        coefficient = .12+.07j
        assembled = np.eye(9)+coefficient*(k @ k.conj().T).toarray()
        operator = schur_operator(k, coefficient)
        vector = rng.normal(size=9)+1j*rng.normal(size=9)
        np.testing.assert_allclose(operator @ vector, assembled @ vector, atol=2e-14)
        np.testing.assert_allclose(operator.H @ vector, assembled.conj().T @ vector, atol=2e-14)
        ctx = mur_context((7, 5, 1), (.17, .23, 1), .0004, 137.035999084)
        vector = rng.normal(size=105)
        op = mur_operator(ctx.field.curl, ctx.trace, ctx.rows, ctx.inactive, ctx.dt**2*ctx.c**2/4)
        np.testing.assert_allclose(op @ vector, ctx.left @ vector, atol=2e-15)

    def test_actual_matlab_mur_maps_all_modes_and_closures(self):
        reference = loadmat(FIXTURES/"mur_7x5_physical_c.mat", simplify_cells=True)
        for case in reference["cases"]:
            ctx = mur_context(reference["grid"], reference["spacing"], reference["dt"],
                reference["physics"]["c"], reference["physics"]["eps0"], case["closure"], 1e-11,
                solver_policy=LinearPolicy("cpu-krylov"))
            self.assertIsNone(ctx.solve)  # No cached Maxwell LU on Krylov route.
            for mode in ("both", "QM", "EM"):
                start = case["initial"]
                state = State(start["psi"], start["A"], start["Y"])
                reports = []
                direct = replace(ctx, solver_policy=LinearPolicy())
                expected = state
                for _ in range(3):
                    state, info = coupled_mur_step(state, ctx, mode=mode,
                        background_charge=reference["backgroundCharge"], background_a=reference["backgroundA"],
                        solve_log=reports)
                    expected, _ = coupled_mur_step(expected, direct, mode=mode,
                        background_charge=reference["backgroundCharge"], background_a=reference["backgroundA"])
                for value, other in ((state.psi, expected.psi), (state.a, expected.a), (state.y, expected.y),
                                     (state.mur.previous_trace_y, expected.mur.previous_trace_y)):
                    if value is not None:
                        np.testing.assert_allclose(value, other, atol=3e-11, rtol=3e-11)
                matlab = next(output["three"] for output in case["outputs"] if output["mode"] == mode)
                for value, name in ((state.psi, "psi"), (state.a, "A"), (state.y, "Y")):
                    np.testing.assert_allclose(value, matlab[name], atol=3e-11, rtol=3e-11)
                self.assertTrue(reports)
                self.assertTrue(all(r.accepted and r.krylov_flag == 0 and r.matrix_free for r in reports))
                self.assertTrue(all(r.original_residual <= 1e-11 and r.device == "cpu" for r in reports))
                self.assertTrue(all(r.fallback_reason is None for r in reports))

    def test_krylov_failure_and_only_explicit_direct_fallback(self):
        matrix = sparse.diags(np.arange(1., 11.), format="csr")
        rhs = np.ones(10)
        policy = LinearPolicy("cpu-krylov", restart=1, max_cycles=1)
        logs = []
        with self.assertRaises(LinearSolveError) as failure:
            solve_system(matrix, rhs, 1e-12, policy, solve_log=logs)
        self.assertFalse(failure.exception.report.accepted)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0].used_method, "cpu-krylov")
        value, report = solve_system(matrix, rhs, 1e-12, replace(policy, allow_direct_fallback=True))
        np.testing.assert_allclose(value, 1/np.arange(1., 11.))
        self.assertTrue(report.accepted)
        self.assertEqual(report.used_method, "cpu-direct")
        self.assertIsNotNone(report.fallback_reason)
        self.assertNotEqual(report.krylov_flag, 0)
        self.assertFalse(report.matrix_free)

    def test_original_residual_acceptance_and_zero_rhs(self):
        matrix = sparse.eye(4, format="csr")
        with self.assertRaises(LinearSolveError):
            solve_system(matrix, np.ones(4), 1e-11, LinearPolicy("cpu-krylov"), acceptance=lambda _: 1.)
        value, report = solve_system(matrix, np.zeros(4), 1e-11, LinearPolicy("cpu-krylov"))
        np.testing.assert_array_equal(value, np.zeros(4))
        self.assertEqual(report.iterations, 0)
        self.assertEqual(report.original_residual, 0)
        self.assertEqual(policy_from_config({}).method, "cpu-direct")
        for options in ({"linearBackend": "gpu-krylov"}, {"allowDirectFallback": True},
                        {"linearBackend": "cpu-krylov", "allowDirectFallback": "yes"}):
            with self.assertRaises(ValueError):
                policy_from_config(options)


if __name__ == "__main__":
    unittest.main()
