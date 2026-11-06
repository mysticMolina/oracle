"""Jacob Matthew Molina. Inception: 2026-11-05.

Single-species Mur/Schur source parity and independent constraint/work tests.
"""

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.io import loadmat

from oracle_py.evolution import State, coupled_mur_step
from oracle_py.matter import charge_density, dirac_current, full_matter_step
from oracle_py.maxwell import (MurHistory, gauss_project, initialize_mur,
                               longitudinal_field, mur_context, mur_field_step,
                               mur_reference_field)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class MurEvolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = loadmat(FIXTURES / "mur_7x5_physical_c.mat", simplify_cells=True)
        r = cls.reference
        cls.contexts = {case["closure"]: mur_context(r["grid"], r["spacing"], r["dt"],
                       r["physics"]["c"], r["physics"]["eps0"], case["closure"], 1e-11)
                       for case in r["cases"]}

    def assertClose(self, actual, expected, atol=4e-13):
        np.testing.assert_allclose(actual, expected, rtol=4e-12, atol=atol)

    def assertInfo(self, actual, expected):
        # Relative residuals are solver-roundoff quantities, not bitwise values.
        residual_keys = {"residual", "matterResidual", "emResidual", "gaussChange", "maxwellGaussChange"}
        self.assertEqual(set(actual), set(expected))
        for key in actual:
            if key in residual_keys:
                self.assertLess(actual[key], 1e-11, key)
                self.assertLess(expected[key], 1e-11, key)
            else:
                self.assertClose(actual[key], expected[key], 2e-11 if key == "continuityResidual" else 4e-13)

    def state(self, reference):
        return State(reference["psi"], reference["A"], reference["Y"])

    def assertState(self, actual, expected):
        for value, name in [(actual.psi, "psi"), (actual.a, "A"), (actual.y, "Y")]:
            self.assertClose(value, expected[name])
        for value, name in [(actual.mur.initial_longitudinal_y, "initialLongitudinalY"),
                            (actual.mur.initial_harmonic_y, "initialHarmonicY")]:
            self.assertClose(value, expected["mur"][name])
        previous = expected["mur"]["previousTraceY"]
        if np.size(previous):
            self.assertClose(actual.mur.previous_trace_y, previous)
        else:
            self.assertIsNone(actual.mur.previous_trace_y)

    def test_fixture_integrity(self):
        manifest = json.loads((FIXTURES / "mur_reference.json").read_text())
        self.assertEqual(hashlib.sha256((FIXTURES / manifest["fixture"]).read_bytes()).hexdigest(),
                         manifest["fixture_sha256"])

    def test_current_charge_and_actual_matter_half_step(self):
        r, p = self.reference, self.reference["physics"]
        for case in r["cases"]:
            ctx = self.contexts[case["closure"]]
            state = self.state(case["initial"])
            self.assertClose(charge_density(state.psi, p["q"], r["backgroundCharge"]), case["rho"])
            self.assertClose(dirac_current(state.psi, p["q"], p["c"]), case["current"])
            psi, y, info = full_matter_step(state.psi, state.a, state.y, ctx.matter,
                r["dt"] / 2, p["q"], p["m"], p["c"], p["hbar"], feedback=True, tolerance=1e-11)
            self.assertClose(psi, case["matter"]["psi"])
            self.assertClose(y, case["matter"]["Y"])
            self.assertClose(info["current"], case["matterInfo"]["current"])
            self.assertClose(info["continuity"], case["matterInfo"]["continuity"], 2e-11)
            self.assertLess(info["residual"], 1e-11)

    def test_mur_geometry_longitudinal_and_history(self):
        for case in self.reference["cases"]:
            ctx = self.contexts[case["closure"]]
            for actual, name in [(ctx.rows + 1, "rows"), (ctx.adjacent + 1, "adjacent"),
                                 (ctx.inactive, "inactive")]:
                np.testing.assert_array_equal(actual, case[name])
            for actual, name in [(ctx.nu, "nu"), (ctx.r, "r"),
                                 (ctx.trace.toarray(), "trace"), (ctx.left.toarray(), "left")]:
                expected = case[name].toarray() if hasattr(case[name], "toarray") else case[name]
                self.assertClose(actual, expected)
            yl = longitudinal_field(case["rho"], ctx)
            self.assertClose(yl, case["longitudinal"])
            history = initialize_mur(case["rho"], ctx)
            self.assertClose(history.initial_harmonic_y, case["initialized"]["mur"]["initialHarmonicY"])
            self.assertClose(ctx.field.free_gradient.T @ yl, case["rho"][ctx.matter.free])
            self.assertClose(ctx.field.curl @ yl, np.zeros_like(yl))
            changed_rho = .8 * case["rho"]
            reference = mur_reference_field(changed_rho, ctx, history)
            self.assertClose(ctx.field.free_gradient.T @ reference, changed_rho[ctx.matter.free])
            self.assertClose(reference[ctx.rows], yl[ctx.rows])

    def test_projection_and_field_step_boundary_work(self):
        r = self.reference
        for case in r["cases"]:
            ctx = self.contexts[case["closure"]]
            state = self.state(case["initial"])
            y, info = gauss_project(state.y, case["rho"], ctx)
            self.assertClose(y, case["projected"]["Y"])
            self.assertInfo(info, case["projectInfo"])
            self.assertClose(ctx.field.free_gradient.T @ y.ravel(order="F"), case["rho"][ctx.matter.free])
            a, y, history, info = mur_field_step(state.a, state.y, case["rho"], ctx,
                                               background_a=r["backgroundA"])
            self.assertClose(a, case["field"]["A"])
            self.assertClose(y, case["field"]["Y"])
            self.assertInfo(info, case["fieldInfo"])
            self.assertClose(history.previous_trace_y, case["field"]["mur"]["previousTraceY"])
            def energy(a, y):
                perturbation = (a-r["backgroundA"]).ravel(order="F")
                magnetic = ctx.field.curl @ perturbation
                return ctx.volume * (np.sum(y*y) / ctx.eps0 + ctx.eps0 * ctx.c**2 * np.dot(magnetic, magnetic)) / 2
            self.assertClose(energy(a, y)-energy(state.a, state.y), info["boundaryWork"], 1e-13)
            self.assertClose(info["boundaryWaveWork"]+info["boundaryReferenceWork"], info["boundaryWork"], 1e-18)

    def test_actual_symmetric_composition_and_modes_through_three_steps(self):
        r, p = self.reference, self.reference["physics"]
        for case in r["cases"]:
            ctx = self.contexts[case["closure"]]
            for output in case["outputs"]:
                state = self.state(case["initial"])
                for step in range(3):
                    state, info = coupled_mur_step(state, ctx, p["q"], p["m"], p["hbar"],
                        output["mode"], background_charge=r["backgroundCharge"], background_a=r["backgroundA"])
                    if step == 0:
                        self.assertState(state, output["one"])
                        self.assertInfo(info, output["oneInfo"])
                self.assertState(state, output["three"])
                self.assertInfo(info, output["threeInfo"])
            state, info = coupled_mur_step(self.state(case["initial"]), ctx, p["q"], p["m"], p["hbar"],
                project_gauss=False, background_charge=r["backgroundCharge"], background_a=r["backgroundA"])
            self.assertState(state, case["unprojected"])
            self.assertInfo(info, case["unprojectedInfo"])

    def test_invalid_boundaries_and_history_rejected(self):
        with self.assertRaises(ValueError):
            mur_context((7, 5, 1), (.17, .23, 1), 1., 137.)
        with self.assertRaises(ValueError):
            mur_context((7, 5, 3), (.17, .23, 1), .0001, 137.)
        with self.assertRaises(ValueError):
            mur_context((7, 5, 1), (.17, .23, 1), .0001, 137., longitudinal="mur2")
        case = self.reference["cases"][0]
        ctx = self.contexts[case["closure"]]
        state = self.state(case["initial"])
        y = state.y.copy().ravel(order="F")
        y[np.flatnonzero(ctx.inactive)[0]] = 1
        with self.assertRaises(ValueError):
            mur_field_step(state.a, y.reshape((-1, 3), order="F"), case["rho"], ctx,
                           background_a=self.reference["backgroundA"])
        broken = MurHistory(None, None, np.zeros(105))
        with self.assertRaises(ValueError):
            initialize_mur(case["rho"], ctx, broken)


if __name__ == "__main__":
    unittest.main()
