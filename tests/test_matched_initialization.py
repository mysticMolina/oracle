"""Jacob Matthew Molina. Inception: 2026-11-05.

Exact box26r B200 CPU case JSON against pinned current MATLAB source.
One-step implementation parity does not certify production convergence.
"""

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.io import loadmat

from oracle_py.diagnostics import diagnostics, boundary_probes
from oracle_py.evolution import State, coupled_mur_step
from oracle_py.initialization import matched_problem, prepare_matched_campaign

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class MatchedInitialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configuration = json.loads((FIXTURES / "matched_B200_box26r.json").read_text())
        cls.reference = loadmat(FIXTURES / "matched_B200_box26r.mat", simplify_cells=True)
        cls.problem, cls.receipt = prepare_matched_campaign(cls.configuration)
        p = cls.problem
        cls.next, cls.info = coupled_mur_step(p.initial, p.context,
            background_charge=p.background_charge, background_a=p.background_a)

    def assertClose(self, actual, expected, atol=3e-12):
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=atol)

    def test_fixture_and_case_identity(self):
        manifest = json.loads((FIXTURES / "matched_reference.json").read_text())
        for key, digest in [("fixture", "fixture_sha256"), ("configuration", "configuration_sha256")]:
            self.assertEqual(hashlib.sha256((FIXTURES / manifest[key]).read_bytes()).hexdigest(), manifest[digest])
        self.assertEqual(self.configuration, json.loads(self.reference["configText"]))
        self.assertEqual(self.configuration, self.problem.configuration)
        self.assertEqual(self.problem.context.grid, (125, 125, 1))
        self.assertEqual(self.configuration["periods"], 100)
        self.assertEqual(self.configuration["steps"], 75000)
        self.assertClose(self.problem.context.dt, self.configuration["dt"], 0.)

    def test_actual_matched_initialization_and_guards(self):
        p, r = self.problem, self.reference
        for actual, name in [(p.initial.psi, "psi"), (p.initial.a, "A"), (p.initial.y, "Y")]:
            self.assertClose(actual, r["state0"][name])
        self.assertClose(p.context.spacing, r["spacing"])
        self.assertClose(p.origin, r["origin"])
        self.assertClose(p.background_a, r["backgroundA"])
        self.assertClose(p.background_b, r["backgroundB"])
        self.assertClose(p.background_charge, r["backgroundCharge"])
        self.assertClose(self.receipt["velocity"], r["velocity"])
        self.assertClose(self.receipt["betaMeasured"], r["betaMeasured"])
        self.assertLess(abs(self.receipt["betaMeasured"]-self.configuration["beta_target"]), 1e-8)
        self.assertLess(self.receipt["initialDiagnostic"]["boundaryMassFraction"], 1e-6)
        self.assertClose(self.receipt["initialDiagnostic"]["norm"], self.configuration["norm"])
        np.testing.assert_array_equal(p.initial.psi[~p.context.matter.free], 0.)
        for name, value in p.resolution_receipt.items():
            self.assertClose(value, r["resolutionReceipt"][name])
        for name, value in p.initial_gauss_projection.items():
            self.assertClose(value, r["initialGaussProjection"][name])

    def test_actual_case_one_step_and_all_single_species_diagnostics(self):
        p, r = self.problem, self.reference
        for actual, name in [(self.next.psi, "psi"), (self.next.a, "A"), (self.next.y, "Y")]:
            self.assertClose(actual, r["state1"][name])
        for name, value in self.info.items():
            self.assertClose(value, r["stepInfo"][name], 1e-10 if name == "continuityResidual" else 3e-12)
        for state, prefix in [(p.initial, "initial"), (self.next, "final")]:
            for units, suffix in [(None, "Diagnostic"), ("physical-compton", "PhysicalDiagnostic")]:
                got = diagnostics(state, p.context, p.origin, background_charge=p.background_charge,
                    background_a=p.background_a, background_b=p.background_b, units=units)
                expected = r[prefix+suffix]
                self.assertEqual(set(got), set(expected))
                for key in got:
                    self.assertClose(got[key], expected[key])
        # This is the measured source screen, not a fabricated zero or pass.
        self.assertGreater(self.info["continuityResidual"], 100.)

    def test_source_boundary_probe_fixture(self):
        reference = json.loads((FIXTURES / "probes_B200_box26r.json").read_text())
        manifest = json.loads((FIXTURES / "probe_reference.json").read_text())
        self.assertEqual(hashlib.sha256((FIXTURES / manifest["fixture"]).read_bytes()).hexdigest(), manifest["fixture_sha256"])
        p = self.problem
        for state, name in [(p.initial, "initial"), (self.next, "final")]:
            values = boundary_probes(state, p.context, p.background_a, p.background_b, p.background_charge)
            self.assertEqual(set(values), set(reference[name]))
            for key in values:
                self.assertClose(values[key], reference[name][key])

    def test_zero_norm_diagnostic_definitions(self):
        p = self.problem
        state = State(np.zeros_like(p.initial.psi), p.background_a.copy(), np.zeros_like(p.initial.y))
        got = diagnostics(state, p.context, p.origin, background_a=p.background_a, background_b=p.background_b)
        for key in ("norm", "charge", "gaussReferenceNorm", "gaussRelativeDefined",
                    "gaussRelative", "highKFraction", "kineticPerNormDefined", "kineticPerNorm"):
            self.assertEqual(got[key], 0.)
        self.assertNotIn("xMean", got)
        self.assertNotIn("freeNegativeFraction", got)

    def test_invalid_preparation_is_rejected_before_evolution(self):
        base = self.configuration
        for patch in [{"radius": -1}, {"norm": 0}, {"steps": -1}, {"steps": 2.5}, {"grid": [9, 9, 3]},
                      {"boundary": {"type": "mur2"}}, {"linearBackend": "gpu-krylov"},
                      {"resolution": {"targetCellsPerMagneticLength": .01}},
                      {"resolution": {"targetCellsPerMagneticLength": 1, "relativeTolerance": 1}}]:
            with self.assertRaises(ValueError):
                matched_problem(dict(base, **patch))
        with self.assertRaises(ValueError):
            prepare_matched_campaign(dict(base, beta_target=0.))


if __name__ == "__main__":
    unittest.main()
