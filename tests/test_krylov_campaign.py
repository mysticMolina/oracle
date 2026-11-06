"""Jacob Matthew Molina. Inception: 2026-11-05.

Unchanged accepted B200 launch, explicit CPU Krylov prefix and exact restart.
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import loadmat

from oracle_py.campaign import run_campaign, source_record
from oracle_py.checkpoint import load_checkpoint, run_lock

FIXTURES = Path(__file__).resolve().parents[1]/"fixtures"


class KrylovCampaign(unittest.TestCase):
    def test_accepted_prefix_matches_direct_and_restarts_exactly(self):
        config = json.loads((FIXTURES/"matched_B200_box26r.json").read_text())
        with tempfile.TemporaryDirectory(prefix="oracle-krylov-") as directory:
            root = Path(directory)
            direct = run_campaign(config, root/"direct", stop_after=3)
            config["linearBackend"] = "cpu-krylov"
            krylov = run_campaign(config, root/"krylov", stop_after=3)
            first = run_campaign(config, root/"split", stop_after=1)
            resumed = run_campaign(config, root/"split", resume=True, stop_after=3)
            source = source_record()[0]
            states, metadata = [], []
            for name, result in (("direct", direct), ("krylov", krylov), ("split", resumed)):
                self.assertEqual(result["status"], "partial")
                self.assertEqual(result["stepsRequested"], 75000)
                self.assertEqual(result["step"], 3)
                self.assertEqual(result["diagnostics"]["linearSolveCount"], 8)
                self.assertEqual(result["diagnostics"]["directFallbackCount"], 0)
                with run_lock(root/name):
                    current, info = load_checkpoint(root/name, result["config"], source, recover=False)
                states.append(current)
                metadata.append(info)
            for d, k, split in zip(*states):
                for name in ("psi", "a", "y"):
                    np.testing.assert_allclose(getattr(k, name), getattr(d, name), atol=3e-10, rtol=3e-10)
                    np.testing.assert_array_equal(getattr(k, name), getattr(split, name))
                for name in ("initial_longitudinal_y", "initial_harmonic_y", "previous_trace_y"):
                    np.testing.assert_array_equal(getattr(k.mur, name), getattr(split.mur, name))
            np.testing.assert_allclose(states[1][0].psi, states[0][0].psi, atol=3e-10, rtol=3e-10)
            self.assertEqual((root/"krylov"/"coupled.csv").read_bytes(), (root/"split"/"coupled.csv").read_bytes())
            self.assertEqual(metadata[1]["info"], metadata[2]["info"])
            for result in (krylov, resumed):
                self.assertEqual(result["linearSolver"]["device"], "cpu")
                self.assertEqual(result["linearSolver"]["poissonBackend"], "cpu-direct")
                self.assertGreater(result["diagnostics"]["krylovIterations"], 0)
                self.assertLessEqual(result["diagnostics"]["maxOriginalSolveResidual"], 1e-11)
                self.assertTrue(all(item["used_method"] == "cpu-krylov" and item["matrix_free"]
                                    for item in result["lastAttemptedLinearSolves"]))
            # Compare first coupled state directly with the existing actual MATLAB fixture.
            one = loadmat(root/"split"/"state_000000001.mat", simplify_cells=True)["snapshot"]
            matlab = loadmat(FIXTURES/"matched_B200_box26r.mat", simplify_cells=True)
            # Fixture uses packed [M,C]; public snapshots retain y,x,z,C.
            for name in ("psi", "A", "Y"):
                expected = matlab["state1"][name].reshape((125, 125, 1, -1), order="F").transpose(1, 0, 2, 3).squeeze()
                np.testing.assert_allclose(one[name], expected, atol=3e-10, rtol=3e-10)


if __name__ == "__main__":
    unittest.main()
