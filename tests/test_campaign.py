"""Jacob Matthew Molina. Inception: 2026-11-05.

Actual accepted-case bounded controls, exact restart and failure recovery.
Only stop_after changes between segments; the100-period target stays fixed.
"""

import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import loadmat

from oracle_py.campaign import run_campaign, source_record
from oracle_py.evolution import coupled_mur_step
from oracle_py.checkpoint import load_checkpoint, run_lock, save_checkpoint

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class CampaignTransactions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="oracle-campaign-tests-")
        cls.root = Path(cls.temporary.name)
        cls.config = json.loads((FIXTURES/"matched_B200_box26r.json").read_text())
        cls.full = cls.root/"full"
        cls.split = cls.root/"split"
        cls.full_result = run_campaign(cls.config, cls.full, stop_after=3, checkpoint_every=2)
        cls.first_result = run_campaign(cls.config, cls.split, stop_after=1, checkpoint_every=1)
        cls.split_result = run_campaign(cls.config, cls.split, resume=True, stop_after=3, checkpoint_every=2)
        cls.source = source_record()[0]
        cls.identity = cls.full_result["config"]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def copy_run(self, suffix):
        folder = self.root/suffix
        shutil.copytree(self.split, folder)
        return folder

    def test_exact_four_branch_restart_and_unchanged_target(self):
        for result in (self.full_result, self.split_result):
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["step"], 3)
            self.assertEqual(result["stepsRequested"], 75000)
            self.assertEqual(result["time"], 3*self.config["dt"])
            self.assertEqual(result["error"], "")
        with run_lock(self.full), run_lock(self.split):
            a, ma = load_checkpoint(self.full, self.identity, self.source, recover=False)
            b, mb = load_checkpoint(self.split, self.identity, self.source, recover=False)
        for left, right in zip(a, b):
            for x, y in [(left.psi, right.psi), (left.a, right.a), (left.y, right.y),
                         (left.mur.initial_harmonic_y, right.mur.initial_harmonic_y),
                         (left.mur.initial_longitudinal_y, right.mur.initial_longitudinal_y),
                         (left.mur.previous_trace_y, right.mur.previous_trace_y)]:
                if x is None:
                    self.assertIsNone(y)
                else:
                    np.testing.assert_array_equal(x, y)
        self.assertEqual(ma["totals"], mb["totals"])
        self.assertEqual(ma["info"], mb["info"])
        self.assertEqual((self.full/"coupled.csv").read_bytes(), (self.split/"coupled.csv").read_bytes())
        with (self.split/"coupled.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["step"]) for row in rows], [0, 1, 2, 3])
        self.assertGreater(float(rows[1]["continuityHalfStepOverPeakCharge"]), 0)
        self.assertGreater(float(rows[1]["chargeChangeLinfOverPeakCharge"]), 0)
        saved = json.loads((self.split/"configuration.json").read_text())["configuration"]
        self.assertEqual(saved, self.config)

    def test_native_snapshot_and_existing_analysis_reader(self):
        from analysis_suite.adapters import open_run
        reader = open_run(self.full, allow_partial=True)
        self.assertTrue(any("Python CPU parity" in note for note in reader.notes))
        self.assertEqual(reader.status, "partial")
        self.assertEqual(reader.load_snapshot(-1)["psi"].shape, (125, 125, 1, 4))
        with run_lock(self.full):
            states, _ = load_checkpoint(self.full, self.identity, self.source, recover=False)
        for branch in range(4):
            read = reader.load_snapshot(-1, branch=branch)
            # Reader uses x,y,z,C, while packed state varies x fastest.
            expected = states[branch].psi.reshape((125, 125, 1, 4), order="F")
            np.testing.assert_array_equal(read["psi"], expected)
        config = loadmat(self.full/"configuration.mat", simplify_cells=True)
        self.assertEqual(config["meta"]["solverVariant"], "oracle_python_planar_v1")

    def test_uncommitted_tail_recovery_preserves_checkpoint_prefix(self):
        folder = self.copy_run("tail")
        journal = folder/"coupled.csv"
        prefix = journal.read_bytes()
        with journal.open("ab") as stream:
            stream.write(b"uncommitted trailing row\n")
        orphan = folder/"state_000000004.mat"
        orphan.write_bytes(b"interrupted snapshot placeholder")
        with run_lock(folder):
            load_checkpoint(folder, self.identity, self.source)
        self.assertEqual(journal.read_bytes(), prefix)
        self.assertFalse(orphan.exists())
        recovered = list((folder/"recovery").glob("state_000000004_*.mat"))
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].read_bytes(), b"interrupted snapshot placeholder")

    def test_missing_or_changed_committed_outputs_rejected(self):
        cases = [("journal", "coupled.csv", "missing"),
                 ("short", "coupled.csv", "short"), ("corrupt", "coupled.csv", "corrupt"),
                 ("snapshot", "state_000000003.mat", "missing"),
                 ("configuration", "configuration.mat", "missing"),
                 ("source", "source_snapshot.zip", "corrupt")]
        for suffix, name, operation in cases:
            folder = self.copy_run(suffix)
            target = folder/name
            if operation == "missing":
                target.unlink()
            elif operation == "short":
                target.write_bytes(target.read_bytes()[:10])
            else:
                content = target.read_bytes()
                target.write_bytes(bytes([content[0] ^ 1])+content[1:])
            with run_lock(folder), self.assertRaises((ValueError, FileNotFoundError)):
                load_checkpoint(folder, self.identity, self.source)

    def test_failed_atomic_replace_keeps_previous_checkpoint(self):
        folder = self.copy_run("atomic_failure")
        original = (folder/"checkpoint.npz").read_bytes()
        with run_lock(folder):
            states, metadata = load_checkpoint(folder, self.identity, self.source, recover=False)
            with patch("oracle_py.checkpoint.os.replace", side_effect=OSError("injected interrupted replace")):
                with self.assertRaises(OSError):
                    save_checkpoint(folder, states, metadata["step"], metadata["totals"], metadata["info"],
                        metadata["initial"], metadata["status"], self.identity, self.source, metadata["artifacts"])
        self.assertEqual((folder/"checkpoint.npz").read_bytes(), original)
        self.assertEqual(list(folder.glob(".checkpoint.npz.*.partial")), [])

    def test_source_config_drift_rejected_before_mutation(self):
        folder = self.copy_run("drift")
        journal = folder/"coupled.csv"
        with journal.open("ab") as stream:
            stream.write(b"must remain for diagnosis\n")
        before = journal.read_bytes()
        with run_lock(folder):
            for config, source in [("different", self.source), (self.identity, "different")]:
                with self.assertRaises(ValueError):
                    load_checkpoint(folder, config, source)
        self.assertEqual(journal.read_bytes(), before)
        with self.assertRaises(ValueError):
            run_campaign(dict(self.config, B=201.), folder, resume=True, stop_after=4)
        with self.assertRaises(ValueError):
            run_campaign(dict(self.config, periods=101), folder, resume=True, stop_after=4)
        self.assertEqual(journal.read_bytes(), before)

    def test_nonfinite_candidate_never_poisoned_accepted_checkpoint(self):
        folder = self.copy_run("nonfinite_candidate")
        def corrupt(*args, **kwargs):
            state, info = coupled_mur_step(*args, **kwargs)
            if kwargs["mode"] == "both":
                y = state.y.copy()
                y[0, 0] = np.nan
                state = replace(state, y=y)
            return state, info
        with patch("oracle_py.campaign.coupled_mur_step", side_effect=corrupt):
            result = run_campaign(self.config, folder, resume=True, stop_after=4)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["lastAcceptedStep"], 3)
        self.assertEqual(result["lastAttemptedStep"], 4)
        with np.load(folder/"checkpoint.npz", allow_pickle=False) as record:
            self.assertTrue(np.all(np.isfinite(record["Y"])))
            self.assertEqual(json.loads(str(record["metadata"].item()))["step"], 3)
        self.assertEqual((folder/"coupled.csv").read_bytes(), (self.split/"coupled.csv").read_bytes())
        forensic = loadmat(folder/result["rejectedCandidate"]["file"], simplify_cells=True)
        self.assertFalse(forensic["candidate"]["accepted"])
        self.assertTrue(np.isnan(forensic["allStates"][0]["Y"]).any())

    def test_em_control_and_mur_history_validated_before_publication(self):
        for defect in ("em_y", "history_nan", "history_shape"):
            folder = self.copy_run(defect)
            def corrupt(*args, **kwargs):
                state, info = coupled_mur_step(*args, **kwargs)
                if kwargs["mode"] == "EM":
                    if defect == "em_y":
                        y = state.y.copy()
                        y[0, 0] = np.nan
                        state = replace(state, y=y)
                    else:
                        history = state.mur.initial_harmonic_y.copy()
                        if defect == "history_nan":
                            history[0] = np.nan
                        else:
                            history = history[:-1]
                        state = replace(state, mur=replace(state.mur, initial_harmonic_y=history))
                return state, info
            with patch("oracle_py.campaign.coupled_mur_step", side_effect=corrupt):
                result = run_campaign(self.config, folder, resume=True, stop_after=4)
            self.assertEqual(result["status"], "stopped")
            self.assertEqual(result["lastAcceptedStep"], 3)
            self.assertEqual(result["lastAttemptedStep"], 4)
            self.assertEqual((folder/"coupled.csv").read_bytes(), (self.split/"coupled.csv").read_bytes())
            self.assertFalse((folder/"state_000000004.mat").exists())
            self.assertTrue((folder/result["rejectedCandidate"]["file"]).exists())
            with np.load(folder/"checkpoint.npz", allow_pickle=False) as record:
                self.assertTrue(np.all(np.isfinite(record["Y"])))
                self.assertTrue(np.all(np.isfinite(record["initial_harmonic_y"])))
                self.assertEqual(json.loads(str(record["metadata"].item()))["step"], 3)

    def test_interrupted_stopped_commit_preserves_each_forensic_attempt(self):
        folder = self.copy_run("interrupted_rejection")
        with patch("oracle_py.campaign._screen", side_effect=ArithmeticError("injected rejection")), \
             patch("oracle_py.campaign.save_checkpoint", side_effect=OSError("interrupted stopped commit")):
            with self.assertRaises(OSError):
                run_campaign(self.config, folder, resume=True, stop_after=4)
        orphan = folder/"failed_candidate_000000004.mat"
        original = orphan.read_bytes()
        with patch("oracle_py.campaign._screen", side_effect=ArithmeticError("injected rejection")):
            result = run_campaign(self.config, folder, resume=True, stop_after=4)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["lastAcceptedStep"], 3)
        self.assertEqual(orphan.read_bytes(), original)
        self.assertNotEqual(result["rejectedCandidate"]["file"], orphan.name)
        self.assertEqual(len(list(folder.glob("failed_candidate_*.mat"))), 2)
        with run_lock(folder), self.assertRaisesRegex(ValueError, "stopped"):
            load_checkpoint(folder, self.identity, self.source)

    def test_lock_missing_checkpoint_and_screen_stopped_recovery(self):
        with run_lock(self.root/"locked"):
            with self.assertRaises(RuntimeError):
                with run_lock(self.root/"locked"):
                    pass
        folder = self.copy_run("no_checkpoint")
        (folder/"checkpoint.npz").unlink()
        with self.assertRaises(ValueError):
            run_campaign(self.config, folder, resume=True, stop_after=4)
        folder = self.copy_run("stopped")
        with patch("oracle_py.campaign._screen", side_effect=ArithmeticError("injected screen stop")):
            result = run_campaign(self.config, folder, resume=True, stop_after=4)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["step"], 3)
        self.assertEqual(result["lastAcceptedStep"], 3)
        self.assertEqual(result["lastAttemptedStep"], 4)
        self.assertEqual(result["rejectedCandidate"]["accepted"], False)
        self.assertTrue((folder/result["rejectedCandidate"]["file"]).exists())
        self.assertFalse((folder/"state_000000004.mat").exists())
        self.assertEqual((folder/"coupled.csv").read_bytes(), (self.split/"coupled.csv").read_bytes())
        with run_lock(folder), self.assertRaises(ValueError):
            load_checkpoint(folder, self.identity, self.source)


if __name__ == "__main__":
    unittest.main()
