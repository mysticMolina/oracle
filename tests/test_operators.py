"""Jacob Matthew Molina. Inception: 2026-11-05.

Actual MATLAB fixture parity plus independent stencil and Dirac checks.
No MATLAB installation is required to consume the committed tiny fixture.
"""

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse
from scipy.io import loadmat

from oracle_py.matter import full_cayley_schur_step
from oracle_py.operators import (centered_operators, compatible_operators,
                                 free_dirac_block, spatial_hamiltonian)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class SpatialOperators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = loadmat(FIXTURES / "spatial_7x5_physical_c.mat", squeeze_me=True)
        cls.grid = tuple(cls.reference["grid"].astype(int))
        cls.spacing = cls.reference["spacing"]
        cls.centered = centered_operators(cls.grid, cls.spacing)
        cls.compatible = compatible_operators(cls.grid, cls.spacing)

    def assertSparseClose(self, actual, reference, tolerance=2e-12):
        error = (actual - sparse.csr_matrix(reference)).tocsr()
        self.assertLessEqual(float(np.max(np.abs(error.data), initial=0)), tolerance)

    def test_fixture_integrity(self):
        manifest = json.loads((FIXTURES / "spatial_reference.json").read_text())
        self.assertEqual(hashlib.sha256((FIXTURES / manifest["fixture"]).read_bytes()).hexdigest(),
                         manifest["fixture_sha256"])

    def test_matlab_centered_operators_and_mask(self):
        r, o = self.reference, self.centered
        for got, name in zip(o.derivatives, ["Dx", "Dy", "Dz"]):
            self.assertSparseClose(got, r[name], 1e-14)
        for name in ["gradient", "divergence", "curl"]:
            self.assertSparseClose(getattr(o, name), r[name], 1e-14)
        np.testing.assert_array_equal(o.free, r["free"].toarray().ravel())

    def test_matlab_compatible_operators_and_mask(self):
        r, o = self.reference, self.compatible
        for attribute, name in [("gradient", "G"), ("curl", "C"),
                                ("magnetic_divergence", "divB"), ("free_curl", "Cf"),
                                ("curl_curl", "fieldK"), ("free_gradient", "gf")]:
            self.assertSparseClose(getattr(o, attribute), r[name], 1e-13)
        np.testing.assert_array_equal(o.free, r["fieldFree"])
        # Tensor forward differences commute even at their terminal zero rows.
        self.assertSparseClose(o.curl @ o.gradient, sparse.csr_matrix((105, 35)), 2e-13)
        self.assertSparseClose(o.magnetic_divergence @ o.curl, sparse.csr_matrix((35, 105)), 2e-13)

    def test_matlab_hamiltonian_charge_sign_and_schur_step(self):
        r, o = self.reference, self.centered
        c = 137.035999084
        for charge, name in [(-1, "H"), (1, "Hpositive")]:
            h = spatial_hamiltonian(r["A"], o, charge, c, 1.)
            self.assertSparseClose(h, r[name])
            mask = np.tile(o.free, 4)
            hf = h[mask, :][:, mask]
            self.assertSparseClose(hf, hf.conj().T)
            self.assertEqual(h[~mask, :].nnz, 0)
        k = free_dirac_block(r["A"], o, -1, c, 1.)
        self.assertSparseClose(k, r["K"])
        got, residual = full_cayley_schur_step(r["psi0"], o.free, k, r["tau"], 1., c, 1.)
        np.testing.assert_allclose(got, r["psi1"], rtol=3e-13, atol=3e-14)
        self.assertLess(residual, 1e-13)
        # Include nonzero frozen values: only their restored global phase changes.
        phase = np.exp(-1j * r["tau"] * c * c)
        np.testing.assert_allclose(got[~o.free], r["psi0"][~o.free] * phase, atol=1e-15)

    def test_centered_plane_wave_symbol_and_boundary_column(self):
        n, h = self.grid, self.spacing
        ix, iy, _ = np.indices(n)
        kx, ky = .7, -.3
        wave = np.exp(1j * (kx * ix * h[0] + ky * iy * h[1])).ravel(order="F")
        for derivative, k, step in zip(self.centered.derivatives[:2], [kx, ky], h[:2]):
            expected = 1j * np.sin(k * step) / step * wave
            got = derivative @ wave
            np.testing.assert_allclose(got[self.centered.free], expected[self.centered.free], atol=3e-15)
            np.testing.assert_array_equal(got[~self.centered.free], 0.)
        # x=0,y=1 is a fixed column feeding x=1,y=1; no periodic wrap.
        self.assertAlmostEqual(self.centered.derivatives[0][8, 7], -1 / (2 * h[0]))
        self.assertEqual(self.centered.derivatives[0][8, 13], 0.)

    def test_invalid_geometry_and_potential_rejected(self):
        for grid, spacing in [((2, 5, 1), (1, 1, 1)), ((1, 1, 1), (1, 1, 1)),
                              ((3.5, 5, 1), (1, 1, 1)), ((7, 5, 1), (0, 1, 1)),
                              ((7, 5, 1), (1, np.nan, 1))]:
            for builder in [centered_operators, compatible_operators]:
                with self.assertRaises(ValueError):
                    builder(grid, spacing)
        a = self.reference["A"]
        for invalid in [a.astype(complex), a[:, :2], a * np.nan]:
            with self.assertRaises(ValueError):
                spatial_hamiltonian(invalid, self.centered, -1, 137., 1.)


if __name__ == "__main__":
    unittest.main()
