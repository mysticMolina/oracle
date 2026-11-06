"""Jacob Matthew Molina. Inception: 2026-11-05. Independent Cayley check."""

import unittest

import numpy as np
from scipy import sparse

from oracle_py.matter import full_cayley_schur_step


class FullMatterSchur(unittest.TestCase):
    def test_against_direct_block_cayley_and_reversal(self):
        rng = np.random.default_rng(9001)
        psi = rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))
        free = np.array([True, False, True])
        k = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        tau, mass, c, hbar = 0.07, 1.2, 2.0, 0.9
        got, residual = full_cayley_schur_step(psi, free, k, tau, mass, c, hbar)
        self.assertLess(residual, 1e-13)
        u = psi[free, :2].reshape(-1, order="F")
        v = psi[free, 2:].reshape(-1, order="F")
        z = np.zeros_like(k)
        shifted_h = np.block([[z, k], [k.conj().T, -2 * mass * c * c * np.eye(4)]])
        identity = np.eye(8)
        scale = tau / (2 * hbar)
        direct = np.linalg.solve(identity + 1j * scale * shifted_h,
                                 (identity - 1j * scale * shifted_h) @ np.r_[u, v])
        direct *= np.exp(-1j * tau * mass * c * c / hbar)
        expected = psi.copy() * np.exp(-1j * tau * mass * c * c / hbar)
        expected[free, :2] = direct[:4].reshape((2, 2), order="F")
        expected[free, 2:] = direct[4:].reshape((2, 2), order="F")
        np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(np.linalg.norm(got), np.linalg.norm(psi), rtol=1e-13)
        back, _ = full_cayley_schur_step(got, free, k, -tau, mass, c, hbar)
        np.testing.assert_allclose(back, psi, rtol=1e-13, atol=1e-13)
        sparse_got, _ = full_cayley_schur_step(psi, free, sparse.csr_matrix(k),
                                              tau, mass, c, hbar)
        np.testing.assert_allclose(sparse_got, got, rtol=1e-13, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
