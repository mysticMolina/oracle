"""Jacob Matthew Molina. Inception: 2026-11-05. First parity invariants."""

import unittest

import numpy as np

from oracle_py.core import dirac_matrices, pack_field, rest_energy_step, unpack_field


class CoreParity(unittest.TestCase):
    def test_dirac_clifford_relations(self):
        alpha, beta = dirac_matrices()
        identity = np.eye(4)
        for i in range(3):
            np.testing.assert_allclose(alpha[i] @ alpha[i], identity, atol=1e-15)
            np.testing.assert_allclose(alpha[i] @ beta + beta @ alpha[i], 0, atol=1e-15)
            for j in range(i):
                np.testing.assert_allclose(alpha[i] @ alpha[j] + alpha[j] @ alpha[i], 0, atol=1e-15)
        np.testing.assert_array_equal(np.diag(beta), [1, 1, -1, -1])

    def test_matlab_x_fastest_pack_and_roundtrip(self):
        grid = (3, 2, 2)
        public = np.empty((grid[1], grid[0], grid[2], 4), dtype=np.complex128)
        for y in range(grid[1]):
            for x in range(grid[0]):
                for z in range(grid[2]):
                    public[y, x, z, :] = 100 * z + 10 * y + x + 1j * np.arange(4)
        packed = pack_field(public)
        np.testing.assert_array_equal(packed[:, 0].real,
                                      [0, 1, 2, 10, 11, 12, 100, 101, 102, 110, 111, 112])
        np.testing.assert_array_equal(unpack_field(packed, grid), public)

    def test_rest_map_norm_and_reversal(self):
        rng = np.random.default_rng(711)
        psi = rng.normal(size=(9, 4)) + 1j * rng.normal(size=(9, 4))
        advanced = rest_energy_step(psi, 0.17, 1.3, 9.0, 0.8)
        np.testing.assert_allclose(np.sum(abs(advanced)**2), np.sum(abs(psi)**2), rtol=1e-14)
        np.testing.assert_allclose(rest_energy_step(advanced, -0.17, 1.3, 9.0, 0.8),
                                   psi, rtol=1e-14, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
