import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

from mikasa_consolidation import ConsolidationSimulator, FiniteStrainConsolidation
from mikasa_consolidation.finite_strain import _tridiagonal_batch


class CudaBoundaryTests(unittest.TestCase):
    def test_explicit_cuda_request_does_not_fall_back_without_cupy(self):
        with patch.dict(sys.modules, {"cupy": None}):
            for cls in (ConsolidationSimulator, FiniteStrainConsolidation):
                with self.subTest(model=cls.__name__), self.assertRaisesRegex(RuntimeError, "CUDA requires CuPy"):
                    cls(nx=3, nz=4, total_time=0.1, device="cuda")


@unittest.skipUnless(os.environ.get("MIKASA_TEST_CUDA") == "1", "set MIKASA_TEST_CUDA=1 on an NVIDIA GPU")
class CudaHardwareTests(unittest.TestCase):
    def test_thomas_kernel_matches_cpu_and_independent_matrix_residual(self):
        from mikasa_consolidation.cuda import make_tridiagonal_solver
        solve = make_tridiagonal_solver()
        rng = np.random.default_rng(42)
        for shape in ((1, 1), (2, 3), (9, 5), (7, 17)):
            lower = rng.uniform(-0.5, 0, shape)
            upper = rng.uniform(-0.5, 0, shape)
            lower[0] = upper[-1] = 0
            diag = 1 + np.abs(lower) + np.abs(upper)
            rhs = rng.normal(size=shape)
            actual = solve(lower, diag, upper, rhs)
            np.testing.assert_allclose(actual, _tridiagonal_batch(lower, diag, upper, rhs),
                                       rtol=1e-12, atol=1e-12)
            for col in range(shape[1]):
                matrix = (np.diag(diag[:, col]) + np.diag(lower[1:, col], -1)
                          + np.diag(upper[:-1, col], 1))
                np.testing.assert_allclose(matrix @ actual[:, col], rhs[:, col], rtol=1e-12, atol=1e-12)

    def test_states_and_reset_match_cpu_on_one_and_two_dimensional_grids(self):
        for cls, grids in ((FiniteStrainConsolidation, ((1, 7), (5, 9))),
                           (ConsolidationSimulator, ((2, 7), (8, 5)))):
            for nx, nz in grids:
                with self.subTest(model=cls.__name__, grid=(nx, nz)):
                    kwargs = dict(nx=nx, nz=nz, dt=0.02, total_time=0.1)
                    cpu = cls(**kwargs)
                    gpu = cls(device="cuda", **kwargs)
                    initial = gpu.get_current_state()
                    cpu.run_simulation(verbose=False)
                    gpu.run_simulation(verbose=False)
                    for expected, actual in zip(cpu.get_current_state(), gpu.get_current_state()):
                        self.assertTrue(np.isfinite(actual).all())
                        self.assertEqual(actual.dtype, np.float64)
                        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
                    gpu.reset()
                    for expected, actual in zip(initial, gpu.get_current_state()):
                        np.testing.assert_array_equal(actual, expected)

    def test_non_convergence_is_reported_on_cuda(self):
        simulator = FiniteStrainConsolidation(nx=2, nz=4, total_time=1, dt=1,
                                              max_iterations=1, max_splits=1, device="cuda")
        with self.assertRaises(RuntimeError):
            simulator.run_simulation(verbose=False)
