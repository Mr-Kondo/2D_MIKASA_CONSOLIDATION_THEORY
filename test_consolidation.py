import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from consolidation import ConsolidationSimulator, is_negligible_field
from Mikasa import MikasaConsolidationSimulator
from Terzaghi import TerzaghiConsolidationSimulator


class ConsolidationTests(unittest.TestCase):
    def test_pressure_decays_toward_analytic_2d_solution_with_grid_refinement(self):
        errors = []
        for nodes in (11, 21, 41):
            simulator = ConsolidationSimulator(nx=nodes, nz=nodes, dt=0.001, total_time=0.4)
            x = np.linspace(0, simulator.length, nodes)
            z = np.linspace(0, simulator.depth, nodes)
            kx, kz = np.pi / simulator.length, np.pi / (2 * simulator.depth)
            vertical = np.sin(kz * z)[:, None]
            simulator.u = 50 * vertical * (1 + 0.2 * np.cos(kx * x))
            before = simulator.u.max()
            simulator.run_simulation(verbose=False)
            expected = 50 * vertical * np.exp(-kz**2 * 0.4) * (
                1 + 0.2 * np.cos(kx * x) * np.exp(-kx**2 * 0.4))
            errors.append(np.max(np.abs(simulator.u - expected)))
            self.assertLess(simulator.u.max(), before)
        self.assertLess(errors[-1], 0.001)
        self.assertGreater(errors[0] / errors[1], 3.9)
        self.assertGreater(errors[1] / errors[2], 3.9)

    def test_time_refinement_converges_to_semidiscrete_solution(self):
        errors = []
        for step in (0.05, 0.025, 0.0125):
            simulator = ConsolidationSimulator(nx=11, nz=11, dt=step, total_time=0.4)
            z = np.linspace(0, simulator.depth, simulator.nz)
            initial = 50 * np.sin(np.pi * z / (2 * simulator.depth))[:, None]
            simulator.u = np.broadcast_to(initial, (simulator.nz, simulator.nx)).copy()
            rate = -4 * np.sin(np.pi / (4 * (simulator.nz - 1)))**2 / simulator.dz**2
            expected = initial * np.exp(rate * simulator.total_time)
            simulator.run_simulation(verbose=False)
            errors.append(np.max(np.abs(simulator.u - expected)))
        self.assertGreater(errors[0] / errors[1], 3.9)
        self.assertGreater(errors[1] / errors[2], 3.9)

    def test_very_slow_vertical_mode_survives_many_small_time_steps(self):
        simulator = ConsolidationSimulator(nx=11, nz=11, depth=1e8, dt=1e10, total_time=1e12)
        z = np.linspace(0, simulator.depth, simulator.nz)
        initial = 50 * np.sin(np.pi * z / (2 * simulator.depth))[:, None]
        simulator.u = np.broadcast_to(initial, (simulator.nz, simulator.nx)).copy()
        rate = -4 * np.sin(np.pi / (4 * (simulator.nz - 1)))**2 / simulator.dz**2
        expected = np.broadcast_to(initial * np.exp(rate * simulator.total_time), simulator.u.shape)
        simulator.run_simulation(verbose=False)
        np.testing.assert_allclose(simulator.u, expected, atol=1e-11, rtol=1e-12)

    def test_fully_decayed_mode_is_finite_without_nan(self):
        simulator = ConsolidationSimulator(length=1, depth=1, nx=2, nz=2, dt=0.25, total_time=50)
        simulator.u[1] = [1, 0]
        simulator.run_simulation(verbose=False)
        self.assertTrue(np.isfinite(simulator.u).all())
        self.assertGreaterEqual(simulator.u.min(), 0)
        self.assertLess(simulator.u.max(), 1e-12)

    def test_eigenmode_evaluation_matches_direct_crank_nicolson_including_boundaries(self):
        for nx, nz, dt, duration in ((2, 2, 0.2, 0.3), (5, 4, 0.07, 0.2),
                                     (5, 4, 10, 2), (5, 4, 10, 5)):
            with self.subTest(nx=nx, nz=nz, dt=dt, duration=duration):
                simulator = ConsolidationSimulator(nx=nx, nz=nz, dt=dt, total_time=duration)
                size = nx * (nz - 1)
                operator = np.zeros((size, size))
                for j in range(1, nz):
                    for i in range(nx):
                        row = (j - 1) * nx + i
                        operator[row, row] = -2 / simulator.dx**2 - 2 / simulator.dz**2
                        for neighbor in (i - 1 if i else 1, i + 1 if i < nx - 1 else nx - 2):
                            operator[row, (j - 1) * nx + neighbor] += 1 / simulator.dx**2
                        for neighbor in (j - 1, j + 1 if j < nz - 1 else nz - 2):
                            if neighbor:
                                operator[row, (neighbor - 1) * nx + i] += 1 / simulator.dz**2
                identity = np.eye(size)
                before = simulator.u.copy()
                expected = before[1:].ravel()
                for _ in range(simulator.nt):
                    expected = np.linalg.solve(identity - 0.5 * simulator.dt * operator,
                                               (identity + 0.5 * simulator.dt * operator) @ expected)
                simulator.run_simulation(verbose=False)
                np.testing.assert_allclose(simulator.u[1:].ravel(), expected, atol=1e-11, rtol=1e-12)
                np.testing.assert_array_equal(simulator.u[0], 0)
                self.assertGreaterEqual(simulator.u.min(), -1e-11)
                self.assertLessEqual(simulator.u.max(), before.max() + 1e-11)
                self.assertAlmostEqual(simulator.nt * simulator.dt, duration)

    def test_strip_load_matches_integrated_line_load_and_surface_limits(self):
        simulator = ConsolidationSimulator(nx=9, nz=5, load_width=5, total_time=1)
        np.testing.assert_allclose(simulator.total_stress[0], [0, 0, 50, 100, 100, 100, 50, 0, 0], atol=1e-12)
        np.testing.assert_allclose(simulator.total_stress, simulator.total_stress[:, ::-1], atol=1e-12)
        sources = np.linspace(-2.5, 2.5, 10001)
        x, z = 1.25, 2.5
        reference = np.trapezoid(200 / np.pi * z**3 / ((x - sources)**2 + z**2)**2, sources)
        self.assertAlmostEqual(simulator.total_stress[2, 5], reference, places=6)

    def test_fully_drained_uniform_load_has_fixed_base_and_correct_surface_settlement(self):
        simulator = ConsolidationSimulator(length=2, depth=1, nx=5, nz=5, E=1000,
                                           load_width=1e9, total_time=1e5)
        settlement = simulator.run_simulation(verbose=False)
        z = np.linspace(0, simulator.depth, simulator.nz)
        expected = (simulator.depth - z) * simulator.load_intensity / simulator.D
        np.testing.assert_allclose(settlement, np.broadcast_to(expected[:, None], settlement.shape), atol=1e-12)
        np.testing.assert_array_equal(simulator.disp_x, 0)
        self.assertAlmostEqual(simulator.get_consolidation_degree(), 100)

    def test_zero_load_is_finite_and_reset_restores_state(self):
        for simulator_type in (MikasaConsolidationSimulator, TerzaghiConsolidationSimulator):
            with self.subTest(simulator=simulator_type.__name__):
                simulator = simulator_type(nx=4, nz=4, load_intensity=0, total_time=0.1)
                with contextlib.redirect_stdout(io.StringIO()):
                    simulator.run_simulation()
                for field in simulator.get_current_state():
                    np.testing.assert_array_equal(field, 0)
                self.assertEqual(simulator.get_consolidation_degree(), 100)
        simulator = ConsolidationSimulator(nx=5, nz=5, total_time=0.1)
        initial = simulator.get_current_state()
        simulator.run_simulation(verbose=False)
        simulator.reset()
        for actual, expected in zip(simulator.get_current_state(), initial):
            np.testing.assert_array_equal(actual, expected)
        state = simulator.get_current_state()
        state[2][:] = -1
        self.assertGreaterEqual(simulator.u.min(), 0)
        self.assertEqual(simulator.get_consolidation_degree(), 0)

    def test_impermeable_material_retains_pressure_and_failures_are_not_clipped(self):
        simulator = ConsolidationSimulator(nx=5, nz=5, cv=0, total_time=10)
        before = simulator.u.copy()
        simulator.run_simulation(verbose=False)
        np.testing.assert_allclose(simulator.u, before, atol=1e-12)
        simulator.u[2, 2] = np.nan
        with self.assertRaises(FloatingPointError):
            simulator.run_simulation(verbose=False)

    def test_subnormal_duration_still_advances_a_positive_time_step(self):
        simulator = ConsolidationSimulator(nx=2, nz=2, cv=0, total_time=float.fromhex("0x0.0000000000001p-1022"))
        self.assertGreater(simulator.dt, 0)
        self.assertEqual(simulator.nt * simulator.dt, simulator.total_time)
        self.assertTrue(np.isfinite(simulator.run_simulation(verbose=False)).all())

    def test_invalid_inputs_are_rejected_at_boundary(self):
        cases = {"length": (0, np.nan, np.inf, "a", None), "depth": (-1, np.inf),
                 "nx": (1, 2.5, True), "nz": (1, 3.0), "cv": (-1, np.nan, np.inf, None),
                 "E": (0, np.inf), "dt": (0, np.nan, np.inf, True), "total_time": (0, np.inf),
                 "load_width": (0, np.nan), "load_intensity": (-1, np.nan, True),
                 "poisson_ratio": (-0.1, 0.5, np.nan, "0.3", True),
                 "device": ("cuda", "mps", "mlx")}
        for parameter, values in cases.items():
            for value in values:
                with self.subTest(parameter=parameter, value=value), self.assertRaises(ValueError):
                    ConsolidationSimulator(**{parameter: value})

    def test_entrypoints_and_headless_plots_use_real_updated_solver(self):
        repository = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
            for script in ("Mikasa.py", "Terzaghi.py"):
                subprocess.run([sys.executable, str(repository / script)], cwd=directory,
                               env=environment, check=True, capture_output=True, text=True, timeout=60)
            for name in ("mikasa_consolidation_result.png", "terzaghi_consolidation_result.png",
                         "terzaghi_mikasa_comparison.png"):
                self.assertEqual(plt.imread(Path(directory) / name).ndim, 3)
        self.assertEqual(plt.get_fignums(), [])

    def test_negligible_field_detection_uses_the_reference_scale(self):
        self.assertTrue(is_negligible_field(np.zeros((3, 4)), 100))
        self.assertTrue(is_negligible_field(np.full((3, 4), 1e-38), 100))
        self.assertFalse(is_negligible_field(np.full((3, 4), 1.0), 100))
        self.assertFalse(is_negligible_field(np.full((3, 4), 1e-6), 100))

    def test_time_step_limit_keeps_worst_mode_amplification_nonnegative(self):
        for nx, nz, dt in ((8, 5, 1e6), (2, 2, 10.0), (51, 26, 0.01)):
            with self.subTest(nx=nx, nz=nz, dt=dt):
                simulator = ConsolidationSimulator(nx=nx, nz=nz, dt=dt, total_time=1.0)
                half_step = 0.5 * simulator.dt * simulator.cv * (
                    4 / simulator.dx**2 + 4 / simulator.dz**2)
                self.assertGreaterEqual(half_step, -1.0)
                self.assertLessEqual(simulator.dt, 0.5 / (simulator.cv * (
                    1 / simulator.dx**2 + 1 / simulator.dz**2)) * (1 + 1e-12))

    def test_plot_results_renders_both_live_and_dissipated_fields(self):
        for total_time in (0.1, 1e5):
            with self.subTest(total_time=total_time), tempfile.TemporaryDirectory() as directory:
                simulator = ConsolidationSimulator(nx=5, nz=5, total_time=total_time)
                simulator.run_simulation(verbose=False)
                simulator.plot_results(save_path=Path(directory) / "field.png", show=False, dpi=30)
                self.assertTrue((Path(directory) / "field.png").is_file())
                self.assertEqual(plt.get_fignums(), [])

    def test_headless_backend_does_not_warn_when_showing(self):
        simulator = ConsolidationSimulator(nx=4, nz=4, total_time=0.1)
        simulator.run_simulation(verbose=False)
        with warnings.catch_warnings(record=True) as caught, tempfile.TemporaryDirectory() as directory:
            warnings.simplefilter("always")
            simulator.plot_results(save_path=Path(directory) / "shown.png", show=True, dpi=30)
        non_interactive = [w for w in caught if "non-interactive" in str(w.message)]
        self.assertEqual(non_interactive, [])

    def test_plot_is_closed_when_output_directory_is_missing(self):
        simulator = ConsolidationSimulator(nx=3, nz=3, total_time=0.1)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                simulator.plot_results(save_path=Path(directory) / "missing" / "figure.png", show=False, dpi=30)
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
