import contextlib
import io
import math
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
from finite_strain import FiniteStrainConsolidation
from Mikasa import MikasaConsolidationSimulator
from Terzaghi import (TerzaghiConsolidationSimulator, matched_cv, matched_modulus,
                      plot_comparison, run_pair, terzaghi_degree)

#: 小ひずみ極限の一致条件。自重と有限ひずみの影響が小さくなるよう荷重を初期有効応力の
#: 1e-6 倍に取り、線形モデルと同じ ``c_v`` を与える ``k0`` を圧縮則から逆算する。
SMALL_STRAIN_CV = 4.0
SMALL_STRAIN = dict(depth=1.0, length=1.0, nx=1, nz=41, e0=1.0, Cc=0.005, Ck=0.05,
                    sigma_ref=1.0e6, sigma_top=1.0e6, load_intensity=1.0, load_width=1.0e9,
                    k0=SMALL_STRAIN_CV * 9.81 * 0.005 / (2.0 * 1.0e6 * math.log(10.0)))


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
        simulator = TerzaghiConsolidationSimulator(nx=4, nz=4, load_intensity=0, total_time=0.1)
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


class FiniteStrainTests(unittest.TestCase):
    """有限ひずみ（三笠／Gibson 系）実装の検証。"""

    @staticmethod
    def _reference_void_ratio(simulator):
        """自重と有効応力の釣り合いを RK4 で独立に積分した参照解 ``e(σ')``。"""
        def void_of(sigma):
            return simulator.e0 - simulator.Cc * math.log10(sigma / simulator.sigma_ref)

        step = 1e-4
        zeta, sigma, points = 0.0, simulator.sigma_top + simulator.load_intensity, []
        while zeta < simulator.d_zeta * simulator.nz:
            points.append((zeta, void_of(sigma)))
            slopes = [simulator.gamma_w * (simulator.gs + void_of(sigma))]
            for offset, weight in ((0.5, 1.0), (0.5, 1.0), (1.0, 1.0)):
                slopes.append(simulator.gamma_w * (
                    simulator.gs + void_of(sigma + weight * offset * step * slopes[-1])))
            sigma += step / 6.0 * (slopes[0] + 2.0 * slopes[1] + 2.0 * slopes[2] + slopes[3])
            zeta += step
        points = np.array(points)
        centers = simulator.d_zeta * (np.arange(1, simulator.nz + 1) - 0.5)
        return np.interp(centers, points[:, 0], points[:, 1])

    def test_no_load_keeps_the_self_weight_equilibrium(self):
        """無載荷では自重と釣り合った初期状態が保たれ、見かけの圧密が起きない。"""
        for nz in (4, 16, 64):
            with self.subTest(nz=nz):
                simulator = FiniteStrainConsolidation(nx=2, nz=nz, load_intensity=0.0,
                                                      total_time=0.1)
                np.testing.assert_allclose(simulator.u, 0.0, atol=1e-9)
                with contextlib.redirect_stdout(io.StringIO()):
                    simulator.run_simulation()
                settlement, horizontal, pressure, strain = simulator.get_current_state()
                self.assertLess(np.abs(settlement).max(), 1e-12)
                self.assertLess(np.abs(pressure).max(), 1e-9)
                self.assertLess(np.abs(strain).max(), 1e-12)
                np.testing.assert_array_equal(horizontal, 0)
                self.assertEqual(simulator.get_consolidation_degree(), 100.0)

    def test_reset_restores_the_initial_state(self):
        simulator = FiniteStrainConsolidation(nx=3, nz=5, total_time=50.0, dt=5.0)
        initial = simulator.get_current_state()
        simulator.run_simulation(verbose=False)
        self.assertGreater(float(np.abs(simulator.disp_z).max()), 0.0)
        simulator.reset()
        self.assertEqual(simulator.split_count, 0)
        for actual, expected in zip(simulator.get_current_state(), initial):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(simulator.get_consolidation_degree(), 0.0)

    def test_steady_state_matches_independent_ode_integration(self):
        """最終状態の ``e`` プロファイルが独立積分と一致し、格子細分化で収束する。"""
        errors = []
        for nz in (11, 21, 41, 81):
            with self.subTest(nz=nz):
                simulator = FiniteStrainConsolidation(nx=1, nz=nz, total_time=3650.0, dt=5.0)
                simulator.run_simulation(verbose=False)
                self.assertLess(float(np.abs(simulator.u).max()), 1e-6)
                errors.append(float(np.abs(
                    self._reference_void_ratio(simulator) - simulator.e[:, 0]).max()))
        self.assertLess(errors[-1], 5e-6)
        for coarse, fine in zip(errors, errors[1:]):
            self.assertLess(fine, 0.5 * coarse)

    def test_small_strain_limit_matches_terzaghi_series(self):
        for factor in (0.1, 0.2, 0.4):
            with self.subTest(time_factor=factor):
                simulator = FiniteStrainConsolidation(
                    total_time=factor / SMALL_STRAIN_CV, dt=0.0005, **SMALL_STRAIN)
                simulator.run_simulation(verbose=False)
                deviation = simulator.get_consolidation_degree() - terzaghi_degree(factor)
                self.assertLess(abs(deviation), 0.5)

    def test_time_step_refinement_reduces_the_deviation_from_terzaghi(self):
        reference = terzaghi_degree(0.4)
        errors = []
        for step in (0.001, 0.00025):
            with self.subTest(dt=step):
                simulator = FiniteStrainConsolidation(total_time=0.4 / SMALL_STRAIN_CV,
                                                      dt=step, **dict(SMALL_STRAIN, nz=21))
                simulator.run_simulation(verbose=False)
                self.assertEqual(simulator.split_count, 0)
                errors.append(abs(simulator.get_consolidation_degree() - reference))
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[1], 0.15)

    def test_uniform_load_two_dimensional_run_reproduces_the_vertical_solution(self):
        parameters = dict(depth=2.0, length=2.0, nz=21, total_time=100.0, dt=5.0,
                          load_width=60.0)
        vertical = FiniteStrainConsolidation(nx=1, **parameters)
        lateral = FiniteStrainConsolidation(nx=5, **parameters)
        vertical.run_simulation(verbose=False)
        lateral.run_simulation(verbose=False)
        np.testing.assert_allclose(lateral.disp_z[:, 2], vertical.disp_z[:, 0], rtol=2e-4)
        np.testing.assert_allclose(lateral.e[:, 2], vertical.e[:, 0], rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(lateral.u[:, 2], vertical.u[:, 0], atol=1e-6)

    def test_two_dimensional_run_converges_toward_the_linear_model(self):
        deviations = []
        for nodes in (11, 21):
            with self.subTest(nodes=nodes):
                finite_parameters = dict(depth=1.0, length=2.0, nx=nodes, nz=nodes, e0=1.0,
                                         Cc=0.005, Ck=0.05, sigma_ref=1.0e6, sigma_top=1.0e6,
                                         load_intensity=1.0, load_width=0.4, dt=0.0005)
                finite_parameters["k0"] = SMALL_STRAIN_CV * 9.81 * finite_parameters["Cc"] / (
                    (1.0 + finite_parameters["e0"]) * finite_parameters["sigma_ref"]
                    * math.log(10.0))
                largest = 0.0
                for factor in (0.1, 0.4):
                    duration = factor / SMALL_STRAIN_CV
                    finite = FiniteStrainConsolidation(total_time=duration,
                                                       **finite_parameters)
                    finite.run_simulation(verbose=False)
                    linear = ConsolidationSimulator(
                        cv=SMALL_STRAIN_CV, E=1.0, poisson_ratio=0.3, total_time=duration,
                        **{name: finite_parameters[name] for name in
                           ("length", "depth", "nx", "nz", "load_intensity", "load_width", "dt")})
                    linear.run_simulation(verbose=False)
                    difference = (finite.get_consolidation_degree()
                                  - linear.get_consolidation_degree())
                    self.assertLess(abs(difference), 3.0)
                    largest = max(largest, abs(difference))
                deviations.append(largest)
        self.assertLess(deviations[1], deviations[0])

    def test_invalid_inputs_are_rejected(self):
        cases = {"length": (0.0, np.nan), "depth": (-1.0,), "nx": (0, 2.5, True),
                 "nz": (0, 1, 3.0), "e0": (0.0, np.nan), "Cc": (0.0, np.inf),
                 "Ck": (0.0, -1.0), "k0": (0.0,), "sigma_ref": (0.0,), "sigma_top": (0.0,),
                 "specific_gravity": (0.0,), "unit_weight_water": (0.0,),
                 "load_width": (0.0,), "load_intensity": (-1.0, np.nan),
                 "dt": (0.0, np.inf), "total_time": (0.0,), "max_iterations": (0,),
                 "max_splits": (0,), "tolerance": (0.0, np.nan)}
        for parameter, values in cases.items():
            for value in values:
                with self.subTest(parameter=parameter, value=value), self.assertRaises(ValueError):
                    FiniteStrainConsolidation(**{parameter: value})

    def test_non_convergence_is_reported_instead_of_clipped(self):
        simulator = FiniteStrainConsolidation(nx=2, nz=4, total_time=1.0, dt=1.0,
                                              max_iterations=1, max_splits=1)
        with self.assertRaises(RuntimeError):
            simulator.run_simulation(verbose=False)

    def test_plot_results_renders_one_and_two_dimensional_fields(self):
        for nx, load in ((1, 50.0), (5, 50.0), (3, 0.0)):
            with self.subTest(nx=nx, load=load), tempfile.TemporaryDirectory() as directory:
                simulator = FiniteStrainConsolidation(nx=nx, nz=5, total_time=50.0, dt=5.0,
                                                      load_intensity=load)
                simulator.run_simulation(verbose=False)
                path = Path(directory) / "finite.png"
                simulator.plot_results(save_path=path, show=False, dpi=30)
                self.assertTrue(path.is_file())
                self.assertEqual(plt.get_fignums(), [])

    def test_module_aliases_point_at_the_finite_strain_model(self):
        self.assertIs(MikasaConsolidationSimulator, FiniteStrainConsolidation)

    def test_terzaghi_comparison_helpers_agree_and_save_a_figure(self):
        self.assertAlmostEqual(matched_cv(), 4.008986074244476, places=9)
        self.assertAlmostEqual(matched_modulus(), 68419.67133468021, places=6)
        self.assertAlmostEqual(terzaghi_degree(0.4), 69.78819062267267, places=8)
        terzaghi, mikasa = run_pair(0.4)
        for simulator in (terzaghi, mikasa):
            deviation = simulator.get_consolidation_degree() - terzaghi_degree(0.4)
            self.assertLess(abs(deviation), 0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.png"
            plot_comparison(save_path=path, show=False, dpi=30)
            self.assertTrue(path.is_file())
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
