import unittest

import numpy as np

from benchmarks.self_weight import SelfWeightColumn, active_parameters, simulate
from benchmarks.self_weight_comparison import final_settlement, linear, source
from benchmarks.takada_centrifuge import centrifuge_parameters
from mikasa_consolidation import FiniteStrainConsolidation


class SelfWeightTests(unittest.TestCase):
    def test_vertical_newton_matrix_matches_residual_differentiation(self):
        case = source("takada_1983")["case"] | {"height": 6}
        parameters, _ = active_parameters(case)
        variable = centrifuge_parameters(source("takada_1983")["centrifuge_case"], 8)
        for model in (SelfWeightColumn(parameters, 7), SelfWeightColumn(variable, 7),
                      FiniteStrainConsolidation(nx=1, nz=7)):
            with self.subTest(model=type(model).__name__):
                void = model.e * np.linspace(.99, 1.01, 7)[:, None]
                lower, diagonal, upper = model._vertical_jacobian(void, .01)
                matrix = np.diag(diagonal[:, 0]) + np.diag(lower[1:, 0], -1) + np.diag(upper[:-1, 0], 1)
                numerical = np.empty((7, 7))
                for column in range(7):
                    perturbation = np.zeros_like(void)
                    perturbation[column, 0] = 1e-6
                    numerical[:, column] = ((model._vertical_residual(void, void + perturbation, .01, 0)
                                             - model._vertical_residual(void, void - perturbation, .01, 0)) / 2e-6)[:, 0]
                np.testing.assert_allclose(matrix, numerical, rtol=1e-7, atol=1e-8)

    def test_constant_cv_self_weight_matches_takada_curve_and_final_label(self):
        case = source("takada_1983")["case"] | {"height": 6}
        parameters, cap = active_parameters(case)
        self.assertAlmostEqual(cap, .0303030303)
        row = simulate(parameters, (81, 1), [2], [0])[0]
        # Fig.9 digitized point: H=6 m, cv*t=2 m²; graph allowance is 0.052 m.
        self.assertAlmostEqual(row["settlement_m"][0], (1479 - 1397) * 4 / 231, delta=.052)
        long_run = simulate(parameters, (81, 1), [1000], [0])[0]
        self.assertAlmostEqual(long_run["settlement_m"][0], final_settlement(case), delta=.001)
        self.assertAlmostEqual(final_settlement(case), 1.79, delta=.01)

    def test_refuses_extrapolation_and_invalid_observation_times(self):
        parameters, _ = active_parameters(source("umehara_1979")["case"])
        for times, positions in (([1, 1], [0]), ([0], [0]), ([1], [-.1]), ([1], [1])):
            with self.subTest(times=times, positions=positions), self.assertRaises(ValueError):
                simulate(parameters, (11, 1), times, positions)

    def test_graph_axis_calibration_uses_both_endpoints(self):
        np.testing.assert_allclose(linear([320, 957], [320, 957], [0, 15]), [0, 15])
        axes = source("umehara_1979")["settlement_axes"]
        np.testing.assert_allclose(10**linear(axes["x"], axes["x"], np.log10(axes["time_min"])), axes["time_min"])

    def test_published_cv_ratio_is_reached_at_bottom_equilibrium(self):
        case = source("takada_1983")["centrifuge_case"]
        parameters = centrifuge_parameters(case, 8)
        model = SelfWeightColumn(parameters, 7)
        bottom_stress = 9.80665 * (case["specific_gravity"] - 1) * case["height"] / case["volume_ratio"]
        bottom_void = model._void_ratio_at(bottom_stress)
        recovered_cv = model._permeability_at(bottom_void) * np.log(10) * bottom_stress * (1 + bottom_void) / (model.gamma_w * model.Cc)
        self.assertAlmostEqual(recovered_cv / case["cv"], 8)
        with self.assertRaises(ValueError):
            SelfWeightColumn(parameters | {"cv_log_slope": float("nan")}, 7)
