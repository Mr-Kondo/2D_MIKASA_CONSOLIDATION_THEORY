import unittest

import numpy as np

from benchmarks.experimental import at_times, error_metrics, material_parameters, observations
from benchmarks.extract_le import axis_values


class ExperimentalComparisonTests(unittest.TestCase):
    def test_reference_to_stage_conversion_preserves_solid_height_and_permeability(self):
        parameters = material_parameters()
        self.assertAlmostEqual(parameters["depth"] / (1 + parameters["e0"]), 0.1405 / 4.165)
        self.assertAlmostEqual(parameters["k0"] * 10 ** ((3.165 - parameters["e0"]) / 1.03), 0.00144)
        self.assertAlmostEqual(parameters["Cc"], 1.002182892337461)
        self.assertAlmostEqual(parameters["sigma_top"] + parameters["load_intensity"], 50)

    def test_axes_use_log_time_and_the_correct_vertical_direction(self):
        for limits in ((20, 65), (40, 0)):
            panel = {"axes_pdf_pt": [10, 20, 70, 120], "vertical_limits": limits}
            np.testing.assert_allclose(axis_values(10, 20, panel), (0.1, limits[0]))
            np.testing.assert_allclose(axis_values(70, 120, panel), (100000, limits[1]))
            self.assertAlmostEqual(axis_values(40, 70, panel)[0], 100)

    def test_observations_exclude_the_pressure_point_without_bracketing_strain_data(self):
        times, measured = observations()
        self.assertEqual(len(times), 13)
        self.assertAlmostEqual(times[-1], 40490.927)
        self.assertTrue(np.isfinite(measured).all())
        self.assertGreater(measured[-1, 0], measured[0, 0])

    def test_interpolation_refuses_extrapolation_and_metrics_keep_physical_units(self):
        states = np.array([[0, 25], [10, 0]])
        np.testing.assert_allclose(at_times(np.array([0, 10]), states, np.array([5])), [[5, 12.5]])
        for times in (np.array([-1]), np.array([11])):
            with self.assertRaises(ValueError):
                at_times(np.array([0, 10]), states, times)
        metrics = error_metrics(np.array([[3, -4], [-3, 4]]), np.zeros((2, 2)))
        self.assertEqual(metrics["settlement_mm"], {"rmse": 3, "max_abs": 3})
        self.assertEqual(metrics["pressure_kpa"], {"rmse": 4, "max_abs": 4})
