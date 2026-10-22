import unittest

import numpy as np

from benchmarks.song_2024 import simulate


class LiteratureComparisonTests(unittest.TestCase):
    def test_settlement_degree_matches_published_cs2_case_at_one_year(self):
        # Song (2024), Table 1: Cc=Ck=1, V-Cv at one year is 42.766%.
        degree, pressure = simulate(1.0, (81, 1), [1.0])[0]
        self.assertAlmostEqual(degree, 42.766, delta=0.3)
        self.assertGreater(pressure, 0)
        self.assertLess(pressure, 400)

    def test_removing_weight_preserves_hydraulic_time_scale(self):
        times = [0.05, 0.1]
        baseline = simulate(0.8, (21, 1), times, 1e-8)
        trial = simulate(0.8, (21, 1), times, 1e-6)
        np.testing.assert_allclose(trial, baseline, rtol=0, atol=0.001)
        self.assertGreater(baseline[1, 0], baseline[0, 0])
