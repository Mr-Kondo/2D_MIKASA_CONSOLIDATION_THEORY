"""Paper-specific self-weight columns sharing the production finite-volume solver.

Takada (1983) excludes the uncompressed cap and imposes its yield stress at
the top of the active column. No fitting to settlement observations is used.
"""

import math

import numpy as np

from mikasa_consolidation import FiniteStrainConsolidation


class SelfWeightColumn(FiniteStrainConsolidation):
    def __init__(self, parameters, nz):
        self.cv = parameters["cv"]
        self.cv_log_slope = parameters.get("cv_log_slope", 0)
        if (not np.isfinite(self.cv) or self.cv <= 0
                or not np.isfinite(self.cv_log_slope) or self.cv_log_slope < 0):
            raise ValueError("cv must be positive and its log slope finite and nonnegative")
        settings = {"load_intensity": 0} | parameters
        super().__init__(nx=1, nz=nz, dt=1, total_time=1,
                         **{key: value for key, value in settings.items() if key not in ("cv", "cv_log_slope")})

    def _build_initial_state(self):
        self.d_zeta = self.depth / (self.nz * (1 + self.e0))
        self.e_initial = np.full((self.nz, 1), self.e0)
        self.sigma_load = np.full((self.nz, 1), self.load_intensity)
        self.e_surface = np.array([self._void_ratio_at(self.sigma_top + self.load_intensity)])

    def _permeability_at(self, void_ratio):
        cv = self.cv * np.power(10.0, (self.e0 - void_ratio) * self.cv_log_slope)
        return cv * self.gamma_w * self.Cc / (
            math.log(10) * self._effective_stress_at(void_ratio) * (1 + void_ratio))

    def _permeability_derivative(self, void_ratio):
        return self._permeability_at(void_ratio) * (math.log(10) * (1 / self.Cc - self.cv_log_slope) - 1 / (1 + void_ratio))


def active_parameters(case):
    if (not all(np.isfinite(value) and value > 0 for value in case.values())
            or case["specific_gravity"] <= 1):
        raise ValueError("invalid self-weight case parameters")
    water_weight = 9.80665
    height, volume_ratio = case["height"], case["volume_ratio"]
    yield_stress, gravity = case["yield_stress"], case["specific_gravity"]
    cap = yield_stress * volume_ratio / (water_weight * (gravity - 1))
    if cap >= height:
        raise ValueError("the selected column has no normally consolidated active layer")
    return dict(depth=height - cap, e0=volume_ratio - 1, sigma_top=yield_stress,
                sigma_ref=yield_stress, Cc=case["Cc"], cv=case["cv"],
                specific_gravity=gravity, unit_weight_water=water_weight), cap


def sample_column(model, time_day, material_positions):
    centers = model.d_zeta * (np.arange(model.nz) + 0.5)
    edges = model.d_zeta * np.arange(model.nz + 1)
    if (not np.isfinite(material_positions).all()
            or np.any(np.asarray(material_positions) < 0)
            or np.any(np.asarray(material_positions) > edges[-1] + 1e-12)):
        raise ValueError("observation outside the active material column")
    displacement = np.r_[model.disp_z[:, 0], 0]
    settlement = np.interp(material_positions, edges, displacement)
    pressure = np.interp(material_positions, np.r_[0, centers, edges[-1]], np.r_[0, model.u[:, 0], model.u[-1, 0]])
    depth = np.interp(material_positions, edges, np.r_[0, np.cumsum(1 + model.e[:, 0]) * model.d_zeta])
    return {"time_day": time_day, "settlement_m": settlement, "pressure_kpa": pressure, "depth_m": depth}


def simulate(parameters, resolution, times_day, positions):
    nz, refinement = resolution
    if (nz < 2 or refinement <= 0 or not np.isfinite(times_day).all()
            or np.any(np.asarray(times_day) <= 0) or np.any(np.diff(times_day) <= 0)):
        raise ValueError("invalid resolution or observation times")
    model = SelfWeightColumn(parameters, nz)
    elapsed, rows = 0.0, []
    for target in times_day:
        while target - elapsed > 1e-12:
            step = min(target - elapsed, max(elapsed, 1e-4) * 0.02 / refinement,
                       parameters["depth"]**2 / parameters["cv"] / (20 * refinement))
            model.total_time = model.dt = step
            model.run_simulation(verbose=False)
            elapsed += step
        rows.append(sample_column(model, target, positions))
    return rows
