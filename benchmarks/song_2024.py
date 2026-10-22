"""Compare the public finite-strain solver with Song et al. (2024), Table 1 V-Cv.

Run: uv run --locked python benchmarks/song_2024.py
The table contains CS2 numerical predictions, not laboratory measurements.
"""

import json
import math
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mikasa_consolidation import FiniteStrainConsolidation


TABLE = Path(__file__).with_name("data") / "song_2024_table1.csv"
YEAR_DAYS = 365.25
FINAL_SETTLEMENT = 10 / 3.7 * math.log10(21)
CK_VALUES = (0.6, 0.8, 1.0, 1.2)


def reference_table():
    table = np.genfromtxt(TABLE, delimiter=",", names=True)
    times = table["time_year"]
    degrees = np.column_stack([table[name] for name in table.dtype.names[1:]])
    if (not np.isfinite(times).all() or not np.isfinite(degrees).all()
            or np.any(times <= 0) or np.any(np.diff(times) <= 0)
            or np.any((degrees < 0) | (degrees > 100))):
        raise ValueError("invalid published time/degree table")
    return times, degrees


def simulate(ck, resolution, times_year, weight_scale=1e-8):
    nz, refinement = resolution
    # Scale k and gamma together to remove self-weight without changing k/gamma.
    model = FiniteStrainConsolidation(
        nx=1, nz=nz, depth=10, e0=2.7, sigma_ref=20, sigma_top=20,
        Cc=1, Ck=ck, k0=2e-8 * 86400 * weight_scale,
        unit_weight_water=9.81 * weight_scale, load_intensity=400,
        dt=1, total_time=1)
    time_day, rows = 0.0, []
    for target in np.asarray(times_year) * YEAR_DAYS:
        while target - time_day > 1e-10:
            step = min(target - time_day, 10 / refinement,
                       max(time_day, 0.1) * 0.02 / refinement)
            model.total_time = model.dt = step
            model.run_simulation(verbose=False)
            time_day += step
        rows.append((100 * model.disp_z[0, 0] / FINAL_SETTLEMENT, model.u[-1, 0]))
    if not np.isfinite(rows).all():
        raise FloatingPointError("non-finite literature comparison")
    return np.asarray(rows)


def comparison_runs(times, published):
    runs, predictions = {}, {}
    for nz, refinement in ((41, 1), (81, 1), (161, 1), (321, 2), (641, 2), (641, 4)):
        key = f"nz{nz}_time{refinement}"
        predictions[key] = np.stack([simulate(ck, (nz, refinement), times) for ck in CK_VALUES], axis=1)
        delta = predictions[key][:, :, 0] - published
        runs[key] = {"rmse_percentage_points": np.sqrt(np.mean(delta**2, axis=0)).tolist(),
                     "max_abs_percentage_points": np.max(np.abs(delta), axis=0).tolist()}
        print(key, runs[key], flush=True)
    return runs, predictions


def comparison_plot(times, published, predicted, destination):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for column, (axis, ck) in enumerate(zip(axes.flat, CK_VALUES)):
        axis.semilogx(times, published[:, column], "o", label="Published CS2 (Song 2024)")
        axis.semilogx(times, predicted[:, column, 0], "-", label="Current solver, negligible weight")
        axis.set(title=f"Cc = 1, Ck = {ck}", xlabel="Time (year)", ylabel="Settlement degree (%)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def time_scale_sensitivity(times, baseline):
    changes = {}
    for name, factor in (("year_365_days", 365 / YEAR_DAYS), ("water_unit_weight_10", 9.81 / 10)):
        trial = np.stack([simulate(ck, (81, 1), times * factor) for ck in CK_VALUES], axis=1)
        changes[name] = np.max(np.abs(trial[:, :, 0] - baseline[:, :, 0]), axis=0).tolist()
    return changes


def main():
    output = Path("outputs/song_2024")
    output.mkdir(parents=True, exist_ok=True)
    times, published = reference_table()
    runs, predictions = comparison_runs(times, published)
    reference = predictions["nz641_time4"]
    scale_trial = np.stack([simulate(ck, (81, 1), times, 1e-6) for ck in CK_VALUES], axis=1)
    report = {"source": "https://doi.org/10.1002/nag.3843", "table": "Table 1, V-Cv columns",
              "environment": {"python": platform.python_version(), "numpy": np.__version__},
              "year_days": YEAR_DAYS, "Ck_columns": CK_VALUES, "final_settlement_m": FINAL_SETTLEMENT,
              "comparison_quantity": "settlement degree, not pressure-integral degree",
              "water_unit_weight_kN_m3": 9.81, "weight_scale": 1e-8,
              "runs": runs,
              "max_grid_change_pp": np.max(np.abs(predictions["nz641_time2"][:, :, 0]
                                                  - predictions["nz321_time2"][:, :, 0]), axis=0).tolist(),
              "max_time_change_pp": np.max(np.abs(reference[:, :, 0]
                                                  - predictions["nz641_time2"][:, :, 0]), axis=0).tolist(),
              "max_weight_scale_change_pp": np.max(np.abs(scale_trial[:, :, 0]
                                                          - predictions["nz81_time1"][:, :, 0]), axis=0).tolist(),
              "time_scale_sensitivity_pp": time_scale_sensitivity(times, predictions["nz81_time1"])}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = [(t, ck, published[i, j], reference[i, j, 0], reference[i, j, 1])
            for i, t in enumerate(times) for j, ck in enumerate(CK_VALUES)]
    np.savetxt(output / "comparison.csv", rows, delimiter=",", comments="",
               header="time_year,Ck,published_CS2_degree_percent,current_degree_percent,current_near_base_pressure_kpa")
    comparison_plot(times, published, reference, output / "comparison.png")


if __name__ == "__main__":
    main()
