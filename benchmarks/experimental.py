"""Exploratory comparison with measured Le (2015) KBS Rowe-cell data.

Run: uv run --locked python benchmarks/experimental.py
Writes comparisons, convergence checks and a figure to outputs/experimental/.
This is a check of the current model's limitations, not an EVP implementation.
"""

import json
import math
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mikasa_consolidation import FiniteStrainConsolidation


DATA_DIR = Path(__file__).with_name("data")
REFERENCE_DEPTH_M = 0.1405
INITIAL_STRAIN = 0.2776
REFERENCE_VOID_RATIO = 3.165


def material_parameters():
    void_ratio = (1 + REFERENCE_VOID_RATIO) * (1 - INITIAL_STRAIN) - 1
    return dict(depth=REFERENCE_DEPTH_M * (1 - INITIAL_STRAIN), e0=void_ratio,
                Cc=math.log(10) * 0.1045 * (1 + REFERENCE_VOID_RATIO), Ck=1.03,
                k0=1e-6 * 1440 * 10 ** ((void_ratio - REFERENCE_VOID_RATIO) / 1.03),
                sigma_ref=25, sigma_top=25, load_intensity=25)


def observations():
    strain = np.genfromtxt(DATA_DIR / "le_2015_strain.csv", delimiter=",", names=True)
    pressure = np.genfromtxt(DATA_DIR / "le_2015_pressure.csv", delimiter=",", names=True)
    for series in (strain, pressure):
        if (not all(np.isfinite(series[name]).all() for name in series.dtype.names)
                or np.any(series["time_min"] <= 0) or np.any(np.diff(series["time_min"]) <= 0)):
            raise ValueError("observations must be finite and have positive increasing times")
    common = pressure[(pressure["time_min"] >= strain["time_min"][0])
                      & (pressure["time_min"] <= strain["time_min"][-1])]
    times = common["time_min"]
    strain_percent = np.interp(np.log10(times), np.log10(strain["time_min"]), strain["strain_percent"])
    settlement_mm = (strain_percent / 100 - INITIAL_STRAIN) * REFERENCE_DEPTH_M * 1000
    return times, np.column_stack((settlement_mm, common["pressure_kpa"]))


def trajectory(nz, dt_min, stop_min, overrides=None):
    parameters = material_parameters() | {"specific_gravity": 2.65} | (overrides or {})
    model = FiniteStrainConsolidation(nx=1, nz=nz, dt=dt_min / 1440,
                                     total_time=dt_min / 1440, **parameters)
    times = np.arange(math.ceil(stop_min / dt_min) + 1) * dt_min
    states = np.empty((times.size, 2))
    for index in range(times.size):
        if index:
            model.run_simulation(verbose=False)
        states[index] = model.disp_z[0, 0] * 1000, model.u[-1, 0]
    if not np.isfinite(states).all():
        raise FloatingPointError("experimental comparison contains non-finite predictions")
    return times, states


def at_times(trajectory_times, states, times):
    if times.min() < trajectory_times[0] or times.max() > trajectory_times[-1]:
        raise ValueError("refusing to extrapolate predictions")
    return np.column_stack([np.interp(times, trajectory_times, column) for column in states.T])


def error_metrics(predicted, measured):
    residual = predicted - measured
    return {name: {"rmse": float(np.sqrt(np.mean(column**2))),
                   "max_abs": float(np.max(np.abs(column)))}
            for name, column in zip(("settlement_mm", "pressure_kpa"), residual.T)}


def readout_allowances():
    source = json.loads((DATA_DIR / "le_2015_source.json").read_text())
    delta = source["coordinate_uncertainty_pt"]
    panels = source["panels"]
    time_factor = max(10 ** (6 * delta / (p["axes_pdf_pt"][2] - p["axes_pdf_pt"][0])) for p in panels)
    vertical = [delta * abs(p["vertical_limits"][1] - p["vertical_limits"][0])
                / (p["axes_pdf_pt"][3] - p["axes_pdf_pt"][1]) for p in panels]
    vertical[0] *= REFERENCE_DEPTH_M * 10  # percent of the original height to mm
    return time_factor, vertical


def plot_comparison(times, measured, reference, output):
    time_factor, vertical_errors = readout_allowances()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    curve_times, curve_states = reference
    for index, (axis, label, error) in enumerate(zip(axes, ("Additional settlement (mm)", "Base excess pressure (kPa)"), vertical_errors)):
        axis.errorbar(times, measured[:, index], yerr=error,
                      xerr=(times - times / time_factor, times * time_factor - times),
                      fmt="o", ms=4, label="Le (2015), digitized measurements")
        axis.plot(curve_times[1:], curve_states[1:, index], label="Current model, no creep")
        axis.set(xscale="log", xlabel="Elapsed time (min)", ylabel=label, xlim=(times.min() / 1.5, curve_times[-1]))
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].invert_yaxis()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    output = Path("outputs/experimental")
    output.mkdir(parents=True, exist_ok=True)
    times, measured = observations()
    source = json.loads((DATA_DIR / "le_2015_source.json").read_text())
    time_factor, vertical_errors = readout_allowances()
    stop_min = float(times.max() * time_factor)
    trajectories, sampled, runs = {}, {}, {}
    for nz, step in ((21, 20), (41, 20), (81, 20), (81, 10), (81, 5)):
        key = f"nz{nz}_dt{step}min"
        trajectories[key] = trajectory(nz, step, stop_min)
        sampled[key] = at_times(*trajectories[key], times)
        runs[key] = error_metrics(sampled[key], measured)
        print(key, runs[key], flush=True)
    reference_key = "nz81_dt5min"
    prediction = sampled[reference_key]
    gravity_changes = {}
    for gravity in (2.5, 2.8):
        trial = at_times(*trajectory(41, 20, stop_min, {"specific_gravity": gravity}), times)
        gravity_changes[str(gravity)] = np.max(np.abs(trial - sampled["nz41_dt20min"]), axis=0).tolist()
    # Scaling k and gamma together preserves k/gamma while taking self-weight to zero.
    no_weight = at_times(*trajectory(41, 20, stop_min, {"unit_weight_water": 9.81e-6,
                         "k0": material_parameters()["k0"] * 1e-6}), times)
    report = {"source": source["url"], "parameters": material_parameters(),
              "environment": {"python": platform.python_version(), "numpy": np.__version__},
              "assumed_specific_gravity": 2.65, "common_observation_count": len(times),
              "runs": runs, "max_grid_change_mm_kpa": np.max(np.abs(sampled["nz81_dt20min"] - sampled["nz41_dt20min"]), axis=0).tolist(),
              "max_time_change_mm_kpa": np.max(np.abs(prediction - sampled["nz81_dt10min"]), axis=0).tolist(),
              "specific_gravity_sensitivity_mm_kpa": gravity_changes,
              "vanishing_self_weight_change_mm_kpa": np.max(np.abs(no_weight - sampled["nz41_dt20min"]), axis=0).tolist(),
              "plot_vertical_uncertainty_mm_kpa": vertical_errors,
              "plot_time_uncertainty_factor": time_factor}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savetxt(output / "comparison.csv", np.column_stack((times, measured, prediction)), delimiter=",",
               header="time_min,observed_settlement_mm,observed_pressure_kpa,predicted_settlement_mm,predicted_pressure_kpa", comments="")
    plot_comparison(times, measured, trajectories[reference_key], output / "comparison.png")
    print(output / "report.json")


if __name__ == "__main__":
    main()
