"""Digitized Takada (1983) numerical curves and Umehara/Zen (1979) measurements.

Run: uv run --locked python benchmarks/self_weight_comparison.py
Raw pixel coordinates and source-specific assumptions are in data/*_source.json.
"""
import json
import math
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .self_weight import active_parameters, simulate
else:
    from self_weight import active_parameters, simulate

DATA = Path(__file__).with_name("data")
RESOLUTIONS = ((81, 1), (161, 1), (321, 1), (321, 2))


def table(name):
    values = np.genfromtxt(DATA / f"{name}.csv", names=True, delimiter=",")
    if not all(np.isfinite(values[key]).all() for key in values.dtype.names):
        raise ValueError(f"non-finite digitization: {name}")
    return values


def source(name):
    return json.loads((DATA / f"{name}_source.json").read_text())


def linear(values, endpoints, limits):
    return limits[0] + (np.asarray(values) - endpoints[0]) * (limits[1] - limits[0]) / (endpoints[1] - endpoints[0])


def final_settlement(case):
    parameters, _ = active_parameters(case)
    length = parameters["depth"] / case["volume_ratio"]
    a = case["yield_stress"] / (9.80665 * (case["specific_gravity"] - 1))
    return case["Cc"] / math.log(10) * ((length + a) * math.log1p(length / a) - length)


def takada_comparison():
    metadata = source("takada_1983")
    axes, points = metadata["axes"], table("takada_1983_settlement")
    times = linear(points["pixel_x"], axes["x"], axes["cv_t_m2"])
    measured = linear(points["pixel_y"], axes["y"], axes["settlement_m"])
    predictions = [[] for _ in RESOLUTIONS]
    final = []
    for height, label in zip((6, 8, 10, 12), (1.79, 2.54, 3.33, 4.15)):
        selected = points["height_m"] == height
        case = metadata["case"] | {"height": height}
        parameters, cap = active_parameters(case)
        for values, resolution in zip(predictions, RESOLUTIONS):
            rows = simulate(parameters, resolution, times[selected], [0])
            values.extend(row["settlement_m"][0] for row in rows)
        final.append(dict(height_m=height, cap_m=cap, published_label_m=label,
                          analytic_m=final_settlement(case)))
    rows = np.column_stack((points["height_m"], times, measured, np.asarray(predictions).T))
    if not np.isfinite(rows).all():
        raise FloatingPointError("non-finite Takada comparison")
    return metadata, rows, {"final_settlements": final,
                           "reading_half_width_m": metadata["read_half_width_y_px"] * 4 / 231,
                           "reading_half_width_cv_t_m2": metadata["read_half_width_x_px"] * 8 / 468}


def umehara_comparison():
    metadata = source("umehara_1979")
    s, p = table("umehara_1979_settlement"), table("umehara_1979_pressure")
    axes = metadata["settlement_axes"]
    times = 10**linear(s["pixel_x"], axes["x"], np.log10(axes["time_min"])) / 1440
    measured_s = linear(s["pixel_y"], axes["y"], axes["settlement_m"])
    p_axes = metadata["pressure_axes"]
    measured_p = linear(p["pixel_x"], p_axes["x"], p_axes["pressure_gf_cm2"]) * 0.0980665
    parameters, cap = active_parameters(metadata["case"])
    depths = np.arange(1, 9) / 10
    positions = (depths - cap) / metadata["case"]["volume_ratio"]
    targets = np.unique(np.r_[times, p["time_day"]])
    s_predictions, p_predictions = [], []
    for resolution in RESOLUTIONS:
        calculated = {row["time_day"]: row for row in simulate(parameters, resolution, targets, positions)}
        s_predictions.append([calculated[t]["settlement_m"][0] for t in times])
        p_predictions.append([calculated[row["time_day"]]["pressure_kpa"][int(round(row["initial_depth_m"] * 10)) - 1] for row in p])
    if not np.isfinite(s_predictions).all() or not np.isfinite(p_predictions).all():
        raise FloatingPointError("non-finite Umehara comparison")
    s_rows = np.column_stack((np.full(len(times), .1), times, measured_s, np.asarray(s_predictions).T))
    p_rows = np.column_stack((p["initial_depth_m"], p["time_day"], measured_p, np.asarray(p_predictions).T))
    # Same-time settlement is interpolated in log(time), never extrapolated.
    if p["time_day"].min() < times.min() or p["time_day"].max() > times.max():
        raise ValueError("pressure times outside measured settlement range")
    paired = [dict(time_day=t, interpolated_measured_settlement_m=float(np.interp(np.log(t), np.log(times), measured_s)),
                   calculated_settlement_m=float(calculated[t]["settlement_m"][0])) for t in np.unique(p["time_day"])]
    report = {"cap_m": cap, "case_parameters": parameters, "same_time_settlement": paired,
              "settlement_reading_half_width_m": 2.5 * .1 / 315,
              "pressure_reading_half_width_kpa": 2.5 * 15 / 637 * .0980665,
              "time_reading_factor": 10**(3 * 2.5 / 599)}
    return metadata, s_rows, p_rows, report


def metrics(rows):
    return {"runs": {f"nz{nz}_time{ref}": {"rmse": float(np.sqrt(np.mean((rows[:, i+3] - rows[:, 2])**2))),
                                                   "max_abs": float(np.max(np.abs(rows[:, i+3] - rows[:, 2])))}
                     for i, (nz, ref) in enumerate(RESOLUTIONS)},
            "max_grid_change": float(np.max(np.abs(rows[:, 5] - rows[:, 4]))),
            "max_time_change": float(np.max(np.abs(rows[:, 6] - rows[:, 5])))}


def main():
    output = Path("outputs/self_weight")
    output.mkdir(parents=True, exist_ok=True)
    takada, t_rows, t_report = takada_comparison()
    umehara, s_rows, p_rows, u_report = umehara_comparison()
    report = {"environment": {"python": platform.python_version(), "numpy": np.__version__},
              "takada_1983": t_report | {"source": takada, "settlement_m": metrics(t_rows)},
              "umehara_1979": u_report | {"source": umehara, "settlement_m": metrics(s_rows), "pressure_kpa": metrics(p_rows),
                                        "pressure_by_day": {str(int(day)): metrics(p_rows[p_rows[:, 1] == day]) for day in (10, 50, 80)}}}
    for name, rows in (("takada_1983", t_rows), ("umehara_1979_settlement", s_rows), ("umehara_1979_pressure", p_rows)):
        np.savetxt(output / f"{name}.csv", rows, delimiter=",", comments="",
                   header="height_or_initial_depth_m,time_day_or_cv_t_m2,published," + ",".join(f"nz{n}_time{r}" for n,r in RESOLUTIONS))
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for height in np.unique(t_rows[:, 0]):
        rows = t_rows[t_rows[:, 0] == height]
        line, = axes[0].plot(rows[:, 1], rows[:, -1], label=f"H={height:g} m")
        axes[0].errorbar(rows[:, 1], rows[:, 2], yerr=t_report["reading_half_width_m"], fmt="o", color=line.get_color())
    axes[0].plot([], [], "ko", label="Published, digitized")
    axes[0].plot([], [], "k-", label="Current solver")
    axes[0].set(title="Takada 1983 Fig.9 (numerical)", xlabel="cv t (m²)", ylabel="Surface settlement (m)")
    axes[1].semilogx(s_rows[:, 1], s_rows[:, -1], label="Current constant-cv solver")
    axes[1].errorbar(s_rows[:, 1], s_rows[:, 2], yerr=u_report["settlement_reading_half_width_m"], fmt="o", label="Measured, digitized")
    axes[1].set(title="Umehara/Zen 1979 Fig.6.1", xlabel="Time (day)", ylabel="Plate settlement (m), initial depth 0.1 m")
    for day in (10, 50, 80):
        rows = p_rows[p_rows[:, 1] == day]
        line, = axes[2].plot(rows[:, -1], rows[:, 0], label=f"Current {day} day")
        axes[2].errorbar(rows[:, 2], rows[:, 0], xerr=u_report["pressure_reading_half_width_kpa"], fmt="o", color=line.get_color())
    axes[2].invert_yaxis()
    axes[2].set(title="Fig.6.2(b): measured markers", xlabel="Excess pressure (kPa)", ylabel="Initial sensor depth (m)")
    for axis in axes:
        axis.grid(alpha=.3)
        axis.legend(fontsize=8)
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"takada": metrics(t_rows), "umehara_settlement": metrics(s_rows), "umehara_pressure": report["umehara_1979"]["pressure_by_day"]}, indent=2))


if __name__ == "__main__":
    main()
