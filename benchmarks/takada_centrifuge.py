"""Compare Takada (1983) Fig.22 single-drainage measured settlement degrees.

Initial cv was calibrated to the initial experimental speed by the original
paper; this is a reproduction with published parameters, not blind prediction.
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .self_weight import active_parameters, simulate
    from .self_weight_comparison import RESOLUTIONS, final_settlement, linear, source, table
else:
    from self_weight import active_parameters, simulate
    from self_weight_comparison import RESOLUTIONS, final_settlement, linear, source, table


def centrifuge_parameters(case, delta):
    parameters, _ = active_parameters(case)
    bottom_stress = 9.80665 * (case["specific_gravity"] - 1) * case["height"] / case["volume_ratio"]
    bottom_change = case["Cc"] * math.log10(bottom_stress / case["yield_stress"])
    return parameters | {"cv_log_slope": math.log10(delta) / bottom_change}


def main():
    metadata = source("takada_1983")
    case, axes = metadata["centrifuge_case"], metadata["centrifuge_axes"]
    points = table("takada_1983_centrifuge")
    times = 10**linear(points["pixel_x"], axes["x"], np.log10(axes["time_1e4_min"])) * 1e4 / 1440
    measured = linear(points["pixel_y"], axes["y"], axes["degree_percent"])
    predictions, conditions = {}, {}
    for delta in (4, 8):
        parameters = centrifuge_parameters(case, delta)
        conditions[str(delta)] = parameters
        for resolution in RESOLUTIONS:
            key = f"delta{delta}_nz{resolution[0]}_time{resolution[1]}"
            rows = simulate(parameters, resolution, times, [0])
            predictions[key] = np.array([100 * row["settlement_m"][0] / final_settlement(case) for row in rows])
    if not np.isfinite(list(predictions.values())).all():
        raise FloatingPointError("non-finite centrifuge predictions")
    report = {"source": metadata, "parameters_by_delta": conditions,
              "final_settlement_m": final_settlement(case),
              "reading_half_width_pp": 3 * 100 / 247,
              "time_reading_factor": 10**(3 * 3 / 465),
              "runs": {key: {"rmse_pp": float(np.sqrt(np.mean((trial-measured)**2))),
                              "max_abs_pp": float(np.max(np.abs(trial-measured)))} for key, trial in predictions.items()},
              "convergence_by_delta": {str(d): {
                  "max_grid_change_pp": float(np.max(np.abs(predictions[f"delta{d}_nz321_time1"] - predictions[f"delta{d}_nz161_time1"]))),
                  "max_time_change_pp": float(np.max(np.abs(predictions[f"delta{d}_nz321_time2"] - predictions[f"delta{d}_nz321_time1"])))} for d in (4, 8)}}
    output = Path("outputs/self_weight")
    output.mkdir(parents=True, exist_ok=True)
    (output / "centrifuge_report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savetxt(output / "centrifuge_comparison.csv", np.column_stack((times, measured, *predictions.values())), delimiter=",", comments="",
               header="prototype_time_day,published_measured_degree_percent," + ",".join(predictions))
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.errorbar(times, measured, yerr=report["reading_half_width_pp"], fmt="o", label="Measured centrifugal model (digitized)")
    for delta in (4, 8):
        axis.plot(times, predictions[f"delta{delta}_nz321_time2"], label=f"Current solver, published delta={delta}")
    axis.set(xscale="log", xlabel="Prototype time (day), tp = 10000 tm", ylabel="Settlement degree (%)",
             title="Takada 1983 Fig.22, Nanko98, f0=4.2, single drainage")
    axis.legend()
    axis.grid(alpha=.3)
    fig.savefig(output / "centrifuge_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report["runs"], indent=2))


if __name__ == "__main__":
    main()
