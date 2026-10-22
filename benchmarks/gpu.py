"""Experimental MLX offload of the existing Thomas solver; not a public backend.

Run: uv run --locked --with mlx==0.32.4 python benchmarks/gpu.py
Outputs JSON to stdout. NumPy residuals and convergence criteria remain float64.
"""

import importlib.metadata
import json
import platform
import statistics
import time

import mlx.core as mx
import numpy as np

from mikasa_consolidation.finite_strain import FiniteStrainConsolidation, _tridiagonal_batch


# One thread per column avoids launching a GPU operation for each row.
metal_thomas = mx.fast.metal_kernel(
    name="thomas_columns",
    input_names=["lower", "diag", "upper", "rhs"],
    output_names=["c", "d", "solution"],
    source=r"""
    uint col = thread_position_in_grid.x;
    int rows = diag_shape[0], cols = diag_shape[1];
    if (col >= cols) return;
    c[col] = upper[col] / diag[col];
    d[col] = rhs[col] / diag[col];
    for (int row = 1; row < rows; ++row) {
        int index = row * cols + col, above = index - cols;
        float pivot = diag[index] - lower[index] * c[above];
        c[index] = upper[index] / pivot;
        d[index] = (rhs[index] - lower[index] * d[above]) / pivot;
    }
    int bottom = (rows - 1) * cols + col;
    solution[bottom] = d[bottom];
    for (int row = rows - 2; row >= 0; --row) {
        int index = row * cols + col;
        solution[index] = d[index] - c[index] * solution[index + cols];
    }
    """,
)


def gpu_solve(lower, diag, upper, rhs):
    arrays = [mx.array(field.astype(np.float32)) for field in (lower, diag, upper, rhs)]
    solution = metal_thomas(inputs=arrays, grid=(diag.shape[1], 1, 1),
                            threadgroup=(min(128, diag.shape[1]), 1, 1),
                            output_shapes=[diag.shape] * 3,
                            output_dtypes=[mx.float32] * 3)[2]
    mx.eval(solution)
    return np.array(solution).astype(np.float64)


def timed_simulation(size, steps, offload, dt):
    started = time.perf_counter()
    simulator = FiniteStrainConsolidation(nx=size, nz=size, dt=dt, total_time=dt * steps)
    if offload:
        simulator._solve_vertical = gpu_solve
        simulator._solve_horizontal = lambda lo, di, up, rhs: gpu_solve(lo.T, di.T, up.T, rhs.T).T
    initialized = time.perf_counter()
    simulator.run_simulation(verbose=False)
    elapsed = time.perf_counter() - started
    return simulator, {"total_s": elapsed, "solve_s": elapsed - (initialized - started)}


def check_thomas():
    # One cell, one column, multiple columns, and an odd grid exercise both ends.
    rng = np.random.default_rng(42)
    for shape in ((1, 1), (2, 1), (7, 5)):
        lower = rng.uniform(-0.5, 0, shape)
        upper = rng.uniform(-0.5, 0, shape)
        lower[0] = upper[-1] = 0
        diag = 1 + np.abs(lower) + np.abs(upper)
        rhs = rng.normal(size=shape)
        solution = gpu_solve(lower, diag, upper, rhs)
        np.testing.assert_allclose(solution, _tridiagonal_batch(lower, diag, upper, rhs),
                                   atol=1e-6, rtol=1e-6)
        for column in range(shape[1]):
            matrix = (np.diag(diag[:, column]) + np.diag(lower[1:, column], -1)
                      + np.diag(upper[:-1, column], 1))
            np.testing.assert_allclose(matrix @ solution[:, column], rhs[:, column], atol=1e-6, rtol=1e-6)


def compare_case(size, steps, dt):
    cpu, cpu_first = timed_simulation(size, steps, False, dt)
    gpu, gpu_first = timed_simulation(size, steps, True, dt)
    errors = {"e_abs": float(np.max(np.abs(cpu.e - gpu.e))),
              "u_abs_kpa": float(np.max(np.abs(cpu.u - gpu.u))),
              "settlement_abs_m": float(np.max(np.abs(cpu.disp_z - gpu.disp_z)))}
    for reference, trial in ((cpu.e, gpu.e), (cpu.u, gpu.u), (cpu.disp_z, gpu.disp_z)):
        np.testing.assert_allclose(trial, reference, atol=1e-8, rtol=1e-8)
    assert gpu.split_count == cpu.split_count
    timings = {"cpu": [], "hybrid_gpu": []}
    # Alternate order to reduce ordering and thermal bias; include initialization and transfers.
    for repeat in range(3):
        for offload in ((False, True) if repeat % 2 == 0 else (True, False)):
            _, timing = timed_simulation(size, steps, offload, dt)
            timings["hybrid_gpu" if offload else "cpu"].append(timing)
    medians = {backend: statistics.median(t["total_s"] for t in runs)
               for backend, runs in timings.items()}
    return {"grid": [size, size], "steps": steps, "dt_day": dt,
            "first_cpu": cpu_first, "first_hybrid_gpu": gpu_first,
            "runs": timings, "median_total_s": medians,
            "speedup": medians["cpu"] / medians["hybrid_gpu"],
            "errors": errors, "splits": cpu.split_count,
            "cpu_center_settlement_mm": float(cpu.disp_z[0, size // 2] * 1000)}


def main():
    if not mx.metal.is_available():
        raise RuntimeError("This experiment requires an Apple Metal GPU")
    mx.set_default_device(mx.gpu)
    print(json.dumps({"environment": {"platform": platform.platform(),
          "python": platform.python_version(), "numpy": np.__version__,
          "mlx": importlib.metadata.version("mlx"), "device": str(mx.default_device()),
          "gpu": mx.device_info()}}, default=str), flush=True)
    try:
        mx.eval(mx.array([1.0], dtype=mx.float64) + 1)
    except ValueError as error:
        print(json.dumps({"float64_gpu_error": str(error)}), flush=True)
    else:
        raise AssertionError("Reassess the float32 experiment: float64 GPU is now available")
    check_thomas()
    for size, steps, dt in ((21, 730, 5.0), (51, 20, 0.05), (101, 20, 0.05),
                            (201, 20, 0.001), (401, 20, 0.001)):
        print(json.dumps(compare_case(size, steps, dt)), flush=True)


if __name__ == "__main__":
    main()
