"""Evaluate SciPy without changing the public solvers or required dependencies.

Run: uv run --locked --with scipy==1.18.1 python benchmarks/scipy_evaluation.py
Writes raw repeated timings and independent accuracy checks to outputs/scipy/.
"""
import argparse
from contextlib import ExitStack, contextmanager
from decimal import Decimal, localcontext
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.linalg import eigh_tridiagonal, solve_banded
from scipy.linalg.lapack import dgtsv

from mikasa_consolidation import finite_strain, linear
from mikasa_consolidation.comparison import terzaghi_degree

BACKENDS = ("current", "scipy_banded", "scipy_lapack")
REPEATS = 5


def banded_batch(lower, diagonal, upper, rhs):
    bands = np.zeros((diagonal.shape[1], 3, diagonal.shape[0]))
    bands[:, 0, 1:] = upper[:-1].T
    bands[:, 1] = diagonal.T
    bands[:, 2, :-1] = lower[1:].T
    return solve_banded((1, 1), bands, rhs.T[..., None])[..., 0].T


def lapack_batch(lower, diagonal, upper, rhs):
    if not all(np.isfinite(field).all() for field in (lower, diagonal, upper, rhs)):
        raise ValueError("non-finite tridiagonal system")
    if diagonal.shape[0] == 1:
        if np.any(diagonal == 0):
            raise np.linalg.LinAlgError("singular tridiagonal system")
        return rhs / diagonal
    solution = np.empty_like(rhs)
    for column in range(diagonal.shape[1]):
        *_, solved, status = dgtsv(lower[1:, column], diagonal[:, column],
                                  upper[:-1, column], rhs[:, column, None])
        if status != 0:
            raise np.linalg.LinAlgError(f"dgtsv failed: info={status}")
        solution[:, column] = solved[:, 0]
    return solution


def scipy_modes(matrix, weights):
    root = np.sqrt(weights)
    symmetric = root[:, None] * matrix / root[None, :]
    return eigh_tridiagonal(np.diag(symmetric), np.diag(symmetric, 1))


@contextmanager
def backend(name):
    with ExitStack() as stack:
        if name in ("scipy_banded", "scipy_lapack"):
            solve = banded_batch if name == "scipy_banded" else lapack_batch
            def single(lower, diagonal, upper, rhs):
                return solve(*(np.asarray(field)[:, None] for field in (lower, diagonal, upper, rhs)))[:, 0]
            stack.enter_context(patch.object(finite_strain, "_tridiagonal", single))
            stack.enter_context(patch.object(finite_strain, "_tridiagonal_batch", solve))
        elif name == "scipy_modes":
            stack.enter_context(patch.object(linear, "_weighted_modes", scipy_modes))
        elif name != "current":
            raise ValueError(f"unknown experimental backend: {name}")
        yield


def solver_checks():
    rng = np.random.default_rng(42)
    errors = {}
    for shape in ((1, 1), (2, 1), (7, 5), (101, 3)):
        lower, upper = (rng.uniform(-.5, 0, shape) for _ in range(2))
        lower[0] = upper[-1] = 0
        diagonal = 1 + np.abs(lower) + np.abs(upper)
        rhs = rng.normal(size=shape)
        inputs = [field.copy() for field in (lower, diagonal, upper, rhs)]
        for name, solve in (("current", finite_strain._tridiagonal_batch),
                            ("scipy_banded", banded_batch), ("scipy_lapack", lapack_batch)):
            solution = solve(*inputs)
            residuals = []
            for column in range(shape[1]):
                matrix = np.diag(diagonal[:, column]) + np.diag(lower[1:, column], -1) + np.diag(upper[:-1, column], 1)
                reference = np.linalg.solve(matrix, rhs[:, column])
                np.testing.assert_allclose(solution[:, column], reference, atol=1e-13, rtol=1e-13)
                residuals.append(float(np.linalg.norm(matrix @ solution[:, column] - rhs[:, column], np.inf)
                                       / (np.linalg.norm(matrix, np.inf) * np.linalg.norm(solution[:, column], np.inf)
                                          + np.linalg.norm(rhs[:, column], np.inf))))
            errors[f"{shape}_{name}"] = max(residuals)
            for original, after in zip((lower, diagonal, upper, rhs), inputs):
                np.testing.assert_array_equal(after, original)
    for solve in (banded_batch, lapack_batch):
        for diagonal in (np.zeros((2, 1)), np.full((2, 1), np.nan)):
            try:
                solve(np.zeros((2, 1)), diagonal, np.zeros((2, 1)), np.ones((2, 1)))
            except (ValueError, np.linalg.LinAlgError):
                pass
            else:
                raise AssertionError("invalid systems must not silently succeed")
    return errors


def pivot_stress_check():
    # This deliberately non-dominant matrix is not a normal Newton matrix here.
    lower = np.array([[0.], [1.]])
    diagonal = np.array([[1e-20], [1.]])
    upper, rhs = np.array([[1.], [0.]]), np.array([[1.], [2.]])
    with localcontext() as ctx:
        ctx.prec = 80
        epsilon = Decimal("1e-20")
        exact = np.array([float(1 / (1 - epsilon)), float((1 - 2 * epsilon) / (1 - epsilon))])
    return {name: float(np.max(np.abs(solve(lower, diagonal, upper, rhs)[:, 0] - exact)))
            for name, solve in (("current", finite_strain._tridiagonal_batch),
                                ("scipy_banded", banded_batch), ("scipy_lapack", lapack_batch))}


def timed_case(kind, parameters, name):
    with backend(name):
        started = time.perf_counter()
        simulator = (finite_strain.FiniteStrainConsolidation if kind == "finite" else linear.ConsolidationSimulator)(**parameters)
        initialized = time.perf_counter()
        simulator.run_simulation(verbose=False)
        finished = time.perf_counter()
    return simulator, {"initialization_s": initialized - started, "advance_s": finished - initialized,
                       "total_s": finished - started}


def compare_case(kind, parameters):
    names = BACKENDS if kind == "finite" else ("current", "scipy_modes")
    baseline, first = timed_case(kind, parameters, "current")
    warmup, errors = {"current": first}, {}
    for name in names[1:]:
        trial, warmup[name] = timed_case(kind, parameters, name)
        fields = ("e", "u", "disp_z") if kind == "finite" else ("u", "disp_z", "disp_x")
        errors[name] = {}
        for field in fields:
            expected, actual = getattr(baseline, field), getattr(trial, field)
            np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)
            errors[name][field] = float(np.max(np.abs(actual - expected)))
        if kind == "finite":
            assert trial.split_count == baseline.split_count
    runs = {name: [] for name in names}
    for repeat in range(REPEATS):
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        for name in order:
            _, timing = timed_case(kind, parameters, name)
            runs[name].append(timing)
    medians = {name: statistics.median(run["total_s"] for run in timings) for name, timings in runs.items()}
    summary = {"kind": kind, "parameters": parameters, "first_runs": warmup, "runs": runs,
               "median_total_s": medians, "speedup": {name: medians["current"] / medians[name] for name in names[1:]},
               "max_field_differences": errors}
    print(json.dumps(summary), flush=True)
    return summary


def accuracy_refinement():
    rows = []
    for nz, step in ((21, .001), (41, .00025), (81, .0000625)):
        parameters = dict(nx=1, nz=nz, depth=1, e0=1, Cc=.005, Ck=.05,
                          sigma_ref=1e6, sigma_top=1e6, load_intensity=1,
                          k0=4 * 9.81 * .005 / (2e6 * math.log(10)), total_time=.1, dt=step)
        for name in BACKENDS:
            model, timing = timed_case("finite", parameters, name)
            rows.append({"nz": nz, "dt_day": step, "backend": name,
                         "degree_percent": model.get_consolidation_degree(),
                         "error_pp": model.get_consolidation_degree() - terzaghi_degree(.4), **timing})
    return rows


def linear_accuracy():
    rows = []
    for name in ("current", "scipy_modes"):
        with backend(name):
            model = linear.ConsolidationSimulator(nx=41, nz=41, dt=.001, total_time=.4)
            x = np.linspace(0, model.length, model.nx)
            z = np.linspace(0, model.depth, model.nz)
            kx, kz = np.pi / model.length, np.pi / (2 * model.depth)
            vertical = np.sin(kz * z)[:, None]
            model.u = 50 * vertical * (1 + .2 * np.cos(kx * x))
            model.run_simulation(verbose=False)
            exact = 50 * vertical * np.exp(-kz**2 * .4) * (1 + .2 * np.cos(kx * x) * np.exp(-kx**2 * .4))
            rows.append({"backend": name, "max_pressure_error_kpa": float(np.max(np.abs(model.u - exact)))})
    return rows



def comparison_plot(report, destination):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for axis, kind in zip(axes, ("finite", "linear")):
        cases = [case for case in report["cases"] if case["kind"] == kind]
        names = ("scipy_banded", "scipy_lapack") if kind == "finite" else ("scipy_modes",)
        for name in names:
            axis.plot([case["speedup"][name] for case in cases], np.arange(len(cases)), "o", label=name)
        axis.axvline(1, color="gray", linestyle="--")
        axis.set(yticks=np.arange(len(cases)),
                 yticklabels=[f"{case['parameters']['nx']} × {case['parameters']['nz']}" for case in cases],
                 xscale="log", xlabel="Current median / SciPy median (above 1 is faster)",
                 title=f"{kind.capitalize()} model, initialization + advance")
        axis.grid(alpha=.3)
        axis.legend()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    output = Path("outputs/scipy")
    output.mkdir(parents=True, exist_ok=True)
    report = {"environment": {"platform": platform.platform(), "processor": platform.processor(),
                              "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                              "numpy_blas": np.show_config(mode="dicts")["Build Dependencies"]["blas"],
                              "scipy_blas": scipy.show_config(mode="dicts")["Build Dependencies"]["blas"],
                              "thread_environment": {key: os.environ.get(key) for key in
                                  ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}},
              "repeat_count": REPEATS, "solver_backward_errors": solver_checks(),
              "non_dominant_pivot_stress_forward_errors": pivot_stress_check(),
              "accuracy_refinement": accuracy_refinement(), "linear_accuracy": linear_accuracy(), "cases": []}
    cases = [("finite", dict(nx=1, nz=81, total_time=3650, dt=5)),
             ("finite", dict(nx=21, nz=21, total_time=3650, dt=5))]
    cases += [("finite", dict(nx=n, nz=n, total_time=.02 if n < 401 else .005, dt=.001)) for n in (51, 101, 201, 401)]
    cases += [("linear", dict(nx=n, nz=n, total_time=3650, dt=5)) for n in (21, 101, 401, 801)]
    for kind, parameters in cases:
        report["cases"].append(compare_case(kind, parameters))
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    comparison_plot(report, output / "comparison.png")
    print(output / "report.json", flush=True)


def check_suite():
    solver_checks()
    stress = pivot_stress_check()
    assert stress["scipy_banded"] < 1e-14 and stress["scipy_lapack"] < 1e-14
    for name in ("scipy_banded", "scipy_lapack", "scipy_modes"):
        with backend(name):
            suite = unittest.defaultTestLoader.discover("tests", top_level_dir=".")
            result = unittest.TextTestRunner(verbosity=1).run(suite)
        if not result.wasSuccessful():
            raise AssertionError(f"existing regression suite failed with {name}")
        print(f"{name}: {result.testsRun} existing tests, {len(result.skipped)} skipped", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run adapter checks and existing regressions without timings")
    arguments = parser.parse_args()
    if arguments.check:
        check_suite()
    else:
        main()
