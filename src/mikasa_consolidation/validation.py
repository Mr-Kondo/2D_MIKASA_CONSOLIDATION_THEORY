"""CPU/CUDA comparisons using equal conditions and synchronized host outputs."""

import importlib.metadata
import platform
import statistics
import time

import numpy as np

from . import ConsolidationSimulator, FiniteStrainConsolidation
from .cuda import cuda_info


def _timed_run(cls, parameters, device):
    started = time.perf_counter()
    simulator = cls(device=device, **parameters)
    simulator.run_simulation(verbose=False)
    return time.perf_counter() - started, simulator.get_current_state()


def compare_devices(parameters):
    report = {"environment": {**cuda_info(), "python": platform.python_version(),
              "numpy": np.__version__, "version": importlib.metadata.version("2d-mikasa-consolidation-theory")},
              "parameters": dict(parameters), "models": {}}
    for cls in (FiniteStrainConsolidation, ConsolidationSimulator):
        first_cpu, reference = _timed_run(cls, parameters, "cpu")
        first_gpu, trial = _timed_run(cls, parameters, "cuda")
        errors = []
        for expected, actual in zip(reference, trial):
            if not np.isfinite(actual).all():
                raise FloatingPointError("CUDA comparison returned non-finite output")
            np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)
            errors.append(float(np.max(np.abs(actual - expected))))
        runs = {"cpu": [], "cuda": []}
        for repeat in range(3):
            for device in (("cpu", "cuda") if repeat % 2 == 0 else ("cuda", "cpu")):
                elapsed, _ = _timed_run(cls, parameters, device)
                runs[device].append(elapsed)
        median = {device: statistics.median(times) for device, times in runs.items()}
        report["models"][cls.__name__] = {"first_s": {"cpu": first_cpu, "cuda": first_gpu},
            "runs_s": runs, "median_s": median, "speedup": median["cpu"] / median["cuda"],
            "max_abs_errors": dict(zip(("settlement_m", "horizontal_m", "pressure_kpa", "strain"), errors))}
    return report
