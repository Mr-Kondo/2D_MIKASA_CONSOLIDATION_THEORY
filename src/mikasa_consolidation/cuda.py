"""Optional double-precision CUDA operations; never fall back to CPU."""

import numpy as np


def require_cuda():
    try:
        import cupy as cp
    except ImportError as error:
        raise RuntimeError("CUDA requires CuPy: install the [cuda] extra on a CUDA machine") from error
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CUDA requested, but no NVIDIA GPU is available")
    except cp.cuda.runtime.CUDARuntimeError as error:
        raise RuntimeError("CUDA runtime is unavailable; check the NVIDIA driver and CuPy installation") from error
    return cp


def cuda_info():
    cp = require_cuda()
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    return {"gpu": device["name"].decode(), "cupy": cp.__version__,
            "runtime": cp.cuda.runtime.runtimeGetVersion(),
            "driver": cp.cuda.runtime.driverGetVersion(), "dtype": "float64"}


_THOMAS_SOURCE = r'''
extern "C" __global__ void thomas(
    const double* lower, const double* diag, const double* upper,
    const double* rhs, double* c, double* d, double* solution,
    const int rows, const int cols) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= cols) return;
    c[col] = upper[col] / diag[col];
    d[col] = rhs[col] / diag[col];
    for (int row = 1; row < rows; ++row) {
        int index = row * cols + col, above = index - cols;
        double pivot = diag[index] - lower[index] * c[above];
        c[index] = upper[index] / pivot;
        d[index] = (rhs[index] - lower[index] * d[above]) / pivot;
    }
    int bottom = (rows - 1) * cols + col;
    solution[bottom] = d[bottom];
    for (int row = rows - 2; row >= 0; --row) {
        int index = row * cols + col;
        solution[index] = d[index] - c[index] * solution[index + cols];
    }
}
'''


def make_tridiagonal_solver():
    cp = require_cuda()
    kernel = cp.RawKernel(_THOMAS_SOURCE, "thomas", options=("--fmad=false",))

    def solve(lower, diag, upper, rhs):
        rows, cols = diag.shape
        arrays = [cp.asarray(field, dtype=cp.float64, order="C")
                  for field in (lower, diag, upper, rhs)]
        scratch = [cp.empty_like(arrays[1]) for _ in range(3)]
        kernel(((cols + 127) // 128,), (128,), (*arrays, *scratch, np.int32(rows), np.int32(cols)))
        # asnumpy blocks until the kernel and the device-to-host copy complete.
        return cp.asnumpy(scratch[-1])

    return solve
