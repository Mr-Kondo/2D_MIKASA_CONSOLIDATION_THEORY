"""一定係数の2次元排水と拘束圧密の簡易モデル。三笠有限ひずみ理論ではない。"""

import math
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.registry import BackendFilter, backend_registry


def _second_difference(size):
    matrix = np.diag(np.full(size, -2.0))
    matrix += np.diag(np.ones(size - 1), 1) + np.diag(np.ones(size - 1), -1)
    matrix[0, 1] = matrix[-1, -2] = 2.0
    return matrix


def _weighted_modes(matrix, weights):
    # 半幅の境界セルで重み付けすると、鏡映Neumann差分が対称行列になる。
    root = np.sqrt(weights)
    rates, basis = np.linalg.eigh(root[:, None] * matrix / root[None, :])
    return rates, basis


def show_if_interactive():
    """非対話バックエンドでは表示しない。Agg などで出る警告を避ける。"""
    if plt.get_backend().lower() in backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE):
        return
    plt.show()


def is_negligible_field(values, reference):
    """基準量に対して無視できる場か。色の濃淡で表さない判定に使う。"""
    return bool(np.max(np.abs(values)) < 1e-9 * reference)


def _require_real(name, value):
    """bool・文字列・Noneなど、数値でない値をValueErrorで弾く。"""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number")


def _validate_scalars(positive, nonnegative):
    for name, value in positive.items():
        _require_real(name, value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0")
    for name, value in nonnegative.items():
        _require_real(name, value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and >= 0")


def _validate_grid(nx, nz):
    for name, value in dict(nx=nx, nz=nz).items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2:
            raise ValueError(f"{name} must be an integer >= 2")


class ConsolidationSimulator:
    """m・day・kPa単位。zは地表から下向き、沈下は固定底面を基準に下向き正。

    2次元拡散を解いた後、各鉛直柱にεx=εy=0の弾性換算を適用する。
    変形と水圧の連成、有限ひずみ、せん断・水平力の釣合いは解かない。
    """

    # 既存APIと局所的な検証を保つため、長い初期化と引数を維持する。API改版時に再検討する。
    def __init__(
        self,
        length: float = 10.0,
        depth: float = 5.0,
        nx: int = 50,
        nz: int = 25,
        cv: float = 1.0,
        E: float = 5000.0,
        dt: float = 0.1,
        total_time: float = 3650.0,
        device: str = "auto",
        load_intensity: float = 100.0,
        load_width: float = 2.0,
        poisson_ratio: float = 0.3,
    ):
        positive = dict(length=length, depth=depth, E=E, dt=dt,
                        total_time=total_time, load_width=load_width)
        nonnegative = dict(cv=cv, load_intensity=load_intensity)
        _validate_scalars(positive, nonnegative)
        _validate_grid(nx, nz)
        _require_real("poisson_ratio", poisson_ratio)
        if not 0 <= poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must satisfy 0 <= value < 0.5")
        if device not in ("auto", "cpu"):
            raise ValueError("device must be 'auto' or 'cpu'; GPU backends are not supported")

        self.length, self.depth = length, depth
        self.nx, self.nz = nx, nz
        self.dx, self.dz = length / (nx - 1), depth / (nz - 1)
        self.cv, self.E, self.nu = cv, E, poisson_ratio
        self.load_intensity, self.load_width = load_intensity, load_width
        self.device, self.total_time = "cpu", total_time
        if self.dx == 0 or self.dz == 0:
            raise ValueError("grid spacing is too small")
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            self.D = E * (1 - poisson_ratio) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
            inverse_spacing = np.reciprocal(np.square([self.dx, self.dz]))
            rate = cv * inverse_spacing.sum()
        if not math.isfinite(self.D) or not math.isfinite(rate):
            raise ValueError("derived material or diffusion coefficients are not finite")
        # 無条件安定でも大きなdtでは振動する。最悪モードの h=½Δtλ (λmax≤4cv(1/Δx²+1/Δz²)) を
        # -1 以上に保ち、各モードの増幅率 (1+h)/(1-h) を非負にする。
        step_limit = min(dt, 0.5 / rate) if rate > 0 else dt
        if not math.isfinite(total_time / step_limit):
            raise ValueError("total_time / time step must be finite")
        self.nt = max(1, math.ceil(total_time / step_limit))
        self.dt = total_time / self.nt
        self._setup_solver()
        self.total_stress = self._strip_stress()
        self.reset()

    def _setup_solver(self):
        x_weights = np.ones(self.nx)
        x_weights[[0, -1]] = 0.5
        z_weights = np.ones(self.nz - 1)
        z_weights[-1] = 0.5
        x_rates, self._x_basis = _weighted_modes(_second_difference(self.nx), x_weights)
        z_rates, self._z_basis = _weighted_modes(_second_difference(self.nz)[1:, 1:], z_weights)
        x_rates[-1] = 0.0  # Neumann定数モードを丸め誤差で減衰させない。
        rates = self.cv * (z_rates[:, None] / self.dz**2 + x_rates[None, :] / self.dx**2)
        half_step = 0.5 * self.dt * rates
        numerator_change = np.where(half_step >= -1, half_step, -half_step - 2)
        # log1pで微小な減衰を保持する。増幅率0のlog(0)=-infは意図した極限。
        with np.errstate(divide="ignore"):
            log_amplification = np.log1p(numerator_change) - np.log1p(-half_step)
        self._decay = np.exp(float(self.nt) * log_amplification)
        if self.nt % 2:
            self._decay *= np.sign(1 + half_step)
        self._weights = np.sqrt(z_weights[:, None] * x_weights[None, :])

    def _strip_stress(self):
        """弾性半無限体の等分布帯荷重。地表の荷重端では片側極限の平均値。"""
        x = np.linspace(-self.length / 2, self.length / 2, self.nx)[None, :]
        z = np.linspace(0, self.depth, self.nz)[:, None]
        left = np.arctan2(x + self.load_width / 2, z)
        right = np.arctan2(x - self.load_width / 2, z)
        angle = left - right
        return self.load_intensity / np.pi * (angle + np.sin(angle) * np.cos(left + right))

    def _update_displacement(self):
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            self.strain_z = (self.total_stress - self.u) / self.D
            increments = 0.5 * (self.strain_z[:-1] + self.strain_z[1:]) * self.dz
            self.disp_z = np.zeros_like(self.u)
            self.disp_z[:-1] = np.cumsum(increments[::-1], axis=0)[::-1]
        self.disp_x = np.zeros_like(self.u)

    def run_simulation(self, verbose: bool = True) -> np.ndarray:
        """現在の水圧からtotal_timeだけ進める。返す配列は(nz, nx)の沈下量[m]。"""
        start = time.perf_counter()
        coefficients = self._z_basis.T @ (self.u[1:] * self._weights) @ self._x_basis
        pressure = self._z_basis @ (coefficients * self._decay) @ self._x_basis.T
        self.u = np.vstack([np.zeros(self.nx), pressure / self._weights])
        if not np.isfinite(self.u).all():
            raise FloatingPointError("non-finite pore pressure; no clipping or substitution was applied")
        self._update_displacement()
        if verbose:
            print(f"Simulation time: {time.perf_counter() - start:.3f} seconds (NumPy CPU)")
            print(f"Crank-Nicolson: {self.nt} steps, dt={self.dt:.6g} day (evaluated in eigenmodes)")
            print(f"Surface center settlement: {self.disp_z[0, self.nx // 2] * 1000:.2f} mm")
            print(f"Remaining excess pore pressure: {self.u.max():.2f} kPa")
            print(f"Consolidation degree (pressure integral): {self.get_consolidation_degree():.1f}%")
        return self.disp_z.copy()

    def get_current_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return tuple(field.copy() for field in (self.disp_z, self.disp_x, self.u, self.strain_z))

    def _pressure_integral(self):
        return float(np.trapezoid(np.trapezoid(self.u, dx=self.dx, axis=1), dx=self.dz))

    def get_consolidation_degree(self) -> float:
        """初期の排水境界適用後の水圧積分に対する消散率[%]。沈下率とは区別する。"""
        if self._initial_pressure_integral == 0:
            return 100.0
        return 100 * (1 - self._pressure_integral() / self._initial_pressure_integral)

    def reset(self):
        self.u = self.total_stress.copy()
        self.u[0] = 0.0
        self._initial_pressure_integral = self._pressure_integral()
        self._update_displacement()

    def plot_results(self, save_path=None, show=False, figsize=(18, 5), dpi=300):
        x = np.linspace(0, self.length, self.nx)
        z = np.linspace(0, self.depth, self.nz)
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fields = (self.disp_z, self.disp_x, self.u)
        references = (self.depth, self.depth, self.load_intensity)
        titles = ("Settlement (fixed base)", "Horizontal displacement (constrained)", "Excess pore pressure")
        units = ("m", "m", "kPa")
        for ax, field, reference, title, unit in zip(axes, fields, references, titles, units):
            if np.ptp(field) == 0:
                # 一定の場は丸め誤差の等値線とカラーバーを作らず、値だけを示す。
                ax.text(0.5, 0.5, f"{field.flat[0]:.3g} {unit} everywhere",
                        transform=ax.transAxes, ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            elif is_negligible_field(field, reference):
                ax.text(0.5, 0.5, f"≈0 (max {np.max(np.abs(field)):.3g} {unit})",
                        transform=ax.transAxes, ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                contour = ax.contourf(x, z, field, levels=50)
                fig.colorbar(contour, ax=ax, label=unit)
            ax.set(xlabel="Distance (m)", ylabel="Depth (m)", title=title)
            ax.invert_yaxis()
        fig.tight_layout()
        try:
            if save_path:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            if show:
                show_if_interactive()
        finally:
            plt.close(fig)
