"""三笠／Gibson 系の有限ひずみ圧密を物質座標で解く。

一定係数の線形拡散（`consolidation.py`）と違い、次の要素を扱う。

- 物質座標 ``ζ = ∫ dz/(1+e)``（固体体積に基づく座標、地表で0、下向き正）
- 圧縮則 ``e = e0 - Cc log10(σ'/σ'_ref)``
- 透水則 ``k = k0 10^{-(e0-e)/Ck}``
- 自重 ``γ_w (G_s + e)`` による全応力と、排水にともなう再配分
- 移動表面（沈下）と、固体体積一定の座標

支配方程式（鉛直一次元、``u`` は過剰間隙水圧）:

    ∂e/∂t = ∂/∂ζ [ k(e)/(γ_w (1+e)) · ∂u/∂ζ ]
    ∂u/∂ζ = γ_w (G_s + e) + (1/a_v) ∂e/∂ζ,   a_v = Cc/(σ' ln10)

平面ひずみ二次元への拡張では ``εx = 0`` を仮定し、水平方向の流れ

    ∂/∂ξ [ (k/γ_w) ∂u/∂ξ ],   ∂u/∂ξ = ∂σ_v/∂ξ + (1/a_v) ∂e/∂ξ

を加える。この二次元化は著者による拡張であり、三笠が公表した理論ではない。

数値解法は、セル中心の有限体積と後退Euler（陰的）。面のフラックスは ``∂u/∂ζ`` の
離散差から直接作るため、自重と釣り合った状態（``u = 0``）が厳密に保たれる。
時間刻みは安定性で制限されないが、反復が収束しない場合は時間刻みを自動的に分割し、
それでも収束しなければ例外で報告する。
"""

import math
import time

import matplotlib.pyplot as plt
import numpy as np

from consolidation import _require_real, _validate_scalars, is_negligible_field, show_if_interactive

LN10 = math.log(10.0)


def _tridiagonal(lower, diag, upper, rhs):
    """三重対角方程式を Thomas 法で解く（1列、純Python）。"""
    size = len(diag)
    c = [0.0] * size
    d = [0.0] * size
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, size):
        pivot = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / pivot
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / pivot
    x = [0.0] * size
    x[-1] = d[-1]
    for i in range(size - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def _tridiagonal_batch(lower, diag, upper, rhs):
    """形状 (size, count) の三重対角方程式を、列ごとに Thomas 法で解く。"""
    size, count = diag.shape
    c = np.zeros_like(upper)
    d = np.zeros_like(rhs)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, size):
        pivot = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / pivot
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / pivot
    x = np.zeros_like(rhs)
    x[-1] = d[-1]
    for i in range(size - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


class FiniteStrainConsolidation:
    """m・day・kPa単位の有限ひずみ圧密。``nx == 1`` は鉛直一次元。

    座標は固体体積に基づく物質座標 ``ζ`` で、地表が ``ζ=0``、増加方向が下向き。
    沈下は固定底面を基準に下向き正で返す。せん断変形、水平力の釣合い、二次圧密、
    載荷にともなう全応力分布の変化は解かない。
    """

    def __init__(
        self,
        length: float = 20.0,
        depth: float = 10.0,
        nx: int = 21,
        nz: int = 21,
        e0: float = 2.0,
        Cc: float = 0.6,
        Ck: float = 0.6,
        k0: float = 0.02,
        sigma_ref: float = 10.0,
        sigma_top: float = 10.0,
        specific_gravity: float = 2.65,
        unit_weight_water: float = 9.81,
        load_intensity: float = 50.0,
        load_width: float = 4.0,
        dt: float = 5.0,
        total_time: float = 3650.0,
        max_iterations: int = 60,
        max_splits: int = 8,
        tolerance: float = 1e-9,
    ):
        positive = dict(length=length, depth=depth, e0=e0, Cc=Cc, Ck=Ck, k0=k0,
                        sigma_ref=sigma_ref, sigma_top=sigma_top,
                        specific_gravity=specific_gravity, unit_weight_water=unit_weight_water,
                        load_width=load_width, dt=dt, total_time=total_time,
                        tolerance=tolerance)
        nonnegative = dict(load_intensity=load_intensity)
        _validate_scalars(positive, nonnegative)
        for name, value in dict(nx=nx, nz=nz, max_iterations=max_iterations,
                                max_splits=max_splits).items():
            _require_real(name, value)
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if nz < 2:
            raise ValueError("nz must be an integer >= 2")
        if Ck <= 0:
            raise ValueError("Ck must be finite and > 0")

        self.length, self.depth, self.nx, self.nz = length, depth, nx, nz
        self.e0, self.Cc, self.Ck, self.k0 = e0, Cc, Ck, k0
        self.sigma_ref, self.sigma_top = sigma_ref, sigma_top
        self.gs, self.gamma_w = specific_gravity, unit_weight_water
        self.load_intensity, self.load_width = load_intensity, load_width
        self.total_time = total_time
        self.max_iterations, self.max_splits, self.tolerance = max_iterations, max_splits, tolerance
        self.d_xi = length / nx
        self.nt = max(1, math.ceil(total_time / dt))
        self.dt = total_time / self.nt
        self._build_initial_state()
        self.reset()

    # ------------------------------------------------------------------ 材料則

    def _void_ratio_at(self, sigma):
        """有効応力を与えたときの間隙比。"""
        return self.e0 - self.Cc * np.log10(sigma / self.sigma_ref)

    def _effective_stress_at(self, void_ratio):
        return self.sigma_ref * np.power(10.0, (self.e0 - void_ratio) / self.Cc)

    def _permeability_at(self, void_ratio):
        return self.k0 * np.power(10.0, -(self.e0 - void_ratio) / self.Ck)

    def _self_weight_profile(self, d_zeta, sigma_surface):
        """自重と地表応力で釣り合う間隙比と有効応力の離散プロファイル。

        台形則の再帰 ``σ'_i = σ'_{i-1} + γ_w Δζ (G_s + (e_{i-1}+e_i)/2)`` を、
        ``e`` が ``σ'`` に依存するため各セルで不動点反復して解く。
        """
        sigma = np.empty(self.nz)
        void = np.empty(self.nz)
        guess = self._void_ratio_at(sigma_surface)
        for _ in range(64):
            sigma[0] = sigma_surface + 0.5 * self.gamma_w * d_zeta * (self.gs + guess)
            updated = self._void_ratio_at(sigma[0])
            if abs(updated - guess) < 1e-14:
                guess = updated
                break
            guess = updated
        void[0] = guess
        for i in range(1, self.nz):
            e_above = void[i - 1]
            guess = self._void_ratio_at(sigma[i - 1])
            for _ in range(64):
                sigma[i] = sigma[i - 1] + self.gamma_w * d_zeta * (self.gs + 0.5 * (e_above + guess))
                updated = self._void_ratio_at(sigma[i])
                if abs(updated - guess) < 1e-14:
                    guess = updated
                    break
                guess = updated
            void[i] = guess
        return void

    def _initial_solid_thickness(self):
        """初期層厚 ``depth`` を与える固体厚 ``ζ`` を二分法で求める。"""
        def residual(solid):
            void = self._self_weight_profile(solid / self.nz, self.sigma_top)
            return (solid / self.nz) * float(np.sum(1.0 + void)) - self.depth

        low, high = 0.0, self.depth
        for _ in range(96):
            middle = 0.5 * (low + high)
            if residual(middle) < 0:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    def _build_initial_state(self):
        self.d_zeta = self._initial_solid_thickness() / self.nz
        profile = self._self_weight_profile(self.d_zeta, self.sigma_top)
        self.e_initial = np.broadcast_to(profile[:, None], (self.nz, self.nx)).copy()
        x = np.linspace(0.0, self.length, self.nx)
        if self.nx == 1:
            self.sigma_load = np.full((self.nz, 1), self.load_intensity)
        else:
            # 初期の物理深さ（セル中心）で帯荷重の弾性応力増分を評価する。
            z = self.d_zeta * (np.cumsum(1.0 + profile) - 0.5 * (1.0 + profile))
            x_center = x[None, :] - 0.5 * self.length
            left = np.arctan2(x_center + 0.5 * self.load_width, z[:, None])
            right = np.arctan2(x_center - 0.5 * self.load_width, z[:, None])
            angle = left - right
            self.sigma_load = self.load_intensity / np.pi * (
                angle + np.sin(angle) * np.cos(left + right))
        self.e_surface = self._void_ratio_at(self.sigma_top + self.sigma_load[0])

    # ------------------------------------------------------------ 応力とフラックス

    def _vertical_weight(self, void_ratio):
        """地表からセル中心までの土の重さ ``γ_w ∫ (G_s + e) dζ``。

        ``_self_weight_profile`` の台形則と同じ離散化にする（初期状態で ``u = σ_load``）。
        """
        weight = self.gamma_w * (self.gs + void_ratio)
        return self.d_zeta * (np.cumsum(weight, axis=0) - 0.5 * weight)

    def _total_stress(self, void_ratio):
        return self.sigma_top + self.sigma_load + self._vertical_weight(void_ratio)

    def _pore_pressure(self, void_ratio):
        return self._total_stress(void_ratio) - self._effective_stress_at(void_ratio)

    def _face_values(self, void_ratio):
        face = np.empty((self.nz + 1, self.nx))
        face[1:self.nz] = 0.5 * (void_ratio[:-1] + void_ratio[1:])
        face[0] = self.e_surface
        face[self.nz] = void_ratio[-1]
        return face

    def _vertical_conductivity(self, void_ratio):
        """面での係数 ``k/(γ_w (1+e))``。"""
        face = self._face_values(void_ratio)
        return self._permeability_at(face) / (self.gamma_w * (1.0 + face))

    def _vertical_flux(self, void_ratio):
        """面での ``F = k/(γ_w (1+e)) ∂u/∂ζ``。底面は非排水で ``F=0``。

        ``u`` の離散差（セル中心値と、地表の排水条件 ``u = 0``）をそのまま使う。
        支配方程式の表式どおりであり、自重と釣り合った状態（``u = 0``）では面の
        フラックスが厳密に 0 になる（無載荷で見かけの圧密が起きない）。
        """
        conductivity = self._vertical_conductivity(void_ratio)
        pressure = self._pore_pressure(void_ratio)
        flux = np.zeros((self.nz + 1, self.nx))
        flux[0] = conductivity[0] * pressure[0] / (0.5 * self.d_zeta)
        flux[1:self.nz] = conductivity[1:self.nz] * (
            pressure[1:] - pressure[:-1]) / self.d_zeta
        return flux

    def _vertical_divergence(self, void_ratio):
        """``∂F/∂ζ``。地表から流入し底面で止まる向き。"""
        flux = self._vertical_flux(void_ratio)
        return (flux[1:] - flux[:-1]) / self.d_zeta

    def _horizontal_flux(self, void_ratio, sigma_total):
        """面での ``G = (k/γ_w) ∂u/∂ξ``。端は非排水で ``G=0``。"""
        face = 0.5 * (void_ratio[:, :-1] + void_ratio[:, 1:])
        sigma = self._effective_stress_at(face)
        perm = self._permeability_at(face)
        gradient = ((sigma_total[:, 1:] - sigma_total[:, :-1])
                    + (sigma * LN10 / self.Cc) * (void_ratio[:, 1:] - void_ratio[:, :-1])) / self.d_xi
        flux = np.zeros((self.nz, self.nx + 1))
        flux[:, 1:-1] = (perm / self.gamma_w) * gradient
        return flux

    def _horizontal_divergence(self, void_ratio, sigma_total):
        if self.nx == 1:
            return np.zeros_like(void_ratio)
        flux = self._horizontal_flux(void_ratio, sigma_total)
        return (flux[:, 1:] - flux[:, :-1]) / self.d_xi

    # ---------------------------------------------------------------- 陰的ソルバー

    def _vertical_residual(self, previous, void_ratio, step, lagged):
        """後退Eulerの残差 ``e - e^n - Δt (∂F/∂ζ + 水平寄与)``。"""
        return void_ratio - previous - step * (self._vertical_divergence(void_ratio) + lagged)

    def _vertical_jacobian(self, void_ratio, step):
        """``_vertical_residual`` の三重対角ヤコビアン（Newton 法、対角は正）。

        面 ``f`` のフラックス依存を ``p_f = ∂F_f/∂e_f``（下側セル）と
        ``m_f = ∂F_f/∂e_{f-1}``（上側セル、``m_0 = 0``）で表す。
        """
        face = self._face_values(void_ratio)
        perm = self._permeability_at(face)
        conductivity = perm / (self.gamma_w * (1.0 + face))
        d_conductivity = conductivity * (LN10 / self.Ck - 1.0 / (1.0 + face))
        pressure = self._pore_pressure(void_ratio)
        sigma = self._effective_stress_at(void_ratio)
        slope = LN10 / (self.Cc * self.d_zeta)
        gradient = (pressure[1:] - pressure[:-1]) / self.d_zeta
        p = np.zeros_like(face)
        m = np.zeros_like(face)
        p[0] = conductivity[0] * (self.gamma_w + 2.0 * sigma[0] * slope)
        p[1:self.nz] = (0.5 * d_conductivity[1:self.nz] * gradient
                        + conductivity[1:self.nz] * (0.5 * self.gamma_w + sigma[1:] * slope))
        m[1:self.nz] = (0.5 * d_conductivity[1:self.nz] * gradient
                        + conductivity[1:self.nz] * (0.5 * self.gamma_w - sigma[:-1] * slope))
        scale = step / self.d_zeta
        lower = np.zeros((self.nz, self.nx))
        upper = np.zeros((self.nz, self.nx))
        diag = np.ones((self.nz, self.nx))
        lower[1:] = scale * m[1:self.nz]
        upper[:-1] = -scale * p[1:self.nz]
        diag[0] = 1.0 - scale * (m[1] - p[0])
        diag[1:self.nz - 1] = 1.0 - scale * (m[2:self.nz] - p[1:self.nz - 1])
        diag[self.nz - 1] = 1.0 + scale * p[self.nz - 1]
        return lower, diag, upper

    def _horizontal_residual(self, previous, void_ratio, sigma_total, step, lagged):
        return void_ratio - previous - step * (self._horizontal_divergence(void_ratio, sigma_total) + lagged)

    def _horizontal_jacobian(self, void_ratio, sigma_total, step):
        """水平方向 Newton の三重対角ヤコビアン。``σ_v`` の ``e`` 依存は固定する。"""
        lower = np.zeros((self.nz, self.nx))
        upper = np.zeros((self.nz, self.nx))
        diag = np.ones((self.nz, self.nx))
        if self.nx == 1:
            return lower, diag, upper
        face = 0.5 * (void_ratio[:, :-1] + void_ratio[:, 1:])
        sigma = self._effective_stress_at(face)
        perm = self._permeability_at(face)
        beta = sigma * LN10 / self.Cc
        p = perm * beta / (self.gamma_w * self.d_xi**2)
        d_perm = perm * LN10 / self.Ck
        d_beta = -sigma * LN10**2 / self.Cc**2
        d_p = -(d_perm * beta + perm * d_beta) / (self.gamma_w * self.d_xi**2)
        q = d_p * (void_ratio[:, 1:] - void_ratio[:, :-1]) \
            - (d_perm / (self.gamma_w * self.d_xi**2)) * (sigma_total[:, 1:] - sigma_total[:, :-1])
        scale = step / self.d_xi
        lower[:, 1:] = -scale * (p - 0.5 * q)
        upper[:, :-1] = -scale * (p + 0.5 * q)
        diag[:, 0] = 1.0 + scale * (p[:, 0] - 0.5 * q[:, 0])
        diag[:, 1:self.nx - 1] = 1.0 + scale * (
            p[:, :-1] + p[:, 1:] + 0.5 * q[:, :-1] - 0.5 * q[:, 1:])
        diag[:, self.nx - 1] = 1.0 + scale * (p[:, self.nx - 2] + 0.5 * q[:, self.nx - 2])
        return lower, diag, upper

    def _solve_vertical(self, lower, diag, upper, rhs):
        if self.nx == 1:
            solution = _tridiagonal(lower[:, 0].tolist(), diag[:, 0].tolist(),
                                    upper[:, 0].tolist(), rhs[:, 0].tolist())
            return np.array(solution)[:, None]
        return _tridiagonal_batch(lower, diag, upper, rhs)

    def _solve_horizontal(self, lower, diag, upper, rhs):
        return _tridiagonal_batch(lower.T, diag.T, upper.T, rhs.T).T

    def _newton_solve(self, residual, jacobian, solve, initial, threshold):
        """1方向の非線形陰的方程式を Newton 法で解く（残差が減るまで半減する）。

        ``threshold`` は残差の許容値。1ステップの変化量に比例させ、
        微小振幅（``Cc`` が小さい）問題でも止まりすぎないようにする。
        """
        void = initial
        value = residual(void)
        for _ in range(self.max_iterations):
            norm = float(np.max(np.abs(value)))
            if norm <= threshold:
                return void
            lower, diag, upper = jacobian(void)
            update = solve(lower, diag, upper, -value)
            damping = 1.0
            for _ in range(12):
                trial = void + damping * update
                trial_value = residual(trial)
                trial_norm = float(np.max(np.abs(trial_value)))
                if trial_norm < norm or trial_norm <= threshold:
                    break
                damping *= 0.5
            else:
                raise RuntimeError("Newton line search could not reduce the residual")
            void, value = trial, trial_value
        raise RuntimeError(f"Newton did not converge in {self.max_iterations} iterations")

    def _advance_step(self, step):
        """1時間刻み進める。反復が収束しなければ ``RuntimeError``。"""
        previous = self.e
        current = previous.copy()
        # 残差の許容値はステップ内の変化量を基準にする。丸め誤差（残差は大きい数の
        # 差で作られる）より十分大きい床を置き、達成不能な閾値で分割が連鎖しないようにする。
        change = abs(step) * float(np.max(np.abs(
            self._vertical_divergence(previous)
            + self._horizontal_divergence(previous, self._total_stress(previous)))))
        threshold = max(self.tolerance * change,
                        1e-13 * max(1.0, float(np.max(np.abs(previous)))))
        for _ in range(self.max_iterations):
            sigma_total = self._total_stress(current)
            lagged_horizontal = self._horizontal_divergence(current, sigma_total)
            vertical = self._newton_solve(
                lambda void: self._vertical_residual(previous, void, step, lagged_horizontal),
                lambda void: self._vertical_jacobian(void, step),
                self._solve_vertical, current, threshold)
            if self.nx == 1:
                updated = vertical
            else:
                lagged_vertical = self._vertical_divergence(vertical)
                updated = self._newton_solve(
                    lambda void: self._horizontal_residual(previous, void, sigma_total, step, lagged_vertical),
                    lambda void: self._horizontal_jacobian(void, sigma_total, step),
                    self._solve_horizontal, vertical, threshold)
            difference = float(np.max(np.abs(updated - current)))
            current = updated
            if difference <= threshold:
                return current
        raise RuntimeError(
            f"finite-strain iteration did not converge in {self.max_iterations} iterations")

    def _advance(self, step, depth=0):
        try:
            return self._advance_step(step)
        except RuntimeError as error:
            if depth >= self.max_splits:
                raise RuntimeError(
                    f"time step {step:.6g} day did not converge after {depth} subdivisions") from error
            self.split_count += 1
            half = self._advance(step / 2.0, depth + 1)
            self.e = half
            return self._advance(step / 2.0, depth + 1)

    # -------------------------------------------------------------------- 出力

    def _update_outputs(self):
        self.sigma_total = self._total_stress(self.e)
        self.u = self.sigma_total - self._effective_stress_at(self.e)
        self.strain_z = (self.e_initial - self.e) / (1.0 + self.e_initial)
        decrement = (self.e_initial - self.e) * self.d_zeta
        self.disp_z = np.cumsum(decrement[::-1], axis=0)[::-1]
        self.disp_x = np.zeros_like(self.e)

    def _pressure_integral(self):
        return float(np.sum(self.u) * self.d_zeta * self.d_xi)

    def get_consolidation_degree(self) -> float:
        """水圧積分の消散率[%]。沈下量に基づく圧密度とは区別する。

        初期の過剰水圧が応力スケールに比べて無視できる（載荷なし）場合は 100 %。
        """
        if (self._initial_pressure_scale <= 1e-9 * max(self.sigma_top, self.load_intensity)
                or self._initial_pressure_integral == 0):
            return 100.0
        return 100 * (1 - self._pressure_integral() / self._initial_pressure_integral)

    def reset(self):
        self.e = self.e_initial.copy()
        self.split_count = 0
        self._update_outputs()
        self._initial_pressure_integral = self._pressure_integral()
        self._initial_pressure_scale = float(np.max(np.abs(self.u)))

    def get_current_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return tuple(field.copy() for field in (self.disp_z, self.disp_x, self.u, self.strain_z))

    def run_simulation(self, verbose: bool = True) -> np.ndarray:
        """現在の状態から ``total_time`` だけ進める。返す配列は (nz, nx) の沈下量[m]。"""
        start = time.perf_counter()
        for _ in range(self.nt):
            self.e = self._advance(self.dt)
            if not np.isfinite(self.e).all():
                raise FloatingPointError("non-finite void ratio; no clipping was applied")
        self._update_outputs()
        if not np.isfinite(self.u).all():
            raise FloatingPointError("non-finite pore pressure; no clipping was applied")
        if verbose:
            print(f"Simulation time: {time.perf_counter() - start:.3f} seconds (NumPy CPU)")
            print(f"Backward Euler: {self.nt} steps, dt={self.dt:.6g} day, "
                  f"{self.split_count} automatic subdivisions")
            print(f"Surface center settlement: {self.disp_z[0, self.nx // 2] * 1000:.2f} mm")
            print(f"Void ratio at surface center: {self.e[0, self.nx // 2]:.4f} "
                  f"(initial {self.e_initial[0, self.nx // 2]:.4f})")
            print(f"Remaining excess pore pressure (max |u|): {np.abs(self.u).max():.2f} kPa")
            print(f"Consolidation degree (pressure integral): {self.get_consolidation_degree():.1f}%")
        return self.disp_z.copy()

    def plot_results(self, save_path=None, show=True, figsize=(18, 5), dpi=300):
        x = np.linspace(0, self.length, self.nx)
        z = self.d_zeta * (np.cumsum(1.0 + self.e, axis=0) - 0.5 * (1.0 + self.e))
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fields = (self.disp_z, self.u, self.e)
        references = (self.depth, max(self.load_intensity, 1e-30), max(self.e0, 1e-30))
        titles = ("Settlement (fixed base)", "Excess pore pressure", "Void ratio")
        units = ("m", "kPa", "–")
        for ax, field, reference, title, unit in zip(axes, fields, references, titles, units):
            if np.ptp(field) == 0:
                ax.text(0.5, 0.5, f"{field.flat[0]:.4g} {unit} everywhere",
                        transform=ax.transAxes, ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            elif is_negligible_field(field, reference):
                ax.text(0.5, 0.5, f"≈0 (max {np.max(np.abs(field)):.3g} {unit})",
                        transform=ax.transAxes, ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            elif self.nx == 1:
                values = field[:, 0] * 1000.0 if unit == "m" else field[:, 0]
                label = "mm" if unit == "m" else unit
                ax.plot(values, z[:, 0], "-")
                ax.set(xlabel=f"{title} ({label})")
            else:
                mesh_x = np.broadcast_to(x, field.shape)
                contour = ax.contourf(mesh_x, z, field, levels=50)
                fig.colorbar(contour, ax=ax, label=unit)
                ax.set(xlabel="Distance (m)")
            ax.set(ylabel="Depth (m)", title=title)
            ax.invert_yaxis()
            ax.set_ylim(self.depth, 0)
        fig.tight_layout()
        try:
            if save_path:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            if show:
                show_if_interactive()
        finally:
            plt.close(fig)


def run_example(save_path):
    simulator = FiniteStrainConsolidation()
    simulator.run_simulation()
    simulator.plot_results(save_path=save_path)
    return simulator
