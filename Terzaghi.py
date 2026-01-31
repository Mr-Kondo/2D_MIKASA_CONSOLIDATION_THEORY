import math
import time
from contextlib import contextmanager
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import mlx.core as mx
from tqdm import tqdm


@contextmanager
def timer(name: str):
    """実行時間計測用コンテキストマネージャ"""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.3f} seconds")


class TerzaghiConsolidationSimulator:
    """Terzaghi法による二次元圧密沈下シミュレーター

    側方ひずみゼロ(εx=0)を仮定した2次元圧密理論(Terzaghi)
    拘束圧密条件(oedometer condition)でひずみから変位を計算
    """

    def __init__(
        self,
        length: float = 10.0,
        depth: float = 5.0,
        nx: int = 50,
        nz: int = 25,
        cv: float = 1.0,  # 圧密係数 (m²/day)
        E: float = 5000.0,  # ヤング率 (kPa)
        dt: float = 0.1,  # 時間刻み (day)
        total_time: float = 3650.0,  # 総時間 (day) = 10年
        device: str = "auto",
        load_intensity: float = 100.0,  # 荷重強度 (kPa)
        load_width: float = 2.0,  # 荷重幅 (m)
        poisson_ratio: float = 0.3,  # ポアソン比
    ):
        """
        Parameters
        ----------
        length : 土層の横幅(m)
        depth : 土層の深さ(m)
        nx : x方向の格子点数
        nz : z方向の格子点数
        cv : 圧密係数 (m²/day)
        E : ヤング率 (kPa)
        dt : 時間刻み (day)
        total_time : 総シミュレーション時間 (day)
        device : 計算デバイス ('auto', 'mps', 'cuda', 'cpu')
        load_intensity : 荷重強度 (kPa)
        load_width : 荷重幅 (m)
        poisson_ratio : ポアソン比
        """
        self._validate_parameters(
            length=length,
            depth=depth,
            nx=nx,
            nz=nz,
            cv=cv,
            E=E,
            dt=dt,
            total_time=total_time,
            load_intensity=load_intensity,
            load_width=load_width,
            poisson_ratio=poisson_ratio,
        )

        # デバイス設定
        self.device = self._select_device(device)
        print(f"Running on: {self.device} (Apple Silicon with MLX)")

        # 幾何パラメータ
        self.length = length
        self.depth = depth
        self.nx = nx
        self.nz = nz
        self.dx = length / (nx - 1)
        self.dz = depth / (nz - 1)

        # 土質パラメータ
        self.cv = cv  # 圧密係数
        self.E = E  # ヤング率
        self.nu = poisson_ratio  # ポアソン比

        # Terzaghi拘束圧密係数 (oedometer condition, εx = 0)
        # 拘束圧密係数: D = E*(1-ν)/[(1+ν)*(1-2ν)]
        self.D = E * (1 - poisson_ratio) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))  # kPa

        # 体積圧縮係数(拘束圧密係数から計算)
        self.mv = 1.0 / self.D  # 1/kPa

        # 透水係数
        self.k = cv * self.mv * 9.81  # m/day

        # 時間パラメータ
        self.dt = dt
        self.total_time = total_time
        self.nt = int(total_time / dt)

        # 荷重パラメータ
        self.load_intensity = load_intensity
        self.load_width = load_width

        # 状態変数の初期化
        dtype = self._get_dtype()
        self.u = mx.zeros((nz, nx), dtype=dtype)  # 過剰間隙水圧 (kPa)
        self.disp_x = mx.zeros((nz, nx), dtype=dtype)  # 水平変位 (m) - Terzaghiでは常にゼロ
        self.disp_z = mx.zeros((nz, nx), dtype=dtype)  # 鉛直変位 (m)
        self.strain_z = mx.zeros((nz, nx), dtype=dtype)  # 鉛直ひずみ

        # 初期過剰間隙水圧の設定(荷重による即時発生)
        self._apply_initial_excess_pore_pressure()

        # 陰解法のセットアップ
        self._setup_implicit_solver()

    @staticmethod
    def _validate_parameters(
        *,
        length: float,
        depth: float,
        nx: int,
        nz: int,
        cv: float,
        E: float,
        dt: float,
        total_time: float,
        load_intensity: float,
        load_width: float,
        poisson_ratio: float,
    ):
        """入力条件の妥当性チェック"""
        if length <= 0 or depth <= 0:
            raise ValueError("length and depth must be positive")
        if nx < 2 or nz < 2:
            raise ValueError("nx and nz must be >= 2")
        if cv < 0:
            raise ValueError("cv must be >= 0")
        if E <= 0:
            raise ValueError("E must be > 0")
        if dt <= 0 or total_time <= 0:
            raise ValueError("dt and total_time must be > 0")
        if load_width <= 0:
            raise ValueError("load_width must be > 0")
        if load_intensity < 0:
            raise ValueError("load_intensity must be >= 0")
        if not (0.0 < poisson_ratio < 0.49):
            raise ValueError("poisson_ratio must be between 0 and 0.49")

    def _select_device(self, device: str) -> str:
        """デバイスを選択

        MLXはMPS対応Appleシリコンで自動的にGPU加速を使用するため、
        デバイス選択は簡略化される
        """
        # MLXはMPS対応デバイスで自動的にGPU使用
        return "mlx"

    def _get_dtype(self) -> mx.Dtype:
        """MLXのデフォルトdtype(float32)を返す"""
        return mx.float32

    def _setup_implicit_solver(self):
        """陰解法のセットアップ

        Crank-Nicolson法の安定性係数を計算
        陰解法は無条件安定なので安定性条件は不要
        """
        self.alpha_x = self.cv * self.dt / (self.dx**2)
        self.alpha_z = self.cv * self.dt / (self.dz**2)

        if not math.isfinite(self.alpha_x) or not math.isfinite(self.alpha_z):
            raise ValueError("alpha coefficients are not finite; check parameters")

        print("Implicit method coefficients:")
        print(f"α_x = {self.alpha_x:.6f}")
        print(f"α_z = {self.alpha_z:.6f}")
        print(f"α_x + α_z = {self.alpha_x + self.alpha_z:.6f}")
        print(f"Constrained modulus D = {self.D:.2f} kPa")
        print("  Method: Crank-Nicolson (Unconditionally stable)")
        print("  Theory: Terzaghi (εx = 0, oedometer condition)")

    def _apply_initial_excess_pore_pressure(self):
        """初期過剰間隙水圧を設定

        Boussinesq解に基づく応力分布を簡易的に適用
        """
        # 荷重中心
        x_center = self.length / 2

        # 初期過剰間隙水圧をPython listで構築
        u_values = []
        for j in range(self.nz):
            row = []
            for i in range(self.nx):
                x_pos = i * self.dx
                dx_from_center = abs(x_pos - x_center)
                z_pos = j * self.dz + 0.01  # 微小値を加えてゼロ除算を回避

                # 矩形分布荷重からの応力増加(簡易化したBoussinesq解)
                if dx_from_center <= self.load_width / 2:
                    # 荷重直下：深さによる減衰
                    influence = 1.0 / (1.0 + (z_pos / self.load_width) ** 2)
                    row.append(self.load_intensity * influence)
                else:
                    # 荷重外側：水平距離と深さによる減衰
                    r = math.sqrt((dx_from_center - self.load_width / 2) ** 2 + z_pos**2)
                    influence = (z_pos / (r + z_pos)) ** 2
                    row.append(self.load_intensity * influence * 0.5)
            u_values.append(row)

        # MLX配列に変換
        self.u = mx.array(u_values, dtype=self._get_dtype())

    def _compute_consolidation_step(self):
        """1ステップの圧密計算(過剰間隙水圧の消散)

        2次元圧密方程式:
        ∂u/∂t = cv * (∂²u/∂x² + ∂²u/∂z²)

        Crank-Nicolson陰解法:
        (I + 0.5*dt*L) * u_{n+1} = (I - 0.5*dt*L) * u_n
        ここでLはラプラシアン演算子

        境界条件:
        - 上面: u = 0 (排水)
        - 下面: ∂u/∂z = 0 (非排水)
        - 側面: ∂u/∂x = 0 (対称)
        """
        # 右辺を計算: rhs = (I - 0.5*dt*L) * u_n
        laplacian_u = self._apply_laplacian(self.u)
        rhs = self.u - 0.5 * laplacian_u

        # 境界条件を適用
        rhs_copy = mx.array(rhs)
        # 上面排水: u[0, :] = 0
        rhs_copy = mx.concatenate([mx.zeros((1, rhs.shape[1]), dtype=rhs.dtype), rhs_copy[1:]], axis=0)

        # 非有限値の処理
        rhs_copy = mx.where(mx.isfinite(rhs_copy), rhs_copy, mx.zeros_like(rhs_copy))

        # 左辺を解く: (I + 0.5*dt*L) * u_{n+1} = rhs
        # Jacobi反復法で解く
        u_new = self._conjugate_gradient_solve(rhs_copy, max_iter=100, tol=1e-4)

        # 非有限値の処理
        u_new = mx.where(mx.isfinite(u_new), u_new, mx.zeros_like(u_new))

        # 過剰間隙水圧は非負であり、初期値を超えない
        u_new = mx.clip(u_new, 0.0, self.load_intensity * 1.1)

        # 境界条件の再適用(確実性のため)
        nz, nx = u_new.shape
        # 上面排水
        u_new = mx.concatenate([mx.zeros((1, nx), dtype=u_new.dtype), u_new[1:]], axis=0)
        # 下面非排水: u[-1, :] = u[-2, :]
        u_new = mx.concatenate([u_new[:-1], mx.expand_dims(u_new[-2, :], axis=0)], axis=0)
        # 左側面対称: u[:, 0] = u[:, 1]
        u_new = mx.concatenate([mx.expand_dims(u_new[:, 1], axis=1), u_new[:, 1:]], axis=1)
        # 右側面対称: u[:, -1] = u[:, -2]
        u_new = mx.concatenate([u_new[:, :-1], mx.expand_dims(u_new[:, -2], axis=1)], axis=1)

        return u_new

    def _apply_laplacian(self, u) -> mx.array:
        """ラプラシアン演算子を適用

        L*u = cv * (∂²u/∂x² + ∂²u/∂z²) * dt

        Parameters
        ----------
        u : 間隙水圧場

        Returns
        -------
        結果 : dt*cv*∇²u
        """
        nz, nx = u.shape

        # より効率的な方法：内部領域を計算して結合
        # ∂²u/∂x² for interior points
        d2u_dx2_interior = u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]
        # ∂²u/∂z² for interior points
        d2u_dz2_interior = u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]

        # 内部点のラプラシアン
        laplacian_interior = self.alpha_x * d2u_dx2_interior + self.alpha_z * d2u_dz2_interior

        # 境界を0で埋める（上下）
        zeros_row = mx.zeros((1, laplacian_interior.shape[1]), dtype=u.dtype)
        laplacian_z_padded = mx.concatenate([zeros_row, laplacian_interior, zeros_row], axis=0)

        # 境界を0で埋める（左右）
        zeros_col = mx.zeros((nz, 1), dtype=u.dtype)
        result = mx.concatenate([zeros_col, laplacian_z_padded, zeros_col], axis=1)

        return result

    def _conjugate_gradient_solve(self, rhs, max_iter=100, tol=1e-4):
        """Jacobi反復法で線形方程式を解く

        (I + 0.5*dt*L) * x = rhs

        Jacobi法は対角優位な行列に対して安定で、
        CG法より条件数に強い

        Parameters
        ----------
        rhs : 右辺ベクトル
        max_iter : 最大反復回数
        tol : 収束許容誤差

        Returns
        -------
        x : 解ベクトル
        """
        nz, nx = self.u.shape

        # 初期解(前ステップの値を使用)
        x = mx.array(self.u)
        x_prev = mx.array(x)

        # Jacobi反復: x^{k+1} = (rhs - 0.5*L*x^k)
        for iteration in range(max_iter):
            # L*x^k を計算
            laplacian_x = self._apply_laplacian(x_prev)

            # x^{k+1} = rhs - 0.5*L*x^k (対角項は1なので除算不要)
            x_new = rhs - 0.5 * laplacian_x

            # 非有限値の処理
            x_new = mx.where(mx.isfinite(x_new), x_new, x_prev)

            # 境界条件の再適用
            # 上面排水: x[0, :] = 0
            x_new = mx.concatenate([mx.zeros((1, nx), dtype=x_new.dtype), x_new[1:]], axis=0)
            # 下面非排水: x[-1, :] = x[-2, :]
            x_new = mx.concatenate([x_new[:-1], mx.expand_dims(x_new[-2, :], axis=0)], axis=0)
            # 左側面対称: x[:, 0] = x[:, 1]
            x_new = mx.concatenate([mx.expand_dims(x_new[:, 1], axis=1), x_new[:, 1:]], axis=1)
            # 右側面対称: x[:, -1] = x[:, -2]
            x_new = mx.concatenate([x_new[:, :-1], mx.expand_dims(x_new[:, -2], axis=1)], axis=1)

            # 収束判定
            diff = mx.sum((x_new - x_prev) ** 2)
            diff_scalar = float(diff)

            if not mx.isfinite(diff) or diff_scalar < tol:
                x = x_new
                break

            x_prev = x_new
            x = x_new

        return x

    def _compute_strain_from_effective_stress(self):
        """有効応力からひずみを計算(Terzaghi拘束圧密理論)

        Terzaghi法では、側方ひずみゼロ(εx=0)を仮定した拘束圧密条件
        ひずみ: εz = σ'z / D
        ここで D は拘束圧密係数 (constrained modulus)
        """
        # 初期過剰間隙水圧を再計算
        dtype = self._get_dtype()
        u_values = []
        x_center = self.length / 2

        for j in range(self.nz):
            row = []
            for i in range(self.nx):
                x_pos = i * self.dx
                dx_from_center = abs(x_pos - x_center)
                z_pos = j * self.dz + 0.01

                if dx_from_center <= self.load_width / 2:
                    influence = 1.0 / (1.0 + (z_pos / self.load_width) ** 2)
                    row.append(self.load_intensity * influence)
                else:
                    r = math.sqrt((dx_from_center - self.load_width / 2) ** 2 + z_pos**2)
                    influence = (z_pos / (r + z_pos)) ** 2
                    row.append(self.load_intensity * influence * 0.5)
            u_values.append(row)

        u_initial = mx.array(u_values, dtype=dtype)

        # 有効応力増加 = 消散した過剰間隙水圧
        delta_sigma_eff = u_initial - self.u

        # 異常値の防止（負の値や無限大は0にクリップ）
        delta_sigma_eff = mx.clip(delta_sigma_eff, 0.0, self.load_intensity * 1.5)

        # Terzaghi拘束圧密条件: εx = 0
        # 拘束圧密係数を用いた鉛直ひずみ計算
        # εz = σ'z / D
        self.strain_z = delta_sigma_eff / self.D

        # ひずみの異常値防止
        self.strain_z = mx.clip(self.strain_z, -0.1, 0.1)

    def _compute_displacement_from_strain(self):
        """ひずみから変位を計算

        沈下量 = ∫ εz dz
        Terzaghi理論: 側方変位はゼロ (εx = 0)
        """
        # 各層のひずみから変位を積分
        # disp_z[j] = ∫_0^z εz dz
        displacement_increments = self.strain_z * self.dz
        self.disp_z = mx.cumsum(displacement_increments, axis=0)

        # Terzaghi理論: 水平変位はゼロ (拘束圧密条件 εx = 0)
        self.disp_x = mx.zeros_like(self.disp_z)

    def run_simulation(self, verbose: bool = True) -> np.ndarray:
        """シミュレーションを実行

        Parameters
        ----------
        verbose : 進捗表示の有無

        Returns
        -------
        result : 最終的な沈下量分布(NumPy配列)
        """
        with timer("Simulation time") if verbose else nullcontext():
            iterator = tqdm(range(self.nt), desc="Terzaghi Consolidation", disable=not verbose)

            for n in iterator:
                # 過剰間隙水圧の更新
                self.u = self._compute_consolidation_step()

                # 定期的にひずみと変位を更新
                if n % 10 == 0:
                    self._compute_strain_from_effective_stress()
                    self._compute_displacement_from_strain()

        # 最終ひずみと変位を計算
        self._compute_strain_from_effective_stress()
        self._compute_displacement_from_strain()

        # CPU転送
        result = np.array(self.disp_z)

        if verbose:
            print(f"Max settlement: {result.max() * 1000:.2f} mm")
            print(f"Surface center settlement: {result[0, self.nx // 2] * 1000:.2f} mm")
            print("Horizontal displacement: 0.00 mm (Terzaghi: εx = 0)")
            print(f"Remaining excess pore pressure: {float(mx.max(self.u)):.2f} kPa")
            print(f"Consolidation degree: {(1 - float(mx.max(self.u)) / self.load_intensity) * 100:.1f}%")

        return result

    def get_current_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """現在の状態を取得"""
        settlement = np.array(self.disp_z)
        horizontal_disp = np.array(self.disp_x)
        pore_pressure = np.array(self.u)
        strain = np.array(self.strain_z)
        return settlement, horizontal_disp, pore_pressure, strain

    def reset(self):
        """シミュレーションをリセット"""
        dtype = self._get_dtype()
        self.u = mx.zeros((self.nz, self.nx), dtype=dtype)
        self.disp_x = mx.zeros((self.nz, self.nx), dtype=dtype)
        self.disp_z = mx.zeros((self.nz, self.nx), dtype=dtype)
        self.strain_z = mx.zeros((self.nz, self.nx), dtype=dtype)
        self._apply_initial_excess_pore_pressure()

    def plot_results(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (18, 5),
        dpi: int = 300,
    ):
        """結果を可視化"""
        settlement, horizontal_disp, pore_pressure, strain = self.get_current_state()

        x = np.linspace(0, self.length, self.nx)
        z = np.linspace(0, self.depth, self.nz)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

        # 沈下量分布
        contour1 = ax1.contourf(x, z, settlement, levels=50, cmap="viridis")
        plt.colorbar(contour1, ax=ax1, label="Vertical Displacement (m)")
        ax1.set_xlabel("Distance (m)")
        ax1.set_ylabel("Depth (m)")
        ax1.set_title("Vertical Displacement (Terzaghi Method)")
        ax1.invert_yaxis()

        # 荷重位置をマーク
        x_center = self.length / 2
        load_x_start = x_center - self.load_width / 2
        load_x_end = x_center + self.load_width / 2
        ax1.axvline(load_x_start, color="r", linestyle="--", alpha=0.5)
        ax1.axvline(load_x_end, color="r", linestyle="--", alpha=0.5)

        # 水平変位分布 (Terzaghiでは常にゼロ)
        contour2 = ax2.contourf(x, z, horizontal_disp, levels=50, cmap="RdBu_r")
        plt.colorbar(contour2, ax=ax2, label="Horizontal Displacement (m)")
        ax2.set_xlabel("Distance (m)")
        ax2.set_ylabel("Depth (m)")
        ax2.set_title("Horizontal Displacement (εx = 0)")
        ax2.invert_yaxis()

        # 過剰間隙水圧分布
        contour3 = ax3.contourf(x, z, pore_pressure, levels=50, cmap="coolwarm")
        plt.colorbar(contour3, ax=ax3, label="Excess Pore Pressure (kPa)")
        ax3.set_xlabel("Distance (m)")
        ax3.set_ylabel("Depth (m)")
        ax3.set_title("Excess Pore Pressure Distribution")
        ax3.invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Figure saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def get_consolidation_degree(self) -> float:
        """現在の圧密度を計算"""
        return (1 - float(mx.max(self.u)) / self.load_intensity) * 100


@contextmanager
def nullcontext():
    """何もしないコンテキストマネージャ(Python 3.7互換用)"""
    yield


def main():
    """メイン実行関数 - TerzaghiとMikasaの比較"""
    # 共通パラメータ
    length = 20.0  # 20m幅
    depth = 10.0  # 10m深さ
    nx = 51
    nz = 26
    cv = 1.0  # 圧密係数 1.0 m²/day
    E = 5000.0  # ヤング率 5000 kPa
    dt = 0.01  # 時間刻み: 0.01 day
    total_time = 3650.0  # 10年間
    device = "auto"
    load_intensity = 100.0  # 100 kPa
    load_width = 4.0  # 4m幅の荷重
    poisson_ratio = 0.3

    print("=" * 70)
    print("Terzaghi 2D Consolidation Simulation (10 years)")
    print("Theory: Oedometer condition (εx = 0, no lateral displacement)")
    print("Implicit Method (Crank-Nicolson) with GPU Optimization")
    print("=" * 70)

    # Terzaghiシミュレーター初期化
    terzaghi_sim = TerzaghiConsolidationSimulator(
        length=length,
        depth=depth,
        nx=nx,
        nz=nz,
        cv=cv,
        E=E,
        dt=dt,
        total_time=total_time,
        device=device,
        load_intensity=load_intensity,
        load_width=load_width,
        poisson_ratio=poisson_ratio,
    )

    # Terzaghiシミュレーション実行
    terzaghi_sim.run_simulation(verbose=True)
    print("=" * 70)

    # Terzaghi結果の可視化
    terzaghi_sim.plot_results(save_path="terzaghi_consolidation_result.png")

    # Mikasaシミュレーターとの比較
    print("\n" + "=" * 70)
    print("Comparison with Mikasa Method")
    print("=" * 70)

    try:
        from Mikasa import MikasaConsolidationSimulator

        # Mikasaシミュレーター初期化
        mikasa_sim = MikasaConsolidationSimulator(
            length=length,
            depth=depth,
            nx=nx,
            nz=nz,
            cv=cv,
            E=E,
            dt=dt,
            total_time=total_time,
            device=device,
            load_intensity=load_intensity,
            load_width=load_width,
            poisson_ratio=poisson_ratio,
        )

        # Mikasaシミュレーション実行
        print("\nRunning Mikasa simulation for comparison...")
        mikasa_sim.run_simulation(verbose=True)

        # 比較プロット
        plot_comparison(terzaghi_sim, mikasa_sim, length, depth, nx, nz)

    except ImportError:
        print("Mikasa.py not found. Skipping comparison.")
        print("Run Terzaghi simulation only.")


def plot_comparison(terzaghi_sim, mikasa_sim, length, depth, nx, nz):
    """TerzaghiとMikasaの結果を比較"""
    # 状態を取得
    terz_settlement, terz_horiz, terz_pore, terz_strain = terzaghi_sim.get_current_state()
    mika_settlement, mika_horiz, mika_pore, mika_strain = mikasa_sim.get_current_state()

    x = np.linspace(0, length, nx)
    z = np.linspace(0, depth, nz)

    # 中心線のインデックス
    center_idx = nx // 2

    # 比較プロット
    plt.figure(figsize=(20, 12))

    # 1. 中心線での沈下量プロファイル比較
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(terz_settlement[:, center_idx] * 1000, z, "b-", linewidth=2, label="Terzaghi")
    ax1.plot(mika_settlement[:, center_idx] * 1000, z, "r--", linewidth=2, label="Mikasa")
    ax1.set_xlabel("Settlement (mm)")
    ax1.set_ylabel("Depth (m)")
    ax1.set_title("Center Line Settlement Profile")
    ax1.legend()
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)

    # 2. 表面沈下量分布比較
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(x, terz_settlement[0, :] * 1000, "b-", linewidth=2, label="Terzaghi")
    ax2.plot(x, mika_settlement[0, :] * 1000, "r--", linewidth=2, label="Mikasa")
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Surface Settlement (mm)")
    ax2.set_title("Surface Settlement Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 水平変位比較（中心線）
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(terz_horiz[:, center_idx] * 1000, z, "b-", linewidth=2, label="Terzaghi (εx=0)")
    ax3.plot(mika_horiz[:, center_idx] * 1000, z, "r--", linewidth=2, label="Mikasa")
    ax3.set_xlabel("Horizontal Displacement (mm)")
    ax3.set_ylabel("Depth (m)")
    ax3.set_title("Center Line Horizontal Displacement")
    ax3.legend()
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3)

    # 4. Terzaghi沈下量コンター
    ax4 = plt.subplot(3, 3, 4)
    contour4 = ax4.contourf(x, z, terz_settlement, levels=50, cmap="viridis")
    plt.colorbar(contour4, ax=ax4, label="Settlement (m)")
    ax4.set_xlabel("Distance (m)")
    ax4.set_ylabel("Depth (m)")
    ax4.set_title("Terzaghi Settlement Contour")
    ax4.invert_yaxis()

    # 5. Mikasa沈下量コンター
    ax5 = plt.subplot(3, 3, 5)
    contour5 = ax5.contourf(x, z, mika_settlement, levels=50, cmap="viridis")
    plt.colorbar(contour5, ax=ax5, label="Settlement (m)")
    ax5.set_xlabel("Distance (m)")
    ax5.set_ylabel("Depth (m)")
    ax5.set_title("Mikasa Settlement Contour")
    ax5.invert_yaxis()

    # 6. 沈下量差分コンター
    ax6 = plt.subplot(3, 3, 6)
    settlement_diff = (mika_settlement - terz_settlement) * 1000  # mm
    contour6 = ax6.contourf(x, z, settlement_diff, levels=50, cmap="RdBu_r")
    plt.colorbar(contour6, ax=ax6, label="Difference (mm)")
    ax6.set_xlabel("Distance (m)")
    ax6.set_ylabel("Depth (m)")
    ax6.set_title("Settlement Difference (Mikasa - Terzaghi)")
    ax6.invert_yaxis()

    # 7. Terzaghi水平変位コンター
    ax7 = plt.subplot(3, 3, 7)
    contour7 = ax7.contourf(x, z, terz_horiz * 1000, levels=50, cmap="RdBu_r")
    plt.colorbar(contour7, ax=ax7, label="Horiz. Disp. (mm)")
    ax7.set_xlabel("Distance (m)")
    ax7.set_ylabel("Depth (m)")
    ax7.set_title("Terzaghi Horizontal Displacement (εx=0)")
    ax7.invert_yaxis()

    # 8. Mikasa水平変位コンター
    ax8 = plt.subplot(3, 3, 8)
    contour8 = ax8.contourf(x, z, mika_horiz * 1000, levels=50, cmap="RdBu_r")
    plt.colorbar(contour8, ax=ax8, label="Horiz. Disp. (mm)")
    ax8.set_xlabel("Distance (m)")
    ax8.set_ylabel("Depth (m)")
    ax8.set_title("Mikasa Horizontal Displacement")
    ax8.invert_yaxis()

    # 9. 統計比較
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis("off")

    stats_text = f"""
    Comparison Statistics:
    
    Maximum Settlement:
      Terzaghi: {terz_settlement.max() * 1000:.2f} mm
      Mikasa:   {mika_settlement.max() * 1000:.2f} mm
      Diff:     {(mika_settlement.max() - terz_settlement.max()) * 1000:.2f} mm
    
    Surface Center Settlement:
      Terzaghi: {terz_settlement[0, center_idx] * 1000:.2f} mm
      Mikasa:   {mika_settlement[0, center_idx] * 1000:.2f} mm
      Diff:     {(mika_settlement[0, center_idx] - terz_settlement[0, center_idx]) * 1000:.2f} mm
    
    Max Horizontal Displacement:
      Terzaghi: {np.abs(terz_horiz).max() * 1000:.2f} mm (≈0, εx=0)
      Mikasa:   {np.abs(mika_horiz).max() * 1000:.2f} mm
    
    Consolidation Degree:
            Terzaghi: {terzaghi_sim.get_consolidation_degree():.1f}%
            Mikasa:   {(1 - float(mx.max(mikasa_sim.u)) / mikasa_sim.load_intensity) * 100:.1f}%
    
    Theory Difference:
      Terzaghi: Oedometer (εx = 0)
      Mikasa:   Lateral displacement allowed
    """

    ax9.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment="center", family="monospace")

    plt.tight_layout()
    plt.savefig("terzaghi_mikasa_comparison.png", dpi=300, bbox_inches="tight")
    print("\nComparison figure saved to: terzaghi_mikasa_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
