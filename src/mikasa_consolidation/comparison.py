"""寺田（Terzaghi）型の線形小ひずみ圧密と、三笠有限ひずみ圧密との比較。

``ConsolidationSimulator`` は ``linear.py`` の一定 ``c_v`` 線形拡散、
``FiniteStrainConsolidation`` は ``finite_strain.py`` の有限ひずみ実装。

比較図は、両理論が一致すべき条件（小ひずみ、自重の影響が小さい）で描く。同じ
``c_v`` と、圧縮則から換算した同じ一次圧縮係数を与える。
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from .linear import ConsolidationSimulator, show_if_interactive
from .finite_strain import FiniteStrainConsolidation

#: 比較条件。帯載荷を十分広く取り、荷重は初期有効応力の 5% に抑える。
GEOMETRY = dict(length=1.0, depth=1.0, nx=2, nz=41, load_intensity=50.0, load_width=1.0e9)
FINITE_STRAIN = dict(GEOMETRY, e0=1.0, Cc=0.05, Ck=0.05, k0=4.27e-4,
                     sigma_ref=1000.0, sigma_top=1000.0)
POISSON_RATIO = 0.3
#: 時間係数の表示点 ``T = c_v t / H^2``。
TIME_FACTORS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)


def matched_cv():
    """圧縮則と初期有効応力から ``c_v = (1+e0) k σ' ln10 /(γ_w Cc)`` を求める。"""
    return ((1.0 + FINITE_STRAIN["e0"]) * FINITE_STRAIN["k0"] * FINITE_STRAIN["sigma_ref"]
            * math.log(10.0) / (9.81 * FINITE_STRAIN["Cc"]))


def matched_modulus():
    """同じ体積圧縮係数 ``m_v = a_v/(1+e0)`` を与える弾性係数 ``E``。"""
    constrained = (1.0 + FINITE_STRAIN["e0"]) * FINITE_STRAIN["sigma_ref"] * math.log(10.0) \
        / FINITE_STRAIN["Cc"]
    return constrained * (1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO) / (1 - POISSON_RATIO)


def terzaghi_degree(time_factor):
    """片面排水の解析解 ``U(T)``[%]。"""
    total = 0.0
    for index in range(200):
        mode = (2 * index + 1) * math.pi / 2.0
        total += 2.0 / mode**2 * math.exp(-mode**2 * time_factor)
    return 100.0 * (1.0 - total)


def run_pair(time_factor, device="cpu"):
    """同じ ``c_v`` で両モデルを ``T = time_factor`` まで走らせ ``(寺田, 三笠)`` を返す。"""
    duration = time_factor * GEOMETRY["depth"]**2 / matched_cv()
    mikasa = FiniteStrainConsolidation(total_time=duration, dt=duration / 200.0,
                                     device=device, **FINITE_STRAIN)
    mikasa.run_simulation(verbose=False)
    terzaghi = ConsolidationSimulator(cv=matched_cv(), E=matched_modulus(),
                                      dt=duration / 200.0, total_time=duration, device=device, **GEOMETRY)
    terzaghi.run_simulation(verbose=False)
    return terzaghi, mikasa


def plot_comparison(save_path="terzaghi_mikasa_comparison.png", show=False, figsize=(15, 5), dpi=300,
                    device="cpu"):
    """寺田（線形・小ひずみ）と三笠（有限ひずみ）の比較図を保存する。"""
    final_factor = TIME_FACTORS[-1] / 2.0
    terzaghi, mikasa = run_pair(final_factor, device=device)
    terzaghi_state = terzaghi.get_current_state()
    mikasa_state = mikasa.get_current_state()
    center = GEOMETRY["nx"] // 2
    linear_z = np.linspace(0.0, GEOMETRY["depth"], GEOMETRY["nz"])
    material_z = mikasa.d_zeta * (np.cumsum(1.0 + mikasa.e, axis=0) - 0.5 * (1.0 + mikasa.e))

    degrees = []
    for factor in TIME_FACTORS:
        pair = run_pair(factor, device=device)
        degrees.append((factor, pair[0].get_consolidation_degree(),
                        pair[1].get_consolidation_degree()))

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].plot(terzaghi_state[0][:, center] * 1000, linear_z, "-",
                 label="Terzaghi (linear, matched E)")
    axes[0].plot(mikasa_state[0][:, center] * 1000, material_z[:, center], "--",
                 label="Mikasa (finite strain)")
    axes[1].plot(terzaghi_state[2][:, center], linear_z, "-", label="Terzaghi (linear)")
    axes[1].plot(mikasa_state[2][:, center], material_z[:, center], "--", label="Mikasa (finite strain)")
    factors = [item[0] for item in degrees]
    axes[2].plot(factors, [item[1] for item in degrees], "o-", label="Terzaghi (linear)")
    axes[2].plot(factors, [item[2] for item in degrees], "s--", label="Mikasa (finite strain)")
    axes[2].plot(factors, [terzaghi_degree(factor) for factor in factors], ":",
                 label="Terzaghi analytic series")
    axes[0].set(xlabel="Center settlement (mm)", ylabel="Depth (m)",
                title=f"Settlement at T={final_factor:g}")
    axes[1].set(xlabel="Center excess pore pressure (kPa)", ylabel="Depth (m)",
                title="Excess pore pressure")
    axes[2].set(xlabel="Time factor T = c_v t / H^2", ylabel="Consolidation degree (%)",
                title="Dissipation rate")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    fig.suptitle("Small-strain limit: the two theories agree (finite strain is O(load/σ'))")
    fig.tight_layout()
    try:
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show:
            show_if_interactive()
    finally:
        plt.close(fig)
