"""旧入口との互換用。2つの旧クラス名は同じ拘束圧密の簡易モデルを指す。"""

import matplotlib.pyplot as plt
import numpy as np

from consolidation import ConsolidationSimulator as TerzaghiConsolidationSimulator
from consolidation import is_negligible_field, run_example


def plot_comparison(terzaghi_sim, mikasa_sim):
    """旧比較API。異なる理論の比較ではなく、渡された2状態を描画する。"""
    simulators = ((terzaghi_sim, "Terzaghi (legacy name)", "-"),
                  (mikasa_sim, "Mikasa (legacy name)", "--"))
    load_scale = max(float(simulator.total_stress.max()) for simulator, _, _ in simulators)
    pressure_visible = any(not is_negligible_field(simulator.u, load_scale)
                           for simulator, _, _ in simulators)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for simulator, label, style in simulators:
        settlement, _, pressure, _ = simulator.get_current_state()
        z = np.linspace(0, simulator.depth, simulator.nz)
        center = simulator.nx // 2
        axes[0].plot(settlement[:, center] * 1000, z, style, label=label)
        if pressure_visible:
            axes[1].plot(pressure[:, center], z, style, label=label)
    if pressure_visible:
        axes[1].legend()
    else:
        # 消散済みの水圧を丸め誤差の軸で描かない。
        axes[1].text(0.5, 0.5, "excess pore pressure ≈ 0", transform=axes[1].transAxes,
                     ha="center", va="center")
        axes[1].set_xticks([])
    axes[0].set(xlabel="Center settlement (mm)", ylabel="Depth (m)")
    axes[1].set(xlabel="Center pore pressure (kPa)", ylabel="Depth (m)")
    for ax in axes:
        ax.invert_yaxis()
    axes[0].legend()
    if not pressure_visible:
        axes[1].set_ylim(axes[0].get_ylim())
    fig.suptitle("Legacy aliases: same simplified model, not two consolidation theories")
    fig.tight_layout()
    try:
        fig.savefig("terzaghi_mikasa_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()
    finally:
        plt.close(fig)


def main():
    simulator = run_example("terzaghi_consolidation_result.png")
    plot_comparison(simulator, simulator)


if __name__ == "__main__":
    main()
