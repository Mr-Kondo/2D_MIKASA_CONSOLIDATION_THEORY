"""旧入口との互換用。2つの旧クラス名は同じ拘束圧密の簡易モデルを指す。"""

import matplotlib.pyplot as plt
import numpy as np

from consolidation import ConsolidationSimulator as TerzaghiConsolidationSimulator
from consolidation import run_example


def plot_comparison(terzaghi_sim, mikasa_sim):
    """旧比較API。異なる理論の比較ではなく、渡された2状態を描画する。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for simulator, label, style in ((terzaghi_sim, "Terzaghi (legacy name)", "-"),
                                    (mikasa_sim, "Mikasa (legacy name)", "--")):
        settlement, _, pressure, _ = simulator.get_current_state()
        z = np.linspace(0, simulator.depth, simulator.nz)
        center = simulator.nx // 2
        axes[0].plot(settlement[:, center] * 1000, z, style, label=label)
        axes[1].plot(pressure[:, center], z, style, label=label)
    for ax, title in zip(axes, ("Center settlement (mm)", "Center pore pressure (kPa)")):
        ax.set(xlabel=title, ylabel="Depth (m)")
        ax.invert_yaxis()
        ax.legend()
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
