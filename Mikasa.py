"""三笠／Gibson 系の有限ひずみ圧密（``finite_strain.py``）の入口。

旧版は一定係数の線形拡散（``consolidation.py``）を三笠理論として提示していた。
本モジュールは物質座標 ``ζ = ∫ dz/(1+e)``、圧縮則、透水則、自重を扱う有限ひずみ
実装を公開する。一定係数 c_v の線形モデルは ``Terzaghi.py`` 側にある。
"""

from finite_strain import FiniteStrainConsolidation

# 旧インポート名。有限ひずみ圧密モデルを指す。
MikasaConsolidationSimulator = FiniteStrainConsolidation


def run_example(save_path):
    """既定条件で有限ひずみ圧密を実行し、結果図を保存する。"""
    simulator = FiniteStrainConsolidation()
    simulator.run_simulation()
    simulator.plot_results(save_path=save_path)
    return simulator


def main():
    run_example("mikasa_consolidation_result.png")


if __name__ == "__main__":
    main()
