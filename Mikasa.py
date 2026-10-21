"""旧インポート名を維持する入口。三笠有限ひずみ理論の実装ではない。"""

from consolidation import ConsolidationSimulator as MikasaConsolidationSimulator
from consolidation import run_example


def main():
    run_example("mikasa_consolidation_result.png")


if __name__ == "__main__":
    main()
