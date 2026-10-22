"""Run the finite-strain example and save its contour figure to outputs/."""

from pathlib import Path

from mikasa_consolidation.finite_strain import run_example


def main():
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    run_example(output / "mikasa_consolidation_result.png")


if __name__ == "__main__":
    main()
