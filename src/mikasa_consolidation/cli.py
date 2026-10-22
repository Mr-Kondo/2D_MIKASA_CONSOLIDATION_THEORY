"""Shared entry point for local scripts and the Colab notebook."""

import argparse
from pathlib import Path

from . import ConsolidationSimulator, FiniteStrainConsolidation
from .comparison import plot_comparison


def run_examples(model="all", device="cpu", output_dir="outputs"):
    if model not in ("mikasa", "terzaghi", "all"):
        raise ValueError("model must be 'mikasa', 'terzaghi' or 'all'")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    simulations = {}
    if model in ("mikasa", "all"):
        simulations["mikasa"] = FiniteStrainConsolidation(device=device)
    if model in ("terzaghi", "all"):
        simulations["terzaghi"] = ConsolidationSimulator(device=device)
    for name, simulator in simulations.items():
        simulator.run_simulation()
        simulator.plot_results(save_path=output / f"{name}_consolidation_result.png")
    if "terzaghi" in simulations:
        plot_comparison(save_path=output / "terzaghi_mikasa_comparison.png", device=device)
    return simulations


def main(model="all"):
    parser = argparse.ArgumentParser(description="Run consolidation examples and save contour figures")
    parser.add_argument("--model", choices=("mikasa", "terzaghi", "all"), default=model)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run_examples(args.model, args.device, args.output_dir)
