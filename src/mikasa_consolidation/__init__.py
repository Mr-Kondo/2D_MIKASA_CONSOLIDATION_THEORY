"""Finite-strain consolidation and its linear small-strain reference."""

from .finite_strain import FiniteStrainConsolidation
from .linear import ConsolidationSimulator

MikasaConsolidationSimulator = FiniteStrainConsolidation
TerzaghiConsolidationSimulator = ConsolidationSimulator

__all__ = ["FiniteStrainConsolidation", "ConsolidationSimulator",
           "MikasaConsolidationSimulator", "TerzaghiConsolidationSimulator"]
