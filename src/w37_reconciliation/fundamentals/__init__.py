"""Bloc 4 — fondamentaux : ce que MinT garantit (absence de biais préservée) et ce qu'il ne garantit pas
(il propage le biais, et il ne réconcilie pas des quantiles)."""
from .bias import bias_allocation, bias_counterexample, bias_sweep, toy_hierarchy
from .probabilistic import (
    GaussianHierarchy,
    bootstrap_coverage,
    coverage_experiment,
    rho_sweep,
)

__all__ = [
    "GaussianHierarchy",
    "bias_allocation",
    "bias_counterexample",
    "bias_sweep",
    "bootstrap_coverage",
    "coverage_experiment",
    "rho_sweep",
    "toy_hierarchy",
]
