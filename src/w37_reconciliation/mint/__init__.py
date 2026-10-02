"""MinT(shrink) écrit de zéro — bloc 2."""
from .checks import incoherence, relative_gap, unbiasedness_error
from .projection import mint_projection, reconcile
from .shrinkage import ShrunkCovariance, schafer_strimmer_lambda, shrunk_covariance

__all__ = [
    "ShrunkCovariance",
    "incoherence",
    "mint_projection",
    "reconcile",
    "relative_gap",
    "schafer_strimmer_lambda",
    "shrunk_covariance",
    "unbiasedness_error",
]
