"""Choix du réconciliateur : mint_shrink, ou repli si W est inutilisable."""
from __future__ import annotations

import numpy as np

from ..mint.shrinkage import shrunk_covariance


def choose_method(residuals: np.ndarray, cfg: dict) -> tuple[str, dict]:
    """residuals : (T × m), one-step in-sample, fenêtre strictement antérieure à l'origine.

    Le repli se déclenche sur des critères MESURÉS (historique trop court, résidu dégénéré,
    conditionnement de W après shrinkage), pas sur la forme m > T : le shrinkage existe pour ce cas.
    """
    rc = cfg["reconciler"]
    rules = rc.get("fallback_when", {})
    T, m = residuals.shape
    diag: dict = {"T": T, "m": m}

    if T < rules.get("min_residuals", 0):
        return rc["fallback"], {**diag, "reason": f"T={T} < min_residuals"}
    if rules.get("residuals_fewer_than_series", False) and m > T:
        return rc["fallback"], {**diag, "reason": "m > T"}
    if np.any(residuals.std(axis=0) == 0):
        return rc["fallback"], {**diag, "reason": "résidu de variance nulle"}

    cov = shrunk_covariance(residuals, base="library")   # même W que celle qu'utilisera la bibliothèque
    diag.update(lam=cov.lam, cond=cov.condition_number)
    if cov.condition_number > rules.get("max_condition_number", np.inf):
        return rc["fallback"], {**diag, "reason": f"cond(W)={cov.condition_number:.1e}"}
    return rc["method"], {**diag, "reason": "ok"}
