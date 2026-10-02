"""Expérience complète du bloc 2 : MinT maison vs `MinTrace(method="mint_shrink")`."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace

from ..data import Hierarchy, build_hierarchy, make_region_channel_data, to_wide, train_test_split
from ..forecasting import fit_base_forecasts, residual_matrix
from .checks import incoherence, relative_gap, unbiasedness_error
from .projection import mint_projection, reconcile
from .shrinkage import ShrunkCovariance, schafer_strimmer_lambda, shrunk_covariance

LIB_COLUMN = "AutoETS/MinTrace_method-mint_shrink"


@dataclass
class MintExperiment:
    hierarchy: Hierarchy
    train: pd.DataFrame
    test: pd.DataFrame
    Y_hat: pd.DataFrame
    Y_fit: pd.DataFrame
    Y_rec: pd.DataFrame
    E: np.ndarray
    cov: ShrunkCovariance
    G: np.ndarray
    P: np.ndarray
    y_hat: np.ndarray
    mine: np.ndarray
    lib: np.ndarray
    metrics: dict = field(default_factory=dict)


def run_mint_experiment(n: int = 80, h: int = 8, seed: int = 0) -> MintExperiment:
    hier = build_hierarchy(make_region_channel_data(n=n, seed=seed))
    train, test = train_test_split(hier.Y_df, h)
    Y_hat, Y_fit = fit_base_forecasts(train, h)
    # Fuite n°1 : W ne doit voir QUE des résidus du train.
    if Y_fit["ds"].max() >= test["ds"].min():
        raise RuntimeError("fuite : les valeurs ajustées débordent sur la période de test")

    hrec = HierarchicalReconciliation(
        reconcilers=[BottomUp(), MinTrace(method="ols"), MinTrace(method="mint_shrink")])
    Y_rec = hrec.reconcile(Y_hat_df=Y_hat, Y_df=Y_fit, S_df=hier.S_df, tags=hier.tags)

    S, order = hier.S, hier.order
    E = residual_matrix(Y_fit, order)
    cov = shrunk_covariance(E, base="plan")
    G, P = mint_projection(S, cov.W)
    y_hat = to_wide(Y_hat, order, "AutoETS").to_numpy()
    mine = reconcile(y_hat, P)
    lib = to_wide(Y_rec, order, LIB_COLUMN).to_numpy()

    metrics = {
        "T": E.shape[0], "m": E.shape[1], "nb": S.shape[1], "lambda": cov.lam,
        "idempotence": float(np.abs(P @ P - P).max()),
        "unbiasedness": unbiasedness_error(P, S),
        "incoherence": incoherence(mine, S),
        "relative_gap": relative_gap(mine, lib),
        "absolute_gap": float(np.abs(mine - lib).max()),
        "mean_level": float(np.abs(lib).mean()),
    }
    return MintExperiment(hier, train, test, Y_hat, Y_fit, Y_rec, E, cov, G, P, y_hat, mine, lib, metrics)


def gap_decomposition(exp: MintExperiment) -> pd.DataFrame:
    """D'où vient l'écart à la bibliothèque ? Covariance de base vs estimateur de λ."""
    S, E = exp.hierarchy.S, exp.E

    def gap(W1: np.ndarray, lam: float) -> float:
        W = lam * np.diag(np.diag(W1)) + (1 - lam) * W1
        return float(np.abs(reconcile(exp.y_hat, mint_projection(S, W)[1]) - exp.lib).max())

    lam_raw, lam_centered = schafer_strimmer_lambda(E), schafer_strimmer_lambda(E, center=True)
    rows = [
        ("E'E/T (plan)", "non centré", lam_raw, gap(E.T @ E / len(E), lam_raw)),
        ("np.cov, ddof=1 (lib)", "non centré", lam_raw, gap(np.cov(E.T), lam_raw)),
        ("np.cov, ddof=1 (lib)", "centré", lam_centered, gap(np.cov(E.T), lam_centered)),
    ]
    return pd.DataFrame(rows, columns=["covariance de base", "λ estimé sur résidus", "λ", "écart absolu"])


def m_greater_than_T_demo(E: np.ndarray, T_short: int | None = None) -> dict:
    """Piège n°3 : avec T < m, W1 est singulière ; le shrinkage la rend inversible."""
    m = E.shape[1]
    E_short = E[: (T_short or m - 2)]
    cov = shrunk_covariance(E_short, base="plan")
    return {"T": len(E_short), "m": m, "rank_W1": int(np.linalg.matrix_rank(cov.W1)),
            "cond_W1": float(np.linalg.cond(cov.W1)), "lambda": cov.lam,
            "cond_W": cov.condition_number}
