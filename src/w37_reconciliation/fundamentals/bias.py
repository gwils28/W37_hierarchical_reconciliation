"""Où ça casse n°1 : SGS = S PRÉSERVE l'absence de biais, il ne la CRÉE pas.

Une seule prévision de base biaisée (ici le total) est redistribuée par MinT sur toutes les feuilles ;
le bottom-up, qui ignore le total, n'en voit rien.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..mint.projection import mint_projection


def toy_hierarchy() -> np.ndarray:
    """1 total + 3 feuilles."""
    return np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)


def bottom_up_projection(S: np.ndarray) -> np.ndarray:
    nb = S.shape[1]
    return S @ np.hstack([np.zeros((nb, S.shape[0] - nb)), np.eye(nb)])


def bias_counterexample(w_diag=(1.0, 9.0, 9.0, 9.0), b=(10.0, 20.0, 30.0),
                        bias_total: float = 15.0) -> pd.DataFrame:
    """Le contre-exemple du plan : vérité exacte partout sauf un biais sur le total.

    Retourne, par série : vérité, base, bottom-up, MinT et leurs erreurs.
    """
    S = toy_hierarchy()
    y = S @ np.asarray(b)
    y_hat = y + np.r_[bias_total, 0.0, 0.0, 0.0]
    _, P = mint_projection(S, np.diag(w_diag))
    out = pd.DataFrame({"vérité": y, "base": y_hat, "bottom-up": bottom_up_projection(S) @ y_hat,
                        "MinT": P @ y_hat}, index=["Total", "Feuille_1", "Feuille_2", "Feuille_3"])
    out["erreur bottom-up"] = out["bottom-up"] - out["vérité"]
    out["erreur MinT"] = out["MinT"] - out["vérité"]
    return out


def bias_allocation(S: np.ndarray, W: np.ndarray, series: int = 0) -> np.ndarray:
    """Comment un biais de +1 sur la prévision de base `series` se retrouve dans chaque série réconciliée.

    C'est la colonne `series` de P : la réconciliation est linéaire, donc biais_réconcilié = P @ biais_base.
    """
    _, P = mint_projection(S, W)
    return P[:, series]


def bias_sweep(biases, sd_total: float = 1.0, sd_leaf: float = 3.0, n: int = 20_000,
               seed: int = 0) -> pd.DataFrame:
    """Monte-Carlo : prévisions de base bruitées (total précis, feuilles bruitées) + biais sur le total.

    MinT utilise la VRAIE covariance des erreurs W = diag(sd_total², sd_leaf², …) :
    seul le biais pose problème.
    Retourne l'erreur absolue moyenne (MAE) par niveau pour base, bottom-up et MinT.
    Le MASE n'est qu'un MAE divisé par une échelle propre à chaque série : les comparaisons entre
    méthodes d'un même niveau sont identiques en MAE et en MASE.
    """
    S = toy_hierarchy()
    rng = np.random.default_rng(seed)
    sd = np.array([sd_total, sd_leaf, sd_leaf, sd_leaf])
    noise = rng.normal(size=(n, 4)) * sd                     # erreurs de base, sans biais
    P_mint = mint_projection(S, np.diag(sd**2))[1]
    P_bu = bottom_up_projection(S)
    rows = []
    for beta in biases:
        err_base = noise + np.r_[beta, 0, 0, 0]               # y_hat − y : la vérité se simplifie
        for name, P in [("base", np.eye(4)), ("bottom-up", P_bu), ("MinT", P_mint)]:
            e = np.abs(err_base @ P.T)
            rows.append({"biais": beta, "méthode": name,
                         "MAE total": e[:, 0].mean(), "MAE feuilles": e[:, 1:].mean()})
    return pd.DataFrame(rows)
