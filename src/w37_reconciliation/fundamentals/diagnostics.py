"""Outils pour la question d'entretien :
« MinT dégrade le SKU et améliore le national, que se passe-t-il ? »
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..mint.projection import mint_projection


def residual_bias_table(E: np.ndarray, names: list[str], t_threshold: float = 2.0) -> pd.DataFrame:
    """Hypothèse 1 : une prévision de base biaisée. Test de moyenne nulle des résidus, série par série.

    t = moyenne / (écart-type / √T). |t| > 2 : biais probable, à corriger AVANT de réconcilier.
    """
    T = E.shape[0]
    mean, sd = E.mean(axis=0), E.std(axis=0, ddof=1)
    t = mean / (sd / np.sqrt(T))
    return pd.DataFrame({"moyenne": mean, "écart-type": sd, "t": t, "biais suspect": np.abs(t) > t_threshold},
                        index=names)


def random_unbiased_G(S: np.ndarray, G_star: np.ndarray, rng: np.random.Generator,
                      scale: float = 0.3) -> np.ndarray:
    """Une matrice G quelconque vérifiant SGS = S (⟺ GS = I si S est de rang plein) : G* + K, avec KS = 0."""
    m = S.shape[0]
    M = np.eye(m) - S @ np.linalg.pinv(S)                  # projecteur sur l'orthogonal de col(S)
    return G_star + scale * rng.normal(size=G_star.shape) @ M


def weighted_trace_check(S: np.ndarray, W: np.ndarray, level_weights: np.ndarray, n_random: int = 2_000,
                         seed: int = 0) -> dict:
    """Vérifie que MinT minimise trace(Λ · Var(erreur réconciliée)) pour une pondération Λ des séries.

    Gauss-Markov : S G* W G*' S' ⪯ S G W G' S' (ordre de Loewner) pour TOUT G non biaisé.
    Donc aucune pondération par niveau ne change l'optimum… tant que W est la vraie covariance.
    """
    rng = np.random.default_rng(seed)
    G_star, _ = mint_projection(S, W)
    Lam = np.diag(level_weights)

    def score(G):
        return float(np.trace(Lam @ S @ G @ W @ G.T @ S.T))

    best = score(G_star)
    others = np.array([score(random_unbiased_G(S, G_star, rng)) for _ in range(n_random)])
    return {"score MinT": best, "meilleur score aléatoire": float(others.min()),
            "part des G aléatoires battant MinT": float((others < best - 1e-12).mean())}
