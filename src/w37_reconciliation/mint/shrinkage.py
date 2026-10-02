"""Covariance des résidus avec shrinkage Schäfer–Strimmer (2005), cible « D » (diagonale).

On rétrécit les corrélations hors diagonale vers 0, sans toucher aux variances :

    W = λ · diag(W1) + (1 − λ) · W1,      λ* = Σ_{i≠j} Var(r_ij) / Σ_{i≠j} r_ij²

avec, sur les résidus standardisés x_ki et w_kij = x_ki · x_kj :

    Var(r_ij) = T / (T − 1)³ · Σ_k (w_kij − mean_k w_kij)²
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def schafer_strimmer_lambda(E: np.ndarray, center: bool = False) -> float:
    """Intensité de shrinkage optimale, bornée à [0, 1].

    E      : résidus (T × m).
    center : recentrer les résidus avant de standardiser. Le squelette du plan suppose des
             résidus one-step centrés (center=False) ; HierarchicalForecast aussi pour λ.

    Seules des sommes hors diagonale interviennent : on les calcule sans jamais former le tenseur
    (T, m, m) des produits croisés ni la matrice r (m × m). Mémoire O(T² + T·m) au lieu de O(T·m²)
    (31 Go pour m = 5 000, T = 156) :

        Σ_{i≠j} r_ij²             = ‖X X'‖²_F / T² − Σ_i r_ii²         (Gram T × T)
        Σ_{i≠j} Σ_k x_ki² x_kj²   = Σ_k [(Σ_i x_ki²)² − Σ_i x_ki⁴]
        Σ_{i≠j} Var(r_ij)         = T / (T − 1)³ · (ligne précédente − T · Σ_{i≠j} r_ij²)
    """
    T, m = E.shape
    if center:
        E = E - E.mean(axis=0)
    X = E / np.sqrt((E**2).mean(axis=0))          # résidus standardisés
    X2 = X**2
    r_diag = X2.mean(axis=0)                       # r_ii (= 1 sans centrage)
    gram = X @ X.T                                 # (T, T)
    sum_r2_off = (gram**2).sum() / T**2 - (r_diag**2).sum()
    sum_w2_off = ((X2.sum(axis=1)) ** 2 - (X2**2).sum(axis=1)).sum()
    sum_var_off = T / (T - 1) ** 3 * (sum_w2_off - T * sum_r2_off)
    if sum_r2_off <= 0:
        return 1.0
    return float(np.clip(sum_var_off / sum_r2_off, 0.0, 1.0))


@dataclass(frozen=True)
class ShrunkCovariance:
    W: np.ndarray      # covariance rétrécie (m × m), celle qui entre dans MinT
    W1: np.ndarray     # covariance empirique de base
    lam: float         # intensité de shrinkage

    @property
    def condition_number(self) -> float:
        return float(np.linalg.cond(self.W))


def shrunk_covariance(E: np.ndarray, base: str = "plan") -> ShrunkCovariance:
    """Covariance rétrécie des résidus.

    base="plan"    : W1 = E'E / T, non centrée (squelette du plan W37).
    base="library" : W1 = np.cov(E.T), centrée, ddof = 1 (ce que fait HierarchicalForecast 1.5.x).
    Dans les deux cas λ est estimé sur résidus non centrés.
    """
    if base == "plan":
        W1 = E.T @ E / E.shape[0]
    elif base == "library":
        W1 = np.cov(E.T)
    else:
        raise ValueError(f"base inconnue : {base!r} (attendu 'plan' ou 'library')")
    lam = schafer_strimmer_lambda(E)
    return ShrunkCovariance(W=lam * np.diag(np.diag(W1)) + (1 - lam) * W1, W1=W1, lam=lam)
