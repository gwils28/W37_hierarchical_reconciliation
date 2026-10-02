"""Où ça casse n°2 : MinT réconcilie des MOYENNES, pas des quantiles.

Trois façons de produire un quantile « réconcilié » à 90 % :
  A. appliquer P à un vecteur de quantiles            -> cohérent, et le quantile de rien ;
  B. projeter des échantillons tirés INDÉPENDAMMENT   -> la bonne mécanique, sans la dépendance ;
  C. projeter des échantillons tirés CONJOINTEMENT    -> correct (Panagiotelis et al., EJOR 2023).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bias import toy_hierarchy

Z90 = 1.2815515655446004            # quantile 0,90 de la loi normale centrée réduite
NAMES = ["Total", "Region_1", "Region_2", "Region_3"]


@dataclass(frozen=True)
class GaussianHierarchy:
    """Feuilles b ~ N(mu_b, Sig_b), corrélation uniforme rho ; y = S b."""

    mu_b: np.ndarray
    sd_b: np.ndarray
    rho: float

    @property
    def S(self) -> np.ndarray:
        return toy_hierarchy()

    @property
    def Sig_b(self) -> np.ndarray:
        n = len(self.mu_b)
        R = (1 - self.rho) * np.eye(n) + self.rho * np.ones((n, n))
        return np.outer(self.sd_b, self.sd_b) * R

    @property
    def mu_y(self) -> np.ndarray:
        return self.S @ self.mu_b

    @property
    def Sig_y(self) -> np.ndarray:
        return self.S @ self.Sig_b @ self.S.T

    @property
    def sd_y(self) -> np.ndarray:
        return np.sqrt(np.diag(self.Sig_y))

    def projection(self) -> np.ndarray:
        """P de MinT avec la W qu'un pipeline « par série » voit : la diagonale de Sig_y."""
        S, W = self.S, np.diag(np.diag(self.Sig_y))
        Wi = np.linalg.inv(W)
        return S @ np.linalg.inv(S.T @ Wi @ S) @ S.T @ Wi

    def exact_quantiles(self, z: float = Z90) -> np.ndarray:
        return self.mu_y + z * self.sd_y


def default_hierarchy(rho: float = 0.6) -> GaussianHierarchy:
    return GaussianHierarchy(np.array([100.0, 60.0, 40.0]), np.array([12.0, 9.0, 7.0]), rho)


def coverage_experiment(rho: float = 0.6, n: int = 400_000, alpha: float = 0.90,
                        seed: int = 37) -> pd.DataFrame:
    """Reproduit exactement le script du plan (même graine, même ordre de tirage).

    Les modèles de base sont parfaits sur leurs marginales, et les moyennes sont déjà cohérentes :
    la réconciliation des moyennes ne fait rien, tout l'écart observé est un pur effet « quantile ».
    """
    rng = np.random.default_rng(seed)
    h = default_hierarchy(rho)
    P, mu_y, sd_y = h.projection(), h.mu_y, h.sd_y

    qA = P @ h.exact_quantiles()
    qB = np.quantile((mu_y + rng.normal(size=(n, 4)) * sd_y) @ P.T, alpha, axis=0)
    L = np.linalg.cholesky(h.Sig_y + 1e-9 * np.eye(4))
    qC = np.quantile((mu_y + rng.normal(size=(n, 4)) @ L.T) @ P.T, alpha, axis=0)
    truth = (h.mu_b + rng.normal(size=(n, 3)) @ np.linalg.cholesky(h.Sig_b).T) @ h.S.T

    def cov(q):
        return (truth <= q).mean(axis=0)

    return pd.DataFrame({"q_exact": h.exact_quantiles(), "A: P@quantiles": qA, "cov A": cov(qA),
                         "B: éch. indép.": qB, "cov B": cov(qB), "C: éch. dépendants": qC, "cov C": cov(qC)},
                        index=NAMES)


def rho_sweep(rhos=(0.0, 0.3, 0.6, 0.9), n: int = 400_000, seed: int = 37) -> pd.DataFrame:
    """Les deux erreurs vont en sens opposé quand la corrélation augmente."""
    rows = []
    for rho in rhos:
        tab = coverage_experiment(rho=rho, n=n, seed=seed)
        h = default_hierarchy(rho)
        sum_q = float((h.mu_b + Z90 * h.sd_b).sum())
        q_sum = float(tab.loc["Total", "q_exact"])
        rows.append({"rho": rho, "somme des quantiles 90 %": sum_q, "quantile 90 % de la somme": q_sum,
                     "écart": sum_q / q_sum - 1, "couverture B (total)": float(tab.loc["Total", "cov B"]),
                     "couverture C (total)": float(tab.loc["Total", "cov C"])})
    return pd.DataFrame(rows)


def bootstrap_coverage(rho: float = 0.6, T: int = 200, n_samples: int = 50_000, n_truth: int = 400_000,
                       alpha: float = 0.90, seed: int = 0) -> pd.DataFrame:
    """La méthode C en pratique : bootstrap JOINT des résidus in-sample.

    On dispose de T vecteurs de résidus historiques (erreurs de base des 4 séries, au même instant).
      - bootstrap joint       : on tire des INSTANTS -> la dépendance entre séries est conservée ;
      - bootstrap indépendant : on tire chaque série séparément -> la dépendance est détruite.
    """
    rng = np.random.default_rng(seed)
    h = default_hierarchy(rho)
    P = h.projection()
    L = np.linalg.cholesky(h.Sig_y + 1e-9 * np.eye(4))
    E = rng.normal(size=(T, 4)) @ L.T                       # T résidus historiques, joints
    truth = (h.mu_b + rng.normal(size=(n_truth, 3)) @ np.linalg.cholesky(h.Sig_b).T) @ h.S.T

    joint = h.mu_y + E[rng.integers(0, T, n_samples)]
    indep = h.mu_y + np.column_stack([E[rng.integers(0, T, n_samples), j] for j in range(4)])
    out = {}
    for name, draws in [("bootstrap joint", joint), ("bootstrap indépendant", indep)]:
        q = np.quantile(draws @ P.T, alpha, axis=0)
        out[f"q {name}"] = q
        out[f"cov {name}"] = (truth <= q).mean(axis=0)
    return pd.DataFrame(out, index=NAMES)
