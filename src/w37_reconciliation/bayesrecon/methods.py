"""Quatre façons de réconcilier les mêmes prévisions de base, avec la même information.

| méthode | ce qui est réconcilié | support des articles |
|---|---|---|
| MinT gaussien | des gaussiennes (moyenne + covariance), par projection | ℝ : ventes négatives possibles |
| bottom-up | rien : les articles gardent leur pmf, les agrégats en sont la somme | 0, 1, 2, … |
| MixCond | loi jointe des articles conditionnée aux agrégats (échantillonnage d'importance) | 0, 1, 2, … |
| TD-cond | en deux temps : agrégats d'abord, puis répartition vers les articles | 0, 1, 2, … |

⚠️ `reconc_td_cond` (BayesReconPy 0.5.0) **modifie en place** les pmf qu'on lui passe (≈ 1e-9 ajouté à
certaines probabilités, sans renormalisation). Un appel suivant avec les mêmes objets peut alors échouer
(`probabilities do not sum to 1`). Chaque méthode ci-dessous reçoit donc des **copies**.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from bayesreconpy.reconc_gaussian import reconc_gaussian
from bayesreconpy.reconc_mix_cond import reconc_mix_cond
from bayesreconpy.reconc_td_cond import reconc_td_cond
from bayesreconpy.shrink_cov import _schafer_strimmer_cov
from scipy.linalg import block_diag

from .example import M5Example
from .pmf import Normal, Pmf


@dataclass
class Reconciled:
    name: str
    upper: list                       # une distribution (Pmf ou Normal) par agrégat
    bottom: list                      # une distribution par article
    seconds: float
    info: dict = field(default_factory=dict)


def upper_covariance(ex: M5Example) -> tuple[np.ndarray, float]:
    """Covariance des agrégats : Schäfer-Strimmer sur leurs 1 941 jours de résidus (comme le dépôt)."""
    s = _schafer_strimmer_cov(ex.residuals_u)
    return np.asarray(s["shrink_cov"]), float(s["lambda_star"])


def base(ex: M5Example) -> Reconciled:
    """Les prévisions de base, NON réconciliées (référence pour l'incohérence et la précision)."""
    upper = [Normal(m, s) for m, s in zip(ex.mu_u, ex.sd_u, strict=True)]
    return Reconciled("base", upper, list(ex.pmf_b), 0.0)


def gaussian_mint(ex: M5Example) -> Reconciled:
    """MinT sur des gaussiennes : W = diag par blocs (Σ des agrégats, variances des pmf d'articles).

    C'est la meilleure W disponible avec cette information : les articles n'ont pas de résidus, donc pas
    de covariance estimable entre eux. Les pmf sont remplacées par leur moyenne et leur variance.
    """
    t = time.perf_counter()
    Su, lam = upper_covariance(ex)
    W = block_diag(Su, np.diag(ex.var_b + 1e-9))
    g = reconc_gaussian(ex.A, list(np.r_[ex.mu_u, ex.mean_b]), W)
    m = np.asarray(g["bottom_reconciled_mean"], dtype=float).ravel()
    C = np.asarray(g["bottom_reconciled_covariance"], dtype=float)
    mu_u, var_u = ex.A @ m, np.einsum("ij,jk,ik->i", ex.A, C, ex.A)
    return Reconciled("MinT gaussien", [Normal(a, np.sqrt(v)) for a, v in zip(mu_u, var_u, strict=True)],
                      [Normal(a, np.sqrt(v)) for a, v in zip(m, np.diag(C), strict=True)],
                      time.perf_counter() - t, {"lambda_agrégats": lam})


def _upper_pmf_from_samples(samples: np.ndarray) -> list[Pmf]:
    return [Pmf.of(np.bincount(s.astype(int))) for s in samples]


def bottom_up(ex: M5Example, num_samples: int, seed: int) -> Reconciled:
    """Bottom-up probabiliste : articles inchangés, agrégats = somme de tirages INDÉPENDANTS des articles."""
    t = time.perf_counter()
    rng = np.random.default_rng(seed)
    U = np.zeros((ex.A.shape[0], num_samples))                # sommes cumulées : pas de matrice 3049 × N
    for j, p in enumerate(ex.pmf_b):
        U[ex.A[:, j] == 1] += rng.choice(len(p.p), size=num_samples, p=p.p)
    return Reconciled("bottom-up", _upper_pmf_from_samples(U), list(ex.pmf_b), time.perf_counter() - t)


def _conditioning(fn, name: str, ex: M5Example, num_samples: int, seed: int) -> Reconciled:
    Su, _ = upper_covariance(ex)
    # copies défensives : reconc_td_cond modifie ses entrées en place
    fc_bottom = {n: p.p.copy() for n, p in zip(ex.bottom_names, ex.pmf_b, strict=True)}
    fc_upper = {"mu": ex.mu_u.copy(), "Sigma": Su}
    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = fn(ex.A, fc_bottom, fc_upper, bottom_in_type="pmf", num_samples=num_samples, return_type="pmf",
               seed=seed, suppress_warnings=True)
    seconds = time.perf_counter() - t
    info = {k: float(v) for k, v in r.items() if k not in ("bottom_reconciled", "upper_reconciled")}
    return Reconciled(name, [Pmf.of(p) for p in r["upper_reconciled"]["pmf"]],
                      [Pmf.of(p) for p in r["bottom_reconciled"]["pmf"]], seconds, info)


def mix_cond(ex: M5Example, num_samples: int, seed: int) -> Reconciled:
    return _conditioning(reconc_mix_cond, "MixCond", ex, num_samples, seed)


def td_cond(ex: M5Example, num_samples: int, seed: int) -> Reconciled:
    return _conditioning(reconc_td_cond, "TD-cond", ex, num_samples, seed)


def run_all(ex: M5Example, num_samples: int, seed: int) -> list[Reconciled]:
    return [base(ex), gaussian_mint(ex), bottom_up(ex, num_samples, seed),
            mix_cond(ex, num_samples, seed), td_cond(ex, num_samples, seed)]
