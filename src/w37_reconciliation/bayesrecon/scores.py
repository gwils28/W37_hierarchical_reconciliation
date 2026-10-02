"""Scores : prévisions négatives, cohérence, précision ponctuelle et probabiliste, par niveau.

| score | ce qu'il mesure | pour |
|---|---|---|
| part de moyennes < 0 | une prévision ponctuelle impossible | la question du plan |
| part de bornes basses < 0 | un intervalle qui contient des ventes impossibles | idem, en probabiliste |
| masse sous zéro | la probabilité donnée à des ventes négatives | idem |
| MASE | erreur de la médiane, divisée par l'échelle naïve de la série (`Q`) | précision ponctuelle |
| MIS | largeur de l'intervalle + pénalité si la valeur sort (Gneiting et Raftery, 2007) | intervalles |
| RPS | écart entre la fonction de répartition prévue et la valeur réalisée, sur les entiers | loi complète |
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pmf import Normal, Pmf


def negatives(dists: list, alpha: float = 0.10) -> dict:
    means = np.array([d.mean for d in dists])
    low = np.array([d.quantile(alpha / 2) for d in dists])
    return {"moyennes < 0": int((means < 0).sum()), "part moyennes < 0": float((means < 0).mean()),
            "bornes basses < 0": int((low < 0).sum()), "part bornes basses < 0": float((low < 0).mean()),
            "masse sous zéro (moyenne)": float(np.mean([d.prob_negative() for d in dists]))}


def interval_score(d, y: float, alpha: float = 0.10) -> float:
    lo, hi = d.quantile(alpha / 2), d.quantile(1 - alpha / 2)
    return (hi - lo) + 2 / alpha * max(lo - y, 0) + 2 / alpha * max(y - hi, 0)


def rps(d, y: float) -> float:
    """Ranked probability score sur les entiers : Σ_k (F(k) − 1{y ≤ k})². Plus bas = mieux."""
    if isinstance(d, Pmf):
        lo, hi = -1, max(len(d.p) - 1, int(y))
    elif isinstance(d, Normal):
        lo = int(np.floor(min(d.mu - 8 * d.sd, y))) - 1
        hi = int(np.ceil(max(d.mu + 8 * d.sd, y)))
    else:
        raise TypeError(type(d))
    k = np.arange(lo, hi + 1)
    return float(((d.cdf(k) - (y <= k)) ** 2).sum())


def level_scores(dists: list, actual: np.ndarray, Q: np.ndarray, alpha: float = 0.10,
                 with_rps: bool = True) -> dict:
    med = np.array([d.quantile(0.5) for d in dists])
    out = {"MASE": float(np.mean(np.abs(actual - med) / Q)),
           "MIS": float(np.mean([interval_score(d, y, alpha) for d, y in zip(dists, actual, strict=True)]))}
    if with_rps:
        out["RPS"] = float(np.mean([rps(d, y) for d, y in zip(dists, actual, strict=True)]))
    return out


def incoherence(rec, A: np.ndarray) -> float:
    """max |moyenne d'un agrégat − Σ moyennes de ses articles|. Pour un échantillonnage, c'est l'erreur
    Monte-Carlo sur les moyennes : les ÉCHANTILLONS, eux, sont exactement cohérents."""
    mu_u = np.array([d.mean for d in rec.upper])
    mu_b = np.array([d.mean for d in rec.bottom])
    return float(np.abs(mu_u - A @ mu_b).max())


def summary(recs: list, ex, alpha: float = 0.10) -> pd.DataFrame:
    rows = {}
    for r in recs:
        row = {"secondes": r.seconds, "incohérence max (ventes)": incoherence(r, ex.A)}
        row.update({f"articles · {k}": v for k, v in negatives(r.bottom, alpha).items()})
        bottom = level_scores(r.bottom, ex.actual_b, ex.Q_b, alpha)
        row.update({f"articles · {k}": v for k, v in bottom.items()})
        row.update({f"agrégats · {k}": v for k, v in
                    level_scores(r.upper, ex.actual_u, ex.Q_u, alpha, with_rps=False).items()})
        rows[r.name] = row
    return pd.DataFrame(rows).T


def per_series(dists: list, actual: np.ndarray, Q: np.ndarray, alpha: float = 0.10) -> pd.DataFrame:
    """MASE, MIS et RPS, une ligne par série (pour les tests appariés et les skill scores)."""
    return pd.DataFrame({
        "MASE": np.abs(actual - np.array([d.quantile(0.5) for d in dists])) / Q,
        "MIS": [interval_score(d, y, alpha) for d, y in zip(dists, actual, strict=True)],
        "RPS": [rps(d, y) for d, y in zip(dists, actual, strict=True)]})


def skill(base: np.ndarray, method: np.ndarray) -> float:
    """Skill score symétrique moyen, en % (positif = mieux que la base), comme la vignette R de bayesRecon :
    100 · (s_base − s) / ((s_base + s) / 2), série par série, 0 quand les deux scores sont nuls."""
    base, method = np.asarray(base, float), np.asarray(method, float)
    den = (base + method) / 2
    s = np.where(den == 0, 0.0, 100 * (base - method) / np.where(den == 0, 1.0, den))
    return float(s.mean())


def skill_table(recs: list, ex, alpha: float = 0.10) -> pd.DataFrame:
    """Skill scores de chaque méthode par rapport à la base, par niveau et par score."""
    by_name = {r.name: r for r in recs}
    rows = {}
    levels = (("agrégats", "upper", ex.actual_u, ex.Q_u), ("articles", "bottom", ex.actual_b, ex.Q_b))
    for level, attr, y, Q in levels:
        ref = per_series(getattr(by_name["base"], attr), y, Q, alpha)
        for name, r in by_name.items():
            if name == "base":
                continue
            s = per_series(getattr(r, attr), y, Q, alpha)
            for col in ref:
                rows.setdefault((level, col), {})[name] = skill(ref[col], s[col])
    return pd.DataFrame(rows).T
