"""Petits outils sur les distributions discrètes (pmf sur 0, 1, 2, …) et gaussiennes, sous une même interface.

Une prévision est soit une pmf (tableau de probabilités indexé par 0, 1, 2, …), soit une loi normale
(moyenne, écart-type). Les scores du module `scores` acceptent l'une ou l'autre.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Pmf:
    p: np.ndarray                                   # p[k] = P(Y = k), k = 0 … len − 1

    @classmethod
    def of(cls, p) -> Pmf:
        p = np.atleast_1d(np.asarray(p, dtype=float)).copy()
        if (p < -1e-12).any() or not np.isfinite(p).all():
            raise ValueError("pmf invalide : probabilités négatives ou non finies")
        return cls(np.clip(p, 0, None) / p.sum())

    @property
    def mean(self) -> float:
        return float(np.arange(len(self.p)) @ self.p)

    @property
    def var(self) -> float:
        k = np.arange(len(self.p))
        return float(((k - self.mean) ** 2) @ self.p)

    def cdf(self, k: np.ndarray) -> np.ndarray:
        c = np.cumsum(self.p)
        k = np.asarray(k)
        return np.where(k < 0, 0.0, c[np.clip(k, 0, len(c) - 1).astype(int)])

    def quantile(self, q: float) -> float:
        return float(np.searchsorted(np.cumsum(self.p), q - 1e-12))

    def prob_negative(self) -> float:
        return 0.0


@dataclass(frozen=True)
class Normal:
    mu: float
    sd: float

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def var(self) -> float:
        return self.sd ** 2

    def cdf(self, k: np.ndarray) -> np.ndarray:
        """Discrétisée sur les entiers : P(Y ≤ k) = Φ((k + 0,5 − μ) / σ), pour comparer à une pmf."""
        return stats.norm.cdf((np.asarray(k) + 0.5 - self.mu) / self.sd)

    def quantile(self, q: float) -> float:
        return float(stats.norm.ppf(q, self.mu, self.sd))

    def prob_negative(self) -> float:
        """Masse de probabilité sur des ventes négatives : P(Y < −0,5) en version discrétisée."""
        return float(stats.norm.cdf((-0.5 - self.mu) / self.sd))
