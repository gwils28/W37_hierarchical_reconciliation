"""Un exemple que l'on peut faire à la main : 2 articles, 1 total, réconciliation par conditionnement EXACTE.

Base : b1 ~ Poisson(λ1), b2 ~ Poisson(λ2), indépendants ; total ~ N(μ, σ²), incohérent avec λ1 + λ2.

Conditionnement (Zambon, Azzimonti, Corani, 2024) : on garde la loi jointe des articles, et on la
repondère par la vraisemblance que le total donne à leur somme :

    P̃(b1, b2) ∝ Pois(b1; λ1) · Pois(b2; λ2) · φ((b1 + b2 − μ) / σ)

Tout reste sur les entiers ≥ 0 : une vente négative a une probabilité nulle par construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .pmf import Normal, Pmf


@dataclass(frozen=True)
class Toy:
    lam1: float = 0.4
    lam2: float = 0.8
    mu_total: float = 0.6
    sd_total: float = 0.5
    K: int = 40                                     # troncature de l'énumération (P(b > 40) négligeable)

    def exact(self) -> dict[str, Pmf]:
        k = np.arange(self.K + 1)
        joint = (stats.poisson.pmf(k, self.lam1)[:, None] * stats.poisson.pmf(k, self.lam2)[None, :]
                 * stats.norm.pdf(k[:, None] + k[None, :], self.mu_total, self.sd_total))
        joint /= joint.sum()
        total = np.bincount((k[:, None] + k[None, :]).ravel(), weights=joint.ravel())
        return {"b1": Pmf.of(joint.sum(axis=1)), "b2": Pmf.of(joint.sum(axis=0)), "total": Pmf.of(total)}

    def base(self) -> dict:
        k = np.arange(self.K + 1)
        return {"b1": Pmf.of(stats.poisson.pmf(k, self.lam1)), "b2": Pmf.of(stats.poisson.pmf(k, self.lam2)),
                "total": Normal(self.mu_total, self.sd_total)}

    def gaussian_mint(self) -> dict[str, Normal]:
        """MinT sur des gaussiennes : chaque Poisson remplacée par N(λ, λ), W diagonale, formule fermée."""
        S = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        W = np.diag([self.sd_total ** 2, self.lam1, self.lam2])
        y = np.array([self.mu_total, self.lam1, self.lam2])
        Wi = np.linalg.inv(W)
        G = np.linalg.solve(S.T @ Wi @ S, S.T @ Wi)
        m = S @ G @ y
        C = S @ G @ W @ G.T @ S.T                    # covariance de l'erreur réconciliée (formule MinT)
        sd = np.sqrt(np.diag(C))
        return {"total": Normal(m[0], sd[0]), "b1": Normal(m[1], sd[1]), "b2": Normal(m[2], sd[2])}

    def buis_by_hand(self, num_samples: int = 200_000, seed: int = 37) -> dict[str, Pmf]:
        """L'algorithme BUIS en trois étapes, sans bibliothèque :

        1. tirer les articles selon leur loi de base (ici indépendants) ;
        2. donner à chaque tirage un poids : la densité que la prévision du total donne à leur somme ;
        3. rééchantillonner les tirages selon ces poids.
        Les tirages retenus suivent la loi conditionnelle de `exact()`, et somment exactement par
        construction.
        """
        rng = np.random.default_rng(seed)
        b1, b2 = rng.poisson(self.lam1, num_samples), rng.poisson(self.lam2, num_samples)
        w = stats.norm.pdf(b1 + b2, self.mu_total, self.sd_total)
        keep = rng.choice(num_samples, size=num_samples, p=w / w.sum())
        b1, b2 = b1[keep], b2[keep]
        return {"total": Pmf.of(np.bincount(b1 + b2)), "b1": Pmf.of(np.bincount(b1)),
                "b2": Pmf.of(np.bincount(b2))}

    def buis(self, num_samples: int = 200_000, seed: int = 37) -> dict[str, Pmf]:
        """Le même conditionnement, par l'algorithme BUIS de BayesReconPy (échantillonnage d'importance)."""
        from bayesreconpy.reconc_buis import reconc_buis
        r = reconc_buis(np.array([[1.0, 1.0]]),
                        [{"mean": self.mu_total, "sd": self.sd_total},
                         {"lambda": self.lam1}, {"lambda": self.lam2}],
                        in_type="params", distr=["gaussian", "poisson", "poisson"],
                        num_samples=num_samples, seed=seed, suppress_warnings=True)
        s = np.asarray(r["reconciled_samples"]).astype(int)
        return {name: Pmf.of(np.bincount(row)) for name, row in zip(("total", "b1", "b2"), s, strict=True)}
