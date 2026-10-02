"""Projection MinT : G = (S' W⁻¹ S)⁻¹ S' W⁻¹ et P = S G."""
from __future__ import annotations

import numpy as np


def mint_projection(S: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (G, P).

    G (nb × m) : combine les m prévisions de base en nb feuilles cohérentes.
    P (m × m)  : projection oblique (P² = P) sur l'espace cohérent, P = S G.
    `solve` plutôt qu'une inversion explicite de S'W⁻¹S : plus stable numériquement.
    `pinv` sur W : tolère une W singulière (m > T sans shrinkage), sans pour autant la rendre fiable.
    """
    Wi = np.linalg.pinv(W)
    G = np.linalg.solve(S.T @ Wi @ S, S.T @ Wi)
    return G, S @ G


def reconcile(y_hat: np.ndarray, P: np.ndarray) -> np.ndarray:
    """y_hat (h × m), une ligne par horizon -> prévisions cohérentes (h × m)."""
    return y_hat @ P.T
