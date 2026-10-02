"""Les assertions qui font foi (bloc 2)."""
from __future__ import annotations

import numpy as np


def unbiasedness_error(P: np.ndarray, S: np.ndarray) -> float:
    """max |P S − S| : nul ⟺ SGS = S ⟺ la réconciliation préserve l'absence de biais."""
    return float(np.abs(P @ S - S).max())


def incoherence(y: np.ndarray, S: np.ndarray) -> float:
    """max |y − S b|, où b = les nb dernières colonnes de y (les feuilles)."""
    nb = S.shape[1]
    return float(np.abs(y - y[:, -nb:] @ S.T).max())


def relative_gap(mine: np.ndarray, reference: np.ndarray) -> float:
    """Écart max rapporté au niveau moyen de la référence."""
    return float(np.abs(mine - reference).max() / np.abs(reference).mean())
