"""Données : hiérarchie synthétique région × canal et découpage train / test."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hierarchicalforecast.utils import aggregate

REGIONS = ("Bretagne", "PaysLoire")
CANAUX = ("GMS", "RHD")


@dataclass(frozen=True)
class Hierarchy:
    """Une hiérarchie prête pour HierarchicalForecast.

    Y_df : format long (unique_id, ds, y), toutes séries confondues.
    S_df : matrice de sommation (unique_id + une colonne par feuille). Agrégats d'abord, feuilles ensuite.
    tags : niveau -> liste des unique_id du niveau.
    """

    Y_df: pd.DataFrame
    S_df: pd.DataFrame
    tags: dict[str, np.ndarray]

    @property
    def order(self) -> list[str]:
        return self.S_df["unique_id"].tolist()

    @property
    def S(self) -> np.ndarray:
        return self.S_df.set_index("unique_id").to_numpy(dtype=float)


def make_region_channel_data(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """Feuilles région × canal trimestrielles : niveau + saisonnalité annuelle + marche aléatoire.

    Reproduit exactement le squelette du plan W37 (graine 0, 80 trimestres depuis 2015).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n, freq="QS")
    rows = []
    for region in REGIONS:
        for canal in CANAUX:
            base = 100 + 30 * (region == "Bretagne") + 20 * (canal == "GMS")
            y = base + 10 * np.sin(2 * np.pi * np.arange(n) / 4) + np.cumsum(rng.normal(0, 1.5, n))
            rows.append(pd.DataFrame({"ds": dates, "Region": region, "Canal": canal, "y": y}))
    return pd.concat(rows, ignore_index=True)


def build_hierarchy(df: pd.DataFrame, spec: list[list[str]] | None = None) -> Hierarchy:
    spec = spec or [["Region"], ["Region", "Canal"]]
    Y_df, S_df, tags = aggregate(df=df, spec=spec)
    return Hierarchy(Y_df=Y_df, S_df=S_df, tags=tags)


def train_test_split(Y_df: pd.DataFrame, h: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Les h dernières dates partent en test. Aucune date n'est commune aux deux."""
    cutoff = np.sort(Y_df["ds"].unique())[-h]
    return Y_df[Y_df["ds"] < cutoff], Y_df[Y_df["ds"] >= cutoff]


def to_wide(df: pd.DataFrame, order: list[str], value: str = "y") -> pd.DataFrame:
    """Format long -> matrice (dates × séries) dans l'ordre des lignes de S."""
    return df.pivot(index="ds", columns="unique_id", values=value)[order]
