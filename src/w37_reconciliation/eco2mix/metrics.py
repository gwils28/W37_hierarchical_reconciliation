"""MASE par niveau (jamais une moyenne globale), incohérence résiduelle, chiffre clé.

Pourquoi par niveau : le total France pèse environ 12 fois une région. Une moyenne globale mélange un
gain au national et une perte en région, et c'est précisément ce que le bloc 4 a montré qu'il fallait
voir séparément.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Hierarchy


def methods_in(forecasts: pd.DataFrame) -> list[str]:
    fixed = {"origin", "unique_id", "ds", "horizon", "y"}
    return [c for c in forecasts.columns if c not in fixed]


def level_of(hier: Hierarchy) -> dict[str, str]:
    """unique_id -> nom de niveau ('France' ou 'Régions')."""
    top, leaves = list(hier.tags.values())
    return {**{u: "France" for u in top}, **{u: "Régions" for u in leaves}}


def mase_long(forecasts: pd.DataFrame, scales: pd.DataFrame, hier: Hierarchy) -> pd.DataFrame:
    """MASE par (origine, série, méthode) : MAE sur les h heures / échelle naïve saisonnière du train."""
    methods = methods_in(forecasts)
    err = forecasts[methods].sub(forecasts["y"], axis=0).abs()
    mae = pd.concat([forecasts[["origin", "unique_id"]], err], axis=1).groupby(["origin", "unique_id"]).mean()
    # aligné sur les origines des PRÉVISIONS (les échelles peuvent en couvrir davantage)
    scale = scales.set_index(["origin", "unique_id"])["scale"].reindex(mae.index)
    if scale.isna().any():
        raise ValueError("échelle MASE manquante pour certaines (origine, série) des prévisions")
    mase = mae.div(scale, axis=0)
    out = mase.reset_index().melt(id_vars=["origin", "unique_id"], var_name="method", value_name="mase")
    out["level"] = out["unique_id"].map(level_of(hier))
    return out


def mase_by_level(mase: pd.DataFrame) -> pd.DataFrame:
    """Une valeur par (origine, niveau, méthode) : France = sa série ; Régions = moyenne des 12."""
    return mase.groupby(["origin", "level", "method"], as_index=False)["mase"].mean()


def summary_table(by_level: pd.DataFrame, order: list[str] | None = None) -> pd.DataFrame:
    """Le tableau de la Definition of Done : MASE moyen ± écart-type entre origines, niveau × méthode."""
    t = by_level.groupby(["level", "method"])["mase"].agg(["mean", "std", "count"]).reset_index()
    wide = t.pivot(index="method", columns="level", values=["mean", "std"])
    wide.columns = [f"{lvl} {stat}" for stat, lvl in wide.columns]
    wide = wide[[f"{lvl} {s}" for lvl in ("France", "Régions") for s in ("mean", "std")]]
    return wide.loc[order] if order else wide.sort_values("Régions mean")


def incoherence(forecasts: pd.DataFrame, hier: Hierarchy, method: str) -> pd.Series:
    """max |total − Σ régions| par origine, en MW. Nul (précision machine) après réconciliation."""
    top, leaves = list(hier.tags.values())
    wide = forecasts.pivot_table(index=["origin", "ds"], columns="unique_id", values=method)
    gap = (wide[top[0]] - wide[list(leaves)].sum(axis=1)).abs()
    return gap.groupby(level="origin").max()


def relative_change(by_level: pd.DataFrame, method: str, reference: str, level: str = "Régions") -> pd.Series:
    """(MASE_method / MASE_reference − 1) par origine, au niveau demandé. Négatif = method fait mieux."""
    w = by_level[by_level["level"] == level].pivot(index="origin", columns="method", values="mase")
    return w[method] / w[reference] - 1


def key_figure(by_level: pd.DataFrame, method: str = "MinT shrink", reference: str = "BU") -> dict:
    """Le chiffre clé du plan : de combien `method` bouge le MASE régional par rapport à `reference`."""
    out = {}
    for level in ("Régions", "France"):
        r = relative_change(by_level, method, reference, level)
        out[level] = {"mean": float(r.mean()), "std": float(r.std()), "min": float(r.min()),
                      "max": float(r.max()), "n_origins": int(r.size),
                      "wins": int((r < 0).sum())}
    return out


def assert_coherent(forecasts: pd.DataFrame, hier: Hierarchy, methods: list[str], tol: float) -> pd.DataFrame:
    """L'assertion de la Definition of Done : échoue si UNE méthode dépasse `tol` à UNE origine."""
    table = pd.DataFrame({m: incoherence(forecasts, hier, m) for m in methods})
    worst = table.max()
    bad = worst[worst > tol]
    if len(bad):
        raise AssertionError(f"incohérence > {tol} : {bad.to_dict()}")
    if not np.isfinite(forecasts[methods].to_numpy()).all():
        raise AssertionError("prévisions réconciliées non finies")
    return table
