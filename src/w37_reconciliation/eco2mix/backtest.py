"""Backtest à origines glissantes : pour chaque origine, entraîner, prévoir, réconcilier, enregistrer.

Le protocole tient en trois invariants, vérifiés par assertion à CHAQUE origine :

- la fenêtre d'entraînement (et donc les résidus qui servent à estimer W) finit strictement avant l'origine ;
- la fenêtre de test commence à l'origine et dure exactement h heures, pour toutes les séries ;
- aucune valeur manquante n'entre dans les modèles.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace
from statsforecast import StatsForecast
from statsforecast.models import MSTL, SeasonalNaive

from ..data import Hierarchy, to_wide
from ..mint import shrunk_covariance

# noms courts, utilisés partout en aval (tableaux, figures)
METHOD_LABELS = {"BottomUp": "BU", "ols": "OLS", "wls_struct": "WLS struct", "wls_var": "WLS var",
                 "mint_shrink": "MinT shrink"}


@dataclass(frozen=True)
class BacktestSpec:
    h: int = 24
    season: int = 168
    train_weeks: int = 52
    reconcile_model: str = "MSTL"
    methods: tuple[str, ...] = ("BottomUp", "ols", "wls_struct", "wls_var", "mint_shrink")
    n_jobs: int = -1

    @classmethod
    def from_config(cls, cfg: dict) -> BacktestSpec:
        b = cfg["backtest"]
        return cls(h=b["h"], season=b["season"], train_weeks=b["train_weeks"],
                   reconcile_model=b["reconcile_model"], methods=tuple(b["methods"]))

    @property
    def train_hours(self) -> int:
        return self.train_weeks * 168


def split_at(Y_df: pd.DataFrame, origin: pd.Timestamp,
             spec: BacktestSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train = [origin − train_hours, origin) ; test = [origin, origin + h)."""
    start, end = origin - pd.Timedelta(hours=spec.train_hours), origin + pd.Timedelta(hours=spec.h)
    train = Y_df[(Y_df["ds"] >= start) & (Y_df["ds"] < origin)]
    test = Y_df[(Y_df["ds"] >= origin) & (Y_df["ds"] < end)]
    n_series = Y_df["unique_id"].nunique()
    if len(test) != n_series * spec.h:
        raise ValueError(f"origine {origin} : test incomplet ({len(test)} lignes, "
                         f"{n_series * spec.h} attendues)")
    if len(train) != n_series * spec.train_hours:
        raise ValueError(f"origine {origin} : historique insuffisant ({len(train)} lignes)")
    if train["y"].isna().any() or test["y"].isna().any():
        raise ValueError(f"origine {origin} : valeurs manquantes")
    return train, test


def mase_scale(train: pd.DataFrame, order: list[str], season: int) -> pd.Series:
    """Dénominateur du MASE : erreur absolue moyenne du naïf saisonnier IN-SAMPLE, sur le train seulement."""
    y = to_wide(train, order).to_numpy()
    return pd.Series(np.abs(y[season:] - y[:-season]).mean(axis=0), index=order)


def run_origin(hier: Hierarchy, origin: pd.Timestamp, spec: BacktestSpec) -> dict[str, pd.DataFrame]:
    """Un pas du backtest. Renvoie les prévisions (format long), l'échelle MASE et le diagnostic de W."""
    order = hier.order
    train, test = split_at(hier.Y_df, origin, spec)

    sf = StatsForecast(models=[SeasonalNaive(season_length=spec.season), MSTL(season_length=[24, 168])],
                       freq="h", n_jobs=spec.n_jobs)
    Y_hat = sf.forecast(df=train, h=spec.h, fitted=True)
    Y_fit = sf.forecast_fitted_values()
    if Y_fit["ds"].max() >= origin:
        raise AssertionError(f"fuite : résidus postérieurs à l'origine {origin}")

    m = spec.reconcile_model
    reconcilers = [BottomUp() if k == "BottomUp" else MinTrace(method=k) for k in spec.methods]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Y_rec = HierarchicalReconciliation(reconcilers=reconcilers).reconcile(
            Y_hat_df=Y_hat[["unique_id", "ds", m]], Y_df=Y_fit[["unique_id", "ds", "y", m]],
            S_df=hier.S_df, tags=hier.tags)

    rename = {m: "base"}
    for k in spec.methods:
        col = f"{m}/BottomUp" if k == "BottomUp" else f"{m}/MinTrace_method-{k}"
        rename[col] = METHOD_LABELS[k]
    fc = (Y_rec.rename(columns=rename)[["unique_id", "ds", *rename.values()]]
          .merge(Y_hat[["unique_id", "ds", "SeasonalNaive"]].rename(columns={"SeasonalNaive": "SN"}),
                 on=["unique_id", "ds"])
          .merge(test[["unique_id", "ds", "y"]], on=["unique_id", "ds"]))
    fc.insert(0, "origin", origin)
    fc.insert(3, "horizon", ((fc["ds"] - origin) / pd.Timedelta(hours=1)).astype(int) + 1)

    E = (to_wide(Y_fit, order, "y") - to_wide(Y_fit, order, m)).dropna().to_numpy()
    cov = shrunk_covariance(E, base="library")
    diag = pd.DataFrame([{"origin": origin, "T_residus": E.shape[0], "m": E.shape[1], "lambda": cov.lam,
                          "cond_W1": float(np.linalg.cond(cov.W1)), "cond_W": cov.condition_number}])
    scale = mase_scale(train, order, spec.season).rename("scale").rename_axis("unique_id").reset_index()
    scale.insert(0, "origin", origin)
    return {"forecasts": fc, "scales": scale, "diagnostics": diag}


def run_backtest(hier: Hierarchy, origins: list[pd.Timestamp], spec: BacktestSpec,
                 log=print) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {"forecasts": [], "scales": [], "diagnostics": []}
    for i, origin in enumerate(origins, 1):
        out = run_origin(hier, origin, spec)
        for k, v in out.items():
            parts[k].append(v)
        log(f"  [{i}/{len(origins)}] {origin:%Y-%m-%d %a} λ={out['diagnostics']['lambda'].iat[0]:.3f}")
    return {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}


def origins_from_config(cfg: dict, which: str = "main") -> list[pd.Timestamp]:
    b = cfg["backtest"]
    if which == "main":
        return [pd.Timestamp(d) for d in b["main_origins"]]
    r = b["robustness"]
    return list(pd.date_range(r["start"], r["end"], freq=f"{r['every_days']}D"))


def reconcile_with_past_errors(forecasts: pd.DataFrame, hier: Hierarchy, min_past_origins: int = 20,
                               model_col: str = "base") -> pd.DataFrame:
    """Expérience contrôlée : la même réconciliation, mais avec W estimée sur les ERREURS HORS ÉCHANTILLON
    des origines passées, au lieu des résidus in-sample.

    À l'origine o, on n'utilise que les origines o' dont la fenêtre de test est terminée : o' + h ≤ o.
    Ajoute deux colonnes : « MinT oos » (W pleine, shrinkage de Schäfer-Strimmer) et « WLS var oos »
    (W diagonale).
    Les origines qui ont moins de `min_past_origins` origines passées sont retirées du résultat.
    """
    from ..mint import mint_projection

    order, S = hier.order, hier.S
    h = int(forecasts["horizon"].max())
    wide = forecasts.pivot_table(index=["origin", "ds"], columns="unique_id", values=[model_col, "y"])
    base, y = wide[model_col][order], wide["y"][order]
    errors = y - base
    origin_of = errors.index.get_level_values("origin")
    parts = []
    for o in sorted(forecasts["origin"].unique()):
        past = errors[origin_of + pd.Timedelta(hours=h) <= o]
        if past.index.get_level_values("origin").nunique() < min_past_origins:
            continue
        cov = shrunk_covariance(past.to_numpy(), base="library")
        b = base.loc[o].to_numpy()
        out = pd.DataFrame({"origin": o, "ds": base.loc[o].index})
        for name, W in (("MinT oos", cov.W), ("WLS var oos", np.diag(np.diag(cov.W1)))):
            rec = b @ mint_projection(S, W)[1].T
            out = out.join(pd.DataFrame(rec, columns=[f"{name}|{u}" for u in order]))
        parts.append(out)
    new = pd.concat(parts, ignore_index=True).melt(id_vars=["origin", "ds"])
    new[["method", "unique_id"]] = new.pop("variable").str.split("|", expand=True)
    new = new.pivot_table(index=["origin", "unique_id", "ds"], columns="method", values="value").reset_index()
    new.columns.name = None
    return forecasts.merge(new, on=["origin", "unique_id", "ds"], how="inner")
