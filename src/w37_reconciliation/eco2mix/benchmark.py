"""Benchmark : baselines naïves et statistiques face aux réconciliations, et études de confirmation.

Le backtest (`backtest.py`) entraîne deux modèles. Ce module ajoute, SANS rien réentraîner :

- **trois baselines naïves**, calculées directement sur l'historique, à chaque origine :
  « Naïf » (dernière valeur connue), « SN 24 h » (la veille, même heure), « Moy. 4 sem. » (moyenne des
  4 mêmes heures des 4 semaines précédentes). Le naïf saisonnier 168 h (« SN ») est déjà dans le backtest ;
- **la prévision J−1 de RTE** au niveau France : repère externe, faite la veille AVEC la météo ;
- les outils de comparaison : gain relatif, courbes par horizon et par mois, convergence de la moyenne,
  plusieurs métriques, probabilité d'être la meilleure méthode (bootstrap sur les origines).

**Les baselines naïves sont cohérentes par construction** : ce sont des combinaisons linéaires de valeurs
passées, et le passé du total est la somme des passés des régions. Les réconcilier ne changerait rien.

Convention de signe : **gain = 1 − MASE(méthode) / MASE(référence)**, positif = la méthode fait mieux. C'est
l'opposé de `metrics.relative_change`, plus naturel pour un benchmark (« combien on gagne »).

**Gain agrégé ou gain moyen par origine.** Le benchmark utilise le gain AGRÉGÉ, rapport des MASE moyens :
il se relit directement dans le tableau des MASE. La moyenne des gains PAR ORIGINE (celle du chiffre clé,
`metrics.key_figure`) explose dès que deux méthodes sont très différentes : un jour facile où la référence
fait une erreur minuscule donne un rapport énorme. Les deux mesures s'accordent sur le signe entre méthodes
proches ; on garde la part d'origines gagnées comme mesure par origine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..data import Hierarchy, to_wide
from .metrics import level_of, mase_by_level, mase_long

NAIVE_BASELINES = ["Naïf", "SN 24 h", "Moy. 4 sem."]
# ordre de lecture du benchmark : du plus simple au plus élaboré, puis le repère externe
BENCHMARK_ORDER = ["Naïf", "Moy. 4 sem.", "SN", "SN 24 h", "base", "BU", "OLS", "WLS struct", "WLS var",
                   "MinT shrink", "RTE J-1", "RTE J-1 débiaisée"]
# libellés lisibles des colonnes du backtest
DISPLAY = {"SN": "SN 168 h", "base": "MSTL (base)"}


def naive_baselines(hier: Hierarchy, origins: list[pd.Timestamp], h: int = 24) -> pd.DataFrame:
    """Les trois baselines naïves, au format long (origin, unique_id, ds), une colonne par baseline.

    Elles n'utilisent que des valeurs STRICTEMENT antérieures à l'origine (assertion), y compris à 24 h
    d'horizon : « SN 24 h » à l'heure origine + k regarde origine + k − 24 < origine.
    """
    order = hier.order
    Y = to_wide(hier.Y_df, order)
    parts = []
    for origin in origins:
        ds = pd.date_range(origin, periods=h, freq="h")
        lag24, last = ds - pd.Timedelta(hours=24), origin - pd.Timedelta(hours=1)
        lags168 = [ds - pd.Timedelta(hours=168 * k) for k in range(1, 5)]
        if max(lag24.max(), last, *(lag.max() for lag in lags168)) >= origin:
            raise AssertionError(f"fuite : une baseline lit une valeur postérieure à l'origine {origin}")
        values = {"Naïf": np.repeat(Y.loc[[last]].to_numpy(), h, axis=0),
                  "SN 24 h": Y.loc[lag24].to_numpy(),
                  "Moy. 4 sem.": np.mean([Y.loc[lag].to_numpy() for lag in lags168], axis=0)}
        frame = pd.DataFrame({"origin": origin, "ds": np.tile(ds, len(order)),
                              "unique_id": np.repeat(order, h)})
        for name, v in values.items():
            frame[name] = v.T.ravel()                      # v : h × m, colonne par série -> série par série
        parts.append(frame)
    out = pd.concat(parts, ignore_index=True)
    if out[NAIVE_BASELINES].isna().any().any():
        raise ValueError("baseline naïve incomplète : historique manquant")
    return out


def rte_forecast(national: pd.DataFrame, origins: list[pd.Timestamp], total_id: str,
                 h: int = 24, debias_weeks: int = 52) -> pd.DataFrame:
    """La prévision J−1 publiée par RTE, pour le seul total France.

    Deux colonnes : « RTE J-1 » telle que publiée, et « RTE J-1 débiaisée » : on lui retire son erreur
    moyenne sur les `debias_weeks` semaines STRICTEMENT antérieures à l'origine. Raison : la prévision
    publiée sous-estime la consommation d'environ 1 GW en permanence (différence de périmètre probable,
    non vérifiée) ; la débiaiser sans fuite la compare à armes plus égales.
    """
    err = national["rte_j1"] - national["y"]
    parts = []
    for origin in origins:
        ds = pd.date_range(origin, periods=h, freq="h")
        v = national["rte_j1"].reindex(ds)
        if v.isna().any():
            raise ValueError(f"prévision RTE absente pour {int(v.isna().sum())} h à l'origine {origin}")
        past = err[(err.index >= origin - pd.Timedelta(weeks=debias_weeks)) & (err.index < origin)]
        parts.append(pd.DataFrame({"origin": origin, "unique_id": total_id, "ds": ds, "RTE J-1": v.to_numpy(),
                                   "RTE J-1 débiaisée": v.to_numpy() - past.mean()}))
    return pd.concat(parts, ignore_index=True)


def with_baselines(forecasts: pd.DataFrame, hier: Hierarchy, national: pd.DataFrame | None = None,
                   ) -> pd.DataFrame:
    """Ajoute les baselines naïves (toutes les séries) et, si `national` est fourni, RTE J−1 (France)."""
    origins = list(pd.to_datetime(sorted(forecasts["origin"].unique())))
    h = int(forecasts["horizon"].max())
    out = forecasts.merge(naive_baselines(hier, origins, h), on=["origin", "unique_id", "ds"], how="left")
    if national is not None:
        top = list(hier.tags.values())[0][0]
        out = out.merge(rte_forecast(national, origins, top, h), on=["origin", "unique_id", "ds"], how="left")
    return out


def by_level(forecasts: pd.DataFrame, scales: pd.DataFrame, hier: Hierarchy) -> pd.DataFrame:
    """`metrics.mase_by_level`, en retirant les méthodes absentes d'un niveau (RTE J−1 hors France)."""
    m = mase_long(forecasts, scales, hier).dropna(subset=["mase"])
    return mase_by_level(m)


def gain(by_lvl: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Gain relatif par (origine, niveau, méthode) : 1 − MASE / MASE(reference). Positif = mieux."""
    ref = by_lvl[by_lvl["method"] == reference].set_index(["origin", "level"])["mase"]
    out = by_lvl.join(ref.rename("ref"), on=["origin", "level"])
    out["gain"] = 1 - out["mase"] / out["ref"]
    return out.drop(columns="ref").dropna(subset=["gain"])


def aggregate_gain(by_lvl: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Gain agrégé par (niveau, méthode) : 1 − MASE moyen / MASE moyen(reference), sur les origines communes.

    Ajoute `wins` : part des origines où la méthode fait mieux que la référence.
    """
    g = gain(by_lvl, reference)
    out = g.groupby(["level", "method"]).agg(mase=("mase", "mean"), wins=("gain", lambda x: (x > 0).mean()),
                                             n=("origin", "nunique"))
    ref = g.join(by_lvl[by_lvl["method"] == reference].set_index(["origin", "level"])["mase"].rename("ref"),
                 on=["origin", "level"]).groupby(["level", "method"])["ref"].mean()
    out["gain"] = 1 - out["mase"] / ref
    return out.reset_index()


def gain_table(by_lvl: pd.DataFrame, references: list[str], order: list[str]) -> pd.DataFrame:
    """Gain agrégé de chaque méthode contre chaque référence, par niveau (méthodes en lignes)."""
    t = pd.concat([aggregate_gain(by_lvl, ref).assign(reference=ref) for ref in references])
    wide = t.pivot_table(index="method", columns=["level", "reference"], values="gain")
    return wide.reindex([m for m in order if m in wide.index])


def mase_by_horizon(forecasts: pd.DataFrame, scales: pd.DataFrame, hier: Hierarchy) -> pd.DataFrame:
    """MASE à chaque pas d'horizon (1 à h), moyenné sur les origines : (level, method, horizon, mase).

    Même échelle que le MASE global : la moyenne sur les h pas redonne exactement le MASE du tableau.
    """
    fixed = ["origin", "unique_id", "ds", "horizon", "y"]
    methods = [c for c in forecasts.columns if c not in fixed]
    scale = scales.set_index(["origin", "unique_id"])["scale"]
    s = scale.reindex(pd.MultiIndex.from_frame(forecasts[["origin", "unique_id"]])).to_numpy()
    err = forecasts[methods].sub(forecasts["y"], axis=0).abs().div(s, axis=0)
    long = pd.concat([forecasts[["origin", "unique_id", "horizon"]], err], axis=1).melt(
        id_vars=["origin", "unique_id", "horizon"], var_name="method", value_name="mase").dropna()
    long["level"] = long["unique_id"].map(level_of(hier))
    per_origin = long.groupby(["origin", "level", "method", "horizon"], as_index=False)["mase"].mean()
    return per_origin.groupby(["level", "method", "horizon"], as_index=False)["mase"].mean()


def gain_by_horizon(h_table: pd.DataFrame, reference: str) -> pd.DataFrame:
    """1 − MASE_h / MASE_h(reference), à chaque horizon. Positif = mieux que la référence."""
    w = h_table.pivot_table(index=["level", "horizon"], columns="method", values="mase")
    return (1 - w.div(w[reference], axis=0)).stack().rename("gain").reset_index()


def monthly(by_lvl: pd.DataFrame) -> pd.DataFrame:
    """MASE moyen par mois de l'origine : (level, method, month, mase, n)."""
    d = by_lvl.assign(month=pd.to_datetime(by_lvl["origin"]).dt.to_period("M").dt.to_timestamp())
    return d.groupby(["level", "method", "month"], as_index=False)["mase"].agg(mase="mean", n="count")


def convergence(by_lvl: pd.DataFrame, method: str, reference: str, level: str = "Régions",
                n_perm: int = 200, seed: int = 37) -> pd.DataFrame:
    """Gain agrégé calculé sur les k premières origines, quand on les ajoute une à une.

    Deux ordres : chronologique (ce qu'aurait vu un analyste au fil de l'année), et aléatoire (n_perm
    permutations : bande 5 %-95 % de ce qu'on peut conclure avec k origines tirées au hasard).
    Renvoie (k, chrono, p05, p50, p95). La dernière valeur est le gain agrégé de `aggregate_gain`.
    """
    w = by_lvl[by_lvl["level"] == level].pivot(index="origin", columns="method", values="mase")
    w = w[[method, reference]].dropna().sort_index().to_numpy()
    k = np.arange(1, len(w) + 1)

    def running(x):
        return 1 - np.cumsum(x[:, 0]) / np.cumsum(x[:, 1])

    rng = np.random.default_rng(seed)
    perm = np.array([running(w[rng.permutation(len(w))]) for _ in range(n_perm)])
    return pd.DataFrame({"k": k, "chrono": running(w), "p05": np.quantile(perm, 0.05, axis=0),
                         "p50": np.quantile(perm, 0.5, axis=0), "p95": np.quantile(perm, 0.95, axis=0)})


def error_metrics(forecasts: pd.DataFrame, hier: Hierarchy, scales: pd.DataFrame) -> pd.DataFrame:
    """Quatre métriques par (origine, niveau, méthode) : MASE, MAE (MW), RMSE (MW), MAPE (%).

    Le classement des méthodes doit tenir quelle que soit la métrique : sinon, la conclusion dépend d'un
    choix d'évaluation, pas de la méthode. Niveau « Régions » = moyenne des 12 séries, comme pour le MASE.
    """
    fixed = ["origin", "unique_id", "ds", "horizon", "y"]
    methods = [c for c in forecasts.columns if c not in fixed]
    e = forecasts[methods].sub(forecasts["y"], axis=0)
    keys = forecasts[["origin", "unique_id"]]
    per_series = {
        "MAE (MW)": pd.concat([keys, e.abs()], axis=1).groupby(["origin", "unique_id"]).mean(),
        "RMSE (MW)": np.sqrt(pd.concat([keys, e ** 2], axis=1).groupby(["origin", "unique_id"]).mean()),
        "MAPE (%)": 100 * pd.concat([keys, e.abs().div(forecasts["y"], axis=0)], axis=1)
        .groupby(["origin", "unique_id"]).mean(),
    }
    rows = []
    for name, t in per_series.items():
        long = t.reset_index().melt(id_vars=["origin", "unique_id"], var_name="method", value_name="value")
        long["metric"] = name
        rows.append(long)
    out = pd.concat(rows).dropna(subset=["value"])
    mase = mase_long(forecasts, scales, hier).rename(columns={"mase": "value"}).assign(metric="MASE")
    out["level"] = out["unique_id"].map(level_of(hier))
    out = pd.concat([out, mase.dropna(subset=["value"])], ignore_index=True)
    return out.groupby(["origin", "level", "metric", "method"], as_index=False)["value"].mean()


def gain_by_metric(metrics_long: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Gain agrégé par métrique : 1 − moyenne de la métrique / moyenne de la métrique de référence."""
    w = metrics_long.pivot_table(index=["level", "metric", "origin"], columns="method", values="value")
    w = w[w[reference].notna()]
    means = w.groupby(level=["level", "metric"]).mean()
    return (1 - means.div(means[reference], axis=0)).stack().rename("gain").reset_index()


def rank_probabilities(by_lvl: pd.DataFrame, methods: list[str], level: str = "Régions",
                       n_boot: int = 2000, seed: int = 37) -> pd.DataFrame:
    """Bootstrap sur les origines : probabilité que chaque méthode ait le meilleur MASE moyen, et rang moyen.

    On ré-échantillonne des ORIGINES (l'unité indépendante, cf. `inference`), avec remise.
    """
    w = by_lvl[by_lvl["level"] == level].pivot(index="origin", columns="method", values="mase")[methods]
    X = w.to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X), size=(n_boot, len(X)))
    means = X[idx].mean(axis=1)                                   # n_boot × méthodes
    # ex aequo : même rang (le plus petit). Au niveau Régions, BU et la base MSTL sont identiques.
    ranks = rankdata(means, method="min", axis=1)
    return pd.DataFrame({"P(meilleure)": (ranks == 1).mean(axis=0), "rang moyen": ranks.mean(axis=0),
                         "rang 5 %": np.quantile(ranks, 0.05, axis=0),
                         "rang 95 %": np.quantile(ranks, 0.95, axis=0)}, index=methods)


def study_summary(results: dict[str, pd.DataFrame], reference: str = "BU") -> pd.DataFrame:
    """Une ligne par (étude, niveau, méthode) : MASE moyen, gain agrégé vs `reference`, origines gagnées.

    results : nom de l'étude -> MASE par niveau (sortie de `by_level`).
    """
    return pd.concat([aggregate_gain(bl, reference).assign(study=study) for study, bl in results.items()],
                     ignore_index=True)


def paired_vs_reference(by_lvl: pd.DataFrame, methods: list[str], reference: str = "BU",
                        alpha: float = 0.05) -> pd.DataFrame:
    """Chaque méthode contre `reference`, par niveau : gain agrégé, origines gagnées, p-valeurs Holm.

    Même protocole que le notebook 04 : une différence de MASE par origine, test de Wilcoxon signé et
    t HAC, puis correction de Holm sur toutes les comparaisons de la table.
    """
    from scipy import stats

    from .inference import hac_mean_test, holm

    agg = aggregate_gain(by_lvl, reference).set_index(["level", "method"])
    rows = {}
    for level in ("Régions", "France"):
        w = by_lvl[by_lvl["level"] == level].pivot(index="origin", columns="method", values="mase")
        for m in methods:
            d = (w[m] - w[reference]).dropna().to_numpy()
            if not np.any(d):
                continue                                            # identique à la référence
            rows[(level, m)] = {"gain": agg.loc[(level, m), "gain"], "origines gagnées": (d < 0).mean(),
                                "p Wilcoxon": stats.wilcoxon(d).pvalue, "p HAC": hac_mean_test(d)["p"]}
    t = pd.DataFrame(rows).T
    for col in ("p Wilcoxon", "p HAC"):
        t[col + " (Holm)"] = holm(t[col], alpha)["p_holm"].to_numpy()
    return t.drop(columns=["p Wilcoxon", "p HAC"])


def mean_region_correlation(E: np.ndarray) -> float:
    """Corrélation moyenne entre régions (hors diagonale). E : temps × séries, le total en colonne 0."""
    R = np.corrcoef(E[:, 1:].T)
    return float(R[~np.eye(R.shape[0], dtype=bool)].mean())


def insample_residuals(hier: Hierarchy, origin: pd.Timestamp, spec) -> pd.DataFrame:
    """Résidus in-sample (y − ajusté) du modèle réconcilié de `spec`, sur la fenêtre d'entraînement.

    C'est exactement ce que voit `mint_shrink` pour estimer W à cette origine.
    """
    from statsforecast import StatsForecast

    from .backtest import split_at

    train, _ = split_at(hier.Y_df, origin, spec)
    model = spec.build_models()[list(spec.models).index(spec.reconcile_model)]
    sf = StatsForecast(models=[model], freq="h", n_jobs=spec.n_jobs)
    sf.forecast(df=train, h=1, fitted=True)
    fv = sf.forecast_fitted_values()
    col = [c for c in fv.columns if c not in ("unique_id", "ds", "y")][0]
    return (to_wide(fv, hier.order, "y") - to_wide(fv, hier.order, col)).dropna()


def oos_errors(forecasts: pd.DataFrame, order: list[str], model_col: str = "base") -> pd.DataFrame:
    """Erreurs hors échantillon y − prévision, (origine, ds) × séries, dans l'ordre de S."""
    e = forecasts.assign(e=forecasts["y"] - forecasts[model_col])
    return e.pivot_table(index=["origin", "ds"], columns="unique_id", values="e")[order]
