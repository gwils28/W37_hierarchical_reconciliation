"""Tests d'hypothèses : ce que MinT suppose, et si une méthode bat vraiment une autre.

Deux pièges guident tout le module :

1. **L'autocorrélation.** Des résidus horaires sont fortement autocorrélés. Un test t classique sur leur
   moyenne compte 8 760 observations indépendantes alors qu'il y en a beaucoup moins : il rejette
   presque toujours. On utilise des erreurs-types HAC (Newey-West), robustes à l'autocorrélation.
2. **L'unité d'analyse.** Pour comparer deux méthodes, l'unité indépendante n'est pas l'heure, c'est
   l'origine : les 24 erreurs d'une même origine viennent du même modèle entraîné sur les mêmes données.
   On compare donc une valeur par origine (test apparié). Diebold-Mariano reste disponible pour comparer
   deux suites d'erreurs à h pas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests


def hac_lags(n: int) -> int:
    """Nombre de retards de la variance HAC : ⌈1,3 √n⌉ (Lazarus, Lewis, Stock et Watson, 2018).

    La règle classique de Newey-West (1994), ⌊4 (n/100)^(2/9)⌋, donne 7 retards pour n = 2 000 : sur un
    bruit AR(1) de coefficient 0,9, le test rejette alors à tort 27 % du temps au lieu de 5 %. Avec ⌈1,3 √n⌉,
    on tombe à environ 9 % (vérifié par simulation, cf. tests/unit/test_eco2mix.py). Des résidus horaires
    sont dans ce régime de forte persistance.
    """
    return int(np.ceil(1.3 * np.sqrt(n)))


def hac_mean_test(x: np.ndarray, lags: int | None = None) -> dict:
    """H0 : E[x] = 0, avec une erreur-type HAC. Sert au test de biais des résidus."""
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("valeurs manquantes ou infinies : aligner les séries avant de tester")
    lags = hac_lags(len(x)) if lags is None else lags
    fit = sm.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    naive_se = x.std(ddof=1) / np.sqrt(len(x))
    return {"mean": float(fit.params[0]), "se_hac": float(fit.bse[0]), "se_naive": float(naive_se),
            "t": float(fit.tvalues[0]), "p": float(fit.pvalues[0]), "lags": lags}


def bias_table(E: pd.DataFrame, lags: int | None = None) -> pd.DataFrame:
    """Test de biais HAC, une ligne par série (colonnes de E). Le t naïf est montré pour comparaison."""
    rows = {}
    for col in E.columns:
        r = hac_mean_test(E[col].to_numpy(), lags)
        rows[col] = {**r, "t_naif": r["mean"] / r["se_naive"]}
    return pd.DataFrame(rows).T


def ljung_box(x: np.ndarray, lags: tuple[int, ...] = (24, 168)) -> pd.DataFrame:
    """H0 : pas d'autocorrélation jusqu'au retard k. Rejet = le modèle laisse de la structure."""
    return acorr_ljungbox(np.asarray(x, dtype=float), lags=list(lags))


def holm(pvalues: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    reject, p_adj, _, _ = multipletests(pvalues.to_numpy(), alpha=alpha, method="holm")
    return pd.DataFrame({"p": pvalues, "p_holm": p_adj, "rejet": reject}, index=pvalues.index)


def paired_comparison(a: pd.Series, b: pd.Series) -> dict:
    """a, b : une valeur de MASE par origine (même index). H0 : pas de différence entre méthodes.

    Deux tests complémentaires : t apparié (sur les différences) et Wilcoxon signé (sans hypothèse
    de normalité). Avec 4 origines, aucun des deux n'a de puissance : on le dit, on ne conclut pas.
    """
    d = (a - b).dropna()
    t = stats.ttest_1samp(d, 0.0)
    ci = (stats.t.interval(0.95, df=len(d) - 1, loc=d.mean(), scale=stats.sem(d)) if len(d) > 1
          else (np.nan, np.nan))
    w = stats.wilcoxon(d) if len(d) >= 6 and (d != 0).any() else None
    return {"n": int(len(d)), "diff_mean": float(d.mean()),
            "ci95_low": float(ci[0]), "ci95_high": float(ci[1]),
            "t_p": float(t.pvalue), "wilcoxon_p": float(w.pvalue) if w else np.nan,
            "share_a_better": float((d < 0).mean())}


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int, power: int = 1) -> dict:
    """Test de Diebold-Mariano avec la correction de Harvey, Leybourne et Newbold (1997).

    e1, e2 : erreurs de prévision alignées. d = |e1|^p − |e2|^p. H0 : E[d] = 0.
    La variance de long terme est estimée avec h − 1 retards (erreurs à h pas : MA(h−1) sous H0).
    """
    d = np.abs(np.asarray(e1)) ** power - np.abs(np.asarray(e2)) ** power
    n = len(d)
    dc = d - d.mean()
    gamma = [np.dot(dc[k:], dc[:n - k]) / n for k in range(h)]
    lrv = gamma[0] + 2 * sum(gamma[1:])
    if lrv <= 0:
        return {"dm": np.nan, "p": np.nan, "mean_d": float(d.mean())}
    dm = d.mean() / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {"dm": float(stat), "p": float(p), "mean_d": float(d.mean())}
