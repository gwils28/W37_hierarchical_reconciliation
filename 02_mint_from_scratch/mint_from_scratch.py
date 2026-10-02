"""Bloc 2 W37 - MinT(shrink) reimplemente de zero en NumPy, compare a HierarchicalForecast.

Hierarchie synthetique region x canal (2 x 2 -> 6 series : 2 regions + 4 feuilles).
Execution : .venv/bin/python 02_mint_from_scratch/mint_from_scratch.py
"""
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive
from hierarchicalforecast.utils import aggregate
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace


def schafer_strimmer_lambda(E: np.ndarray) -> float:
    """Intensite de shrinkage optimale vers la diagonale (Schafer & Strimmer 2005, cible "D").

    On retrecit les correlations hors diagonale vers 0, variances inchangees :
        lam* = sum_{i!=j} Var(r_ij) / sum_{i!=j} r_ij^2
    avec, sur les residus standardises x_ki et w_kij = x_ki * x_kj :
        Var(r_ij) = T / (T-1)^3 * sum_k (w_kij - mean_k w_kij)^2
    Residus one-step supposes centres : pas de recentrage (coherent avec W1 = E'E/T).
    """
    T = E.shape[0]
    X = E / np.sqrt((E**2).mean(axis=0))          # residus standardises
    Wk = X[:, :, None] * X[:, None, :]             # (T, m, m) produits croises par instant
    r = Wk.mean(axis=0)                            # correlations empiriques
    var_r = T / (T - 1) ** 3 * ((Wk - r) ** 2).sum(axis=0)
    off = ~np.eye(E.shape[1], dtype=bool)
    return float(np.clip(var_r[off].sum() / (r[off] ** 2).sum(), 0.0, 1.0))


def mint_projection(Sm: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """G = (S' W^-1 S)^-1 S' W^-1 et P = S G (projection oblique sur l'espace coherent)."""
    Wi = np.linalg.pinv(W)
    G = np.linalg.solve(Sm.T @ Wi @ Sm, Sm.T @ Wi)
    return G, Sm @ G


# --- 1. hierarchie synthetique : 2 regions x 2 canaux -> 6 series (2 agregats + 4 feuilles)
rng = np.random.default_rng(0)
n, h = 80, 8
dates = pd.date_range("2015-01-01", periods=n, freq="QS")
rows = []
for region in ["Bretagne", "PaysLoire"]:
    for canal in ["GMS", "RHD"]:
        base = 100 + 30 * (region == "Bretagne") + 20 * (canal == "GMS")
        y = base + 10 * np.sin(2 * np.pi * np.arange(n) / 4) + np.cumsum(rng.normal(0, 1.5, n))
        rows.append(pd.DataFrame({"ds": dates, "Region": region, "Canal": canal, "y": y}))
df = pd.concat(rows)

Y_df, S_df, tags = aggregate(df=df, spec=[["Region"], ["Region", "Canal"]])
train, test = Y_df[Y_df.ds < dates[-h]], Y_df[Y_df.ds >= dates[-h]]

# --- 2. previsions de base + valeurs ajustees IN-SAMPLE (fitted=True : indispensable)
sf = StatsForecast(models=[AutoETS(season_length=4), SeasonalNaive(season_length=4)],
                   freq="QS", n_jobs=1)
Y_hat = sf.forecast(df=train, h=h, fitted=True)
Y_fit = sf.forecast_fitted_values()

# Garde-fou anti-fuite n°1 : W ne doit voir QUE des residus du train.
assert Y_fit.ds.max() < test.ds.min(), "fuite : les valeurs ajustees debordent sur le test"

# --- 3. reference bibliotheque
hrec = HierarchicalReconciliation(
    reconcilers=[BottomUp(), MinTrace(method="ols"), MinTrace(method="mint_shrink")])
Y_rec = hrec.reconcile(Y_hat_df=Y_hat, Y_df=Y_fit, S_df=S_df, tags=tags)

# --- 4. implementation maison
S = S_df.set_index("unique_id"); order = S.index.tolist(); Sm = S.values   # (m x nb)
E = (Y_fit.pivot(index="ds", columns="unique_id", values="y")
     - Y_fit.pivot(index="ds", columns="unique_id", values="AutoETS"))[order].dropna().values
T, m = E.shape

W1 = E.T @ E / T                                  # covariance empirique des residus 1-pas
lam = schafer_strimmer_lambda(E)
W = lam * np.diag(np.diag(W1)) + (1 - lam) * W1   # shrinkage vers la diagonale
G, P = mint_projection(Sm, W)

yhat = Y_hat.pivot(index="ds", columns="unique_id", values="AutoETS")[order].values
mine = yhat @ P.T
lib = Y_rec.pivot(index="ds", columns="unique_id",
                  values="AutoETS/MinTrace_method-mint_shrink")[order].values

# --- 5. les trois assertions qui font foi
n_agg = m - Sm.shape[1]
err_sgs = np.abs(P @ Sm - Sm).max()
err_coh = np.abs(mine[:, :n_agg] - mine[:, n_agg:] @ Sm[:n_agg, :].T).max()
err_rel = np.abs(mine - lib).max() / np.abs(lib).mean()
err_abs = np.abs(mine - lib).max()

print(f"series (m) = {m}, feuilles (nb) = {Sm.shape[1]}, residus (T) = {T}")
print(f"ordre      : {order}")
print(f"lambda SS  : {lam:.6f}")
print(f"P^2 == P   : {np.abs(P @ P - P).max():.3e}")
print(f"SGS == S   : {err_sgs:.3e}")
print(f"incoherence: {err_coh:.3e}")
print(f"ecart rel. : {err_rel:.3e}   (abs. {err_abs:.3e} sur des valeurs ~{np.abs(lib).mean():.0f})")

assert err_sgs < 1e-12, "SGS != S : la projection ne preserve pas l'absence de biais"
assert err_coh < 1e-9, "previsions reconciliees incoherentes"
assert err_rel < 1e-3, "ecart a la bibliotheque trop grand"
assert err_rel > 0, "ecart nul : implementation identique a la bibliotheque ?"
print("OK : 3 assertions vertes")

# --- 6. d'ou vient l'ecart residuel ? decomposition en deux sources
# (a) la covariance de base : E'E/T (non centree, squelette du plan) vs np.cov (centree, ddof=1)
# (b) l'estimateur de lambda lui-meme
def gap_to_lib(W_base, lam_):
    W_ = lam_ * np.diag(np.diag(W_base)) + (1 - lam_) * W_base
    return np.abs(yhat @ mint_projection(Sm, W_)[1].T - lib).max()

Ec = E - E.mean(axis=0)
lam_c = schafer_strimmer_lambda(Ec)               # meme formule, residus centres
W_cov = np.cov(E.T)                               # centree, ddof=1
print("decomposition de l'ecart absolu a la lib :")
print(f"  base E'E/T   + lambda non centre ({lam:.6f}) : {gap_to_lib(W1, lam):.3e}")
print(f"  base np.cov  + lambda non centre ({lam:.6f}) : {gap_to_lib(W_cov, lam):.3e}")
print(f"  base np.cov  + lambda centre     ({lam_c:.6f}) : {gap_to_lib(W_cov, lam_c):.3e}")

# --- 7. piege n°3 : m > T -> W1 singuliere, le shrinkage la rend inversible
E_short = E[: m - 2]                              # T' = m-2 < m observations
W1s = E_short.T @ E_short / len(E_short)
lam_s = schafer_strimmer_lambda(E_short)
Ws = lam_s * np.diag(np.diag(W1s)) + (1 - lam_s) * W1s
print(f"m > T : T'={len(E_short)}, rang(W1)={np.linalg.matrix_rank(W1s)}/{m}, "
      f"cond(W1)={np.linalg.cond(W1s):.1e} -> lambda={lam_s:.3f}, cond(W_shrink)={np.linalg.cond(Ws):.1e}")
