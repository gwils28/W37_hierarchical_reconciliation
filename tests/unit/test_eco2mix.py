"""Bloc 5 : préparation, protocole de backtest, métriques et tests statistiques, sur données synthétiques."""
import numpy as np
import pandas as pd
import pytest

from w37_reconciliation.eco2mix import inference, metrics, prepare
from w37_reconciliation.eco2mix.backtest import BacktestSpec, mase_scale, split_at


def half_hourly(start="2024-10-26 20:00", periods=16, value=100.0):
    idx = pd.date_range(start, periods=periods, freq="30min")
    return pd.Series(value + np.arange(periods, dtype=float), index=idx)


# --- préparation -------------------------------------------------------------------------------------

def test_october_dst_gap_is_reproduced_and_imputed():
    """Le défaut de la source : l'heure 00:00 UTC du dernier dimanche d'octobre manque (2 demi-heures)."""
    y = half_hourly()
    y = y.drop(pd.to_datetime(["2024-10-27 00:00", "2024-10-27 00:30"]))
    y30, counts = prepare.clean_half_hourly(y)
    assert counts["missing_halfhours"] == 2
    yh, imputed, c2 = prepare.to_hourly(y30)
    assert c2["imputed_hours"] == 1 and imputed[pd.Timestamp("2024-10-27 00:00")]
    before, after = yh[pd.Timestamp("2024-10-26 23:00")], yh[pd.Timestamp("2024-10-27 01:00")]
    assert yh[pd.Timestamp("2024-10-27 00:00")] == pytest.approx((before + after) / 2)


def test_march_dst_duplicates_are_counted_and_removed():
    y = half_hourly("2024-03-30 23:00", 8)
    dup = y.iloc[[4]]                                         # même instant, même valeur
    conflict = pd.Series([999.0], index=[y.index[5]])         # même instant, valeur différente
    y30, counts = prepare.clean_half_hourly(pd.concat([y, dup, conflict]))
    assert counts["duplicates_identical"] == 1 and counts["duplicates_conflicting"] == 1
    assert not y30.index.duplicated().any()
    assert y30.iloc[5] == pytest.approx((y.iloc[5] + 999.0) / 2)


def test_hourly_resampling_is_a_mean_not_a_sum():
    """Des MW restent des MW : 100 et 200 MW sur deux demi-heures font 150 MW sur l'heure."""
    y = pd.Series([100.0, 200.0], index=pd.date_range("2024-01-01", periods=2, freq="30min"))
    yh, _, _ = prepare.to_hourly(y)
    assert yh.iloc[0] == 150.0


def test_long_gaps_are_refused_not_interpolated():
    y = half_hourly(periods=20)
    y.iloc[4:10] = np.nan                                     # 3 heures manquantes
    with pytest.raises(ValueError, match="trou de 3 h"):
        prepare.to_hourly(y, max_gap_hours=1)


def regions_frame(n_hours=24 * 7 * 3, n_regions=3, seed=0):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    t = np.arange(n_hours)
    rows = []
    for r in range(n_regions):
        y = 1000 * (r + 1) + 100 * np.sin(2 * np.pi * t / 24) + 50 * np.sin(2 * np.pi * t / 168)
        rows.append(pd.DataFrame({"region": f"R{r}", "ds": ds, "y": y + rng.normal(0, 10, n_hours)}))
    return pd.concat(rows, ignore_index=True)


def test_hierarchy_total_is_the_exact_sum_of_regions():
    hier = prepare.build_eco2mix_hierarchy(regions_frame(), "France")
    wide = hier.Y_df.pivot(index="ds", columns="unique_id", values="y")
    leaves = [u for u in hier.order if u != "France"]
    np.testing.assert_allclose(wide["France"], wide[leaves].sum(axis=1))
    assert hier.order[0] == "France" and hier.S.shape == (4, 3)


def test_a_nan_in_a_leaf_is_refused_by_aggregate():
    df = regions_frame()
    df.loc[5, "y"] = np.nan
    with pytest.raises(ValueError, match="null values"):
        prepare.build_eco2mix_hierarchy(df)


def test_a_missing_row_silently_breaks_the_total_and_is_caught():
    """Le vrai piège : une ligne ABSENTE ne lève rien dans aggregate, mais le total de l'heure est faux."""
    df = regions_frame().drop(index=5)
    hier = prepare.build_eco2mix_hierarchy(df)
    wide = hier.Y_df.pivot(index="ds", columns="unique_id", values="y")
    assert wide["France"].iloc[5] < wide["France"].iloc[4] - 500          # une région manque à la somme
    with pytest.raises(ValueError, match="longueurs ou de bornes"):
        prepare.assert_no_missing(hier.Y_df)


# --- protocole de backtest ---------------------------------------------------------------------------

def test_split_is_strictly_before_origin_and_exactly_h_long():
    hier = prepare.build_eco2mix_hierarchy(regions_frame())
    spec = BacktestSpec(h=24, train_weeks=2)
    origin = pd.Timestamp("2024-01-15")
    train, test = split_at(hier.Y_df, origin, spec)
    assert train["ds"].max() < origin <= test["ds"].min()
    assert test["ds"].nunique() == 24 and train["ds"].nunique() == 2 * 168


def test_split_refuses_an_origin_without_enough_history():
    hier = prepare.build_eco2mix_hierarchy(regions_frame())
    with pytest.raises(ValueError, match="historique insuffisant"):
        split_at(hier.Y_df, pd.Timestamp("2024-01-10"), BacktestSpec(h=24, train_weeks=2))


def test_mase_scale_is_the_in_sample_seasonal_naive_error():
    ds = pd.date_range("2024-01-01", periods=4, freq="h")
    train = pd.DataFrame({"unique_id": "a", "ds": ds, "y": [1.0, 3.0, 2.0, 7.0]})
    assert mase_scale(train, ["a"], season=2).iloc[0] == pytest.approx((1 + 4) / 2)


# --- métriques -----------------------------------------------------------------------------------------

def toy_forecasts():
    ds = pd.date_range("2024-01-01", periods=2, freq="h")
    rows = []
    for uid, y, good, bad in [("France", 30, 30, 36), ("France/A", 10, 10, 11), ("France/B", 20, 20, 22)]:
        for d in ds:
            rows.append({"origin": ds[0], "unique_id": uid, "ds": d, "horizon": 1, "y": y,
                         "BU": good, "base": bad})
    fc = pd.DataFrame(rows)
    scales = pd.DataFrame({"origin": ds[0], "unique_id": ["France", "France/A", "France/B"],
                           "scale": [2.0, 1.0, 1.0]})
    hier = prepare.build_eco2mix_hierarchy(pd.DataFrame(
        {"region": ["A", "B"] * 2, "ds": np.repeat(ds, 2), "y": [10.0, 20.0] * 2}))
    return fc, scales, hier


def test_mase_is_reported_per_level_never_globally():
    fc, scales, hier = toy_forecasts()
    by_level = metrics.mase_by_level(metrics.mase_long(fc, scales, hier))
    base = by_level[by_level["method"] == "base"].set_index("level")["mase"]
    assert base["France"] == pytest.approx(6 / 2)
    assert base["Régions"] == pytest.approx((1 + 2) / 2)


def test_mase_ignores_scales_of_origins_without_forecasts():
    fc, scales, hier = toy_forecasts()
    extra = scales.assign(origin=scales["origin"] + pd.Timedelta(days=1))
    out = metrics.mase_long(fc, pd.concat([scales, extra]), hier)
    assert out["origin"].nunique() == 1 and out["mase"].notna().all()


def test_incoherence_detects_a_non_summing_forecast():
    fc, _, hier = toy_forecasts()
    assert metrics.incoherence(fc, hier, "BU").max() == 0
    assert metrics.incoherence(fc, hier, "base").max() == pytest.approx(36 - 33)
    with pytest.raises(AssertionError, match="incohérence"):
        metrics.assert_coherent(fc, hier, ["base"], tol=1e-6)


# --- tests statistiques ----------------------------------------------------------------------------------

def ar1(n, phi, seed=0, mean=0.0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return x + mean


def test_naive_t_overrejects_on_autocorrelated_noise_but_hac_does_not():
    """Sur 200 bruits AR(1) de moyenne nulle, le t naïf rejette bien plus que 5 % ; le HAC reste proche."""
    naive = hac = 0
    for seed in range(200):
        r = inference.hac_mean_test(ar1(2_000, 0.9, seed))
        naive += abs(r["mean"] / r["se_naive"]) > 1.96
        hac += r["p"] < 0.05
    assert naive / 200 > 0.4
    assert hac / 200 < 0.15                                  # ≈ 0,09 avec ⌈1,3 √n⌉ retards


def test_hac_detects_a_real_bias():
    assert inference.hac_mean_test(ar1(5_000, 0.5, mean=1.0))["p"] < 1e-6


def test_diebold_mariano_separates_good_from_bad_and_not_equals():
    rng = np.random.default_rng(1)
    e_good, e_bad = rng.normal(0, 1, 500), rng.normal(0, 2, 500)
    assert inference.diebold_mariano(e_good, e_bad, h=1)["p"] < 1e-6
    same = inference.diebold_mariano(rng.normal(0, 1, 500), rng.normal(0, 1, 500), h=1)["p"]
    assert same > 0.01


def test_paired_comparison_has_no_power_with_four_origins():
    a = pd.Series([1.00, 1.02, 0.98, 1.01])
    r = inference.paired_comparison(a, a + 0.01)
    assert r["n"] == 4 and np.isnan(r["wilcoxon_p"])


def test_past_error_reconciliation_only_uses_finished_origins():
    """Une erreur énorme injectée à une origine ne doit pas influencer la réconciliation de cette origine."""
    from w37_reconciliation.eco2mix.backtest import reconcile_with_past_errors

    rng = np.random.default_rng(0)
    hier = prepare.build_eco2mix_hierarchy(regions_frame(n_hours=48))
    rows = []
    for o in pd.date_range("2024-01-01", periods=6, freq="2D"):
        for hh in range(24):
            ds = o + pd.Timedelta(hours=hh)
            leaves = 100 + rng.normal(0, 5, 3)
            base = leaves + rng.normal(0, 2, 3)
            vals = {"France": (leaves.sum(), base.sum() + 1.0)}
            vals.update({u: (leaves[i], base[i]) for i, u in enumerate(hier.order[1:])})
            for u, (y, b) in vals.items():
                rows.append({"origin": o, "unique_id": u, "ds": ds, "horizon": hh + 1, "y": y, "base": b})
    fc = pd.DataFrame(rows)
    out1 = reconcile_with_past_errors(fc, hier, min_past_origins=3)
    poisoned = fc.copy()
    last = poisoned["origin"] == poisoned["origin"].max()
    poisoned.loc[last, "y"] += 1e4 * rng.normal(size=last.sum())   # le futur de la dernière origine change
    out2 = reconcile_with_past_errors(poisoned, hier, min_past_origins=3)
    a = out1[out1["origin"] == fc["origin"].max()]["MinT oos"].to_numpy()
    b = out2[out2["origin"] == fc["origin"].max()]["MinT oos"].to_numpy()
    np.testing.assert_allclose(a, b)
    assert out1["origin"].nunique() == 3
    gap = out1.pivot_table(index=["origin", "ds"], columns="unique_id", values="MinT oos")
    np.testing.assert_allclose(gap["France"], gap[hier.order[1:]].sum(axis=1), atol=1e-8)
