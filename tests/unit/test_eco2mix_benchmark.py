"""Bloc 5, benchmark : baselines naïves, RTE J−1, gains et études de confirmation (données synthétiques)."""
import numpy as np
import pandas as pd
import pytest
from test_eco2mix import regions_frame

from w37_reconciliation.eco2mix import benchmark as bm
from w37_reconciliation.eco2mix import metrics, prepare
from w37_reconciliation.eco2mix.backtest import BacktestSpec, origins_from_config, reconcile_with_past_errors


@pytest.fixture(scope="module")
def hier():
    return prepare.build_eco2mix_hierarchy(regions_frame(n_hours=24 * 7 * 6), "France")


ORIGINS = [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-04")]


# --- baselines naïves --------------------------------------------------------------------------------

def test_naive_baselines_read_the_right_past_values(hier):
    out = bm.naive_baselines(hier, ORIGINS, h=24)
    Y = hier.Y_df.set_index(["unique_id", "ds"])["y"]
    row = out[(out["unique_id"] == "France/R1") & (out["ds"] == ORIGINS[0] + pd.Timedelta(hours=5))].iloc[0]
    ds = row["ds"]
    assert row["Naïf"] == Y[("France/R1", ORIGINS[0] - pd.Timedelta(hours=1))]
    assert row["SN 24 h"] == Y[("France/R1", ds - pd.Timedelta(hours=24))]
    expected = np.mean([Y[("France/R1", ds - pd.Timedelta(hours=168 * k))] for k in range(1, 5)])
    assert row["Moy. 4 sem."] == pytest.approx(expected)
    assert len(out) == len(ORIGINS) * 24 * len(hier.order)


def test_naive_baselines_are_coherent_by_construction(hier):
    out = bm.naive_baselines(hier, ORIGINS, h=24)
    metrics.assert_coherent(out, hier, bm.NAIVE_BASELINES, tol=1e-6)


def test_naive_baselines_refuse_an_origin_without_four_weeks_of_history(hier):
    with pytest.raises(KeyError):
        bm.naive_baselines(hier, [pd.Timestamp("2024-01-10")], h=24)


def test_rte_debiasing_only_uses_errors_before_the_origin():
    ds = pd.date_range("2024-01-01", periods=24 * 20, freq="h")
    national = pd.DataFrame({"y": 100.0, "rte_j1": 90.0}, index=ds)     # biais constant de −10
    origin = pd.Timestamp("2024-01-15")
    national.loc[national.index >= origin, "rte_j1"] = 500.0             # le futur ne doit rien changer
    out = bm.rte_forecast(national, [origin], "France", h=24, debias_weeks=1)
    assert (out["RTE J-1"] == 500.0).all()
    np.testing.assert_allclose(out["RTE J-1 débiaisée"], 510.0)


# --- gains ---------------------------------------------------------------------------------------------

def by_level_toy():
    rows = []
    for i, o in enumerate(pd.date_range("2024-01-01", periods=4, freq="3D")):
        for level in ("France", "Régions"):
            rows += [{"origin": o, "level": level, "method": "BU", "mase": 1.0 + i},
                     {"origin": o, "level": level, "method": "A", "mase": 0.5 + i},
                     {"origin": o, "level": level, "method": "B", "mase": 1.0 + i}]   # ex aequo avec BU
    return pd.DataFrame(rows)


def test_aggregate_gain_is_the_ratio_of_mean_mase():
    g = bm.aggregate_gain(by_level_toy(), "BU").set_index(["level", "method"])
    assert g.loc[("Régions", "A"), "gain"] == pytest.approx(1 - 2.0 / 2.5)
    assert g.loc[("Régions", "A"), "wins"] == 1.0
    assert g.loc[("Régions", "BU"), "gain"] == 0.0


def test_convergence_ends_on_the_aggregate_gain():
    bl = by_level_toy()
    c = bm.convergence(bl, "A", "BU", n_perm=20)
    agg = bm.aggregate_gain(bl, "BU").set_index(["level", "method"]).loc[("Régions", "A"), "gain"]
    assert c["chrono"].iat[-1] == pytest.approx(agg)
    assert c["p05"].iat[-1] == pytest.approx(agg) == c["p95"].iat[-1]


def test_rank_probabilities_count_ties_as_shared_first_place():
    bl = by_level_toy()
    bl.loc[bl["method"] == "A", "mase"] += 10                            # A devient la pire
    r = bm.rank_probabilities(bl, ["BU", "A", "B"], n_boot=50)
    assert r.loc["BU", "P(meilleure)"] == 1.0 == r.loc["B", "P(meilleure)"]
    assert r.loc["A", "rang moyen"] == 3.0


def test_mase_by_horizon_averages_back_to_the_global_mase():
    from test_eco2mix import toy_forecasts

    fc, scales, hier = toy_forecasts()
    fc["horizon"] = fc.groupby(["origin", "unique_id"]).cumcount() + 1
    fc.loc[fc["horizon"] == 2, "base"] += 1.0                             # erreur différente par horizon
    h = bm.mase_by_horizon(fc, scales, hier)
    glob = metrics.mase_by_level(metrics.mase_long(fc, scales, hier))
    for level in ("France", "Régions"):
        hm = h[(h["level"] == level) & (h["method"] == "base")]["mase"].mean()
        gm = glob[(glob["level"] == level) & (glob["method"] == "base")]["mase"].mean()
        assert hm == pytest.approx(gm)


def test_error_metrics_give_four_metrics_per_level():
    from test_eco2mix import toy_forecasts

    fc, scales, hier = toy_forecasts()
    em = bm.error_metrics(fc, hier, scales)
    assert set(em["metric"]) == {"MASE", "MAE (MW)", "RMSE (MW)", "MAPE (%)"}
    mae = em[(em["metric"] == "MAE (MW)") & (em["method"] == "base")].set_index("level")["value"]
    assert mae["France"] == pytest.approx(6.0) and mae["Régions"] == pytest.approx(1.5)


# --- études de confirmation ------------------------------------------------------------------------------

CFG = {"backtest": {"h": 24, "season": 168, "train_weeks": 52, "models": ["SeasonalNaive", "MSTL"],
                    "reconcile_model": "MSTL", "methods": ["BottomUp", "mint_shrink"],
                    "main_origins": ["2024-11-05"],
                    "robustness": {"start": "2024-01-09", "end": "2024-01-15", "every_days": 3}},
       "confirmation": {"autre": {"start": "2023-01-10", "end": "2023-01-16", "every_days": 3,
                                  "models": ["SeasonalNaive", "AutoETS"], "reconcile_model": "AutoETS"}}}


def test_a_confirmation_study_changes_only_what_it_declares():
    main, other = BacktestSpec.from_config(CFG), BacktestSpec.from_config(CFG, "autre")
    assert other.reconcile_model == "AutoETS" and other.models == ("SeasonalNaive", "AutoETS")
    assert (other.h, other.train_weeks, other.methods) == (main.h, main.train_weeks, main.methods)
    assert len(origins_from_config(CFG, "autre")) == 3


def test_an_unknown_study_is_refused():
    with pytest.raises(ValueError, match="étude inconnue"):
        origins_from_config(CFG, "annee_2099")


def test_past_error_window_keeps_only_the_last_finished_origins():
    """Avec window=1, une erreur énorme sur l'avant-dernière origine passée ne compte plus."""
    rng = np.random.default_rng(1)
    hier = prepare.build_eco2mix_hierarchy(regions_frame(n_hours=48))
    rows = []
    for o in pd.date_range("2024-01-01", periods=5, freq="2D"):
        for hh in range(24):
            leaves = 100 + rng.normal(0, 5, 3)
            base = leaves + rng.normal(0, 2, 3)
            vals = {"France": (leaves.sum(), base.sum() + 1.0)}
            vals.update({u: (leaves[i], base[i]) for i, u in enumerate(hier.order[1:])})
            for u, (y, b) in vals.items():
                rows.append({"origin": o, "unique_id": u, "ds": o + pd.Timedelta(hours=hh), "horizon": hh + 1,
                             "y": y, "base": b})
    fc = pd.DataFrame(rows)
    poisoned = fc.copy()
    third = poisoned["origin"] == sorted(fc["origin"].unique())[2]
    poisoned.loc[third, "y"] += 1e3 * rng.normal(size=third.sum())
    last = fc["origin"].max()
    a = reconcile_with_past_errors(fc, hier, min_past_origins=2, window=1)
    b = reconcile_with_past_errors(poisoned, hier, min_past_origins=2, window=1)
    np.testing.assert_allclose(a[a["origin"] == last]["MinT oos"], b[b["origin"] == last]["MinT oos"])
    c = reconcile_with_past_errors(poisoned, hier, min_past_origins=2)          # sans fenêtre : elle compte
    assert not np.allclose(a[a["origin"] == last]["MinT oos"], c[c["origin"] == last]["MinT oos"])
