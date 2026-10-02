"""Bloc 5 : le backtest complet (MSTL + 5 réconciliations) sur une hiérarchie synthétique, sans réseau."""
import numpy as np
import pandas as pd
import pytest

from w37_reconciliation.eco2mix import metrics, prepare
from w37_reconciliation.eco2mix.backtest import BacktestSpec, reconcile_with_past_errors, run_backtest

pytestmark = pytest.mark.slow

RECONCILED = ["BU", "OLS", "WLS struct", "WLS var", "MinT shrink"]


@pytest.fixture(scope="module")
def backtest():
    rng = np.random.default_rng(5)
    ds = pd.date_range("2024-01-01", periods=24 * 7 * 8, freq="h")
    t = np.arange(len(ds))
    common = np.cumsum(rng.normal(0, 5, len(ds)))                     # choc national partagé
    rows = []
    for r in range(3):
        y = (1000 * (r + 1) + 150 * np.sin(2 * np.pi * t / 24) + 60 * np.sin(2 * np.pi * t / 168)
             + common + rng.normal(0, 15, len(ds)))
        rows.append(pd.DataFrame({"region": f"R{r}", "ds": ds, "y": y}))
    hier = prepare.build_eco2mix_hierarchy(pd.concat(rows, ignore_index=True))
    origins = list(pd.date_range("2024-02-05", periods=4, freq="3D"))
    out = run_backtest(hier, origins, BacktestSpec(train_weeks=4, n_jobs=1), log=lambda *_: None)
    return hier, origins, out


def test_every_reconciliation_is_coherent(backtest):
    hier, _, out = backtest
    table = metrics.assert_coherent(out["forecasts"], hier, RECONCILED, tol=1e-6)
    assert table.shape == (4, 5)
    assert metrics.incoherence(out["forecasts"], hier, "base").max() > 1e-3     # MSTL, lui, ne l'est pas


def test_outputs_have_the_expected_shape(backtest):
    hier, origins, out = backtest
    fc = out["forecasts"]
    assert len(fc) == len(origins) * len(hier.order) * 24
    assert fc["horizon"].between(1, 24).all()
    assert set(RECONCILED) | {"base", "SN", "y"} <= set(fc.columns)
    assert (out["diagnostics"]["lambda"].between(0, 1)).all()
    assert len(out["scales"]) == len(origins) * len(hier.order)


def test_mase_by_level_has_both_levels_for_every_method(backtest):
    hier, origins, out = backtest
    by_level = metrics.mase_by_level(metrics.mase_long(out["forecasts"], out["scales"], hier))
    assert set(by_level["level"]) == {"France", "Régions"}
    assert by_level.groupby("method")["origin"].nunique().eq(len(origins)).all()


def test_past_error_reconciliation_runs_on_real_backtest_output(backtest):
    hier, _, out = backtest
    rec = reconcile_with_past_errors(out["forecasts"], hier, min_past_origins=2)
    assert rec["origin"].nunique() == 2
    metrics.assert_coherent(rec, hier, ["MinT oos", "WLS var oos"], tol=1e-6)
