"""Un run de l'option A : hash_check → choose_method → réconciliation → quality gate → artefact immuable."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from ..data import Hierarchy, to_wide, train_test_split
from ..forecasting import residual_matrix
from .hierarchy import hash_check
from .method_selection import choose_method
from .quality_gate import GateReport, mase, quality_gate


@dataclass
class JobResult:
    run_id: str
    method: str
    lineage: dict
    report: GateReport
    published_to: Path | None


def run_job(hier: Hierarchy, cfg: dict, output_dir: str | Path, source: str,
            levels: dict[str, str], h: int = 8, season: int = 4, freq: str = "QS") -> JobResult:
    """levels : nom de niveau (tel que dans level_weights) -> clé de `hier.tags`.

    Backtest à une origine : les h dernières dates servent à mesurer le MASE du gate.
    """
    output_dir = Path(output_dir)
    s_hash = hash_check(hier.S_df, source, output_dir / cfg["hierarchy"]["hash_registry"])

    # base forecasts + résidus in-sample dans une fenêtre glissante strictement antérieure
    train, test = train_test_split(hier.Y_df, h)
    sf = StatsForecast(models=[AutoETS(season_length=season)], freq=freq, n_jobs=1)
    Y_hat = sf.forecast(df=train, h=h, fitted=True)
    Y_fit = sf.forecast_fitted_values()
    window = Y_fit["ds"].sort_values().unique()[-cfg["reconciler"]["covariance_window"]:]
    Y_fit = Y_fit[Y_fit["ds"].isin(window)]
    if Y_fit["ds"].max() >= test["ds"].min():
        raise RuntimeError("fuite : la fenêtre de covariance chevauche le test")

    order = hier.order
    method, why = choose_method(residual_matrix(Y_fit, order), cfg)

    hrec = HierarchicalReconciliation(reconcilers=[BottomUp(), MinTrace(method=method)])
    Y_rec = hrec.reconcile(Y_hat_df=Y_hat, Y_df=Y_fit, S_df=hier.S_df, tags=hier.tags)
    col_rec = next(c for c in Y_rec.columns if c.startswith("AutoETS/MinTrace"))

    y_true, y_train = to_wide(test, order).to_numpy(), to_wide(train, order).to_numpy()
    y_rec = to_wide(Y_rec, order, col_rec).to_numpy()
    y_bu = to_wide(Y_rec, order, "AutoETS/BottomUp").to_numpy()
    level_idx = {name: [order.index(u) for u in hier.tags[tag]] for name, tag in levels.items()}
    report = quality_gate(y_rec, hier.S, level_idx, cfg,
                          mase_rec=mase(y_true, y_rec, y_train, season),
                          mase_baseline=mase(y_true, y_bu, y_train, season))

    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    lineage = {"run_id": run_id, "s_hash": s_hash, "source": source, "method_used": method,
               "method_reason": why, "w_version": f"window={len(window)}:{pd.Timestamp(window[-1]).date()}",
               "lib_versions": {p: version(p) for p in ("hierarchicalforecast", "statsforecast", "numpy")}}

    published_to = None
    if report.passed:
        published_to = output_dir / cfg["outputs"]["table"] / f"run_id={run_id}"
        published_to.mkdir(parents=True)
        out = Y_rec[["unique_id", "ds", col_rec]].rename(columns={col_rec: "y_reconciled"})
        out = out.assign(forecast_date=str(test["ds"].min().date()))
        out.to_csv(published_to / "part-0.csv", index=False)
        (published_to / "_lineage.json").write_text(json.dumps(lineage, indent=2, default=float))
        (published_to / "_gate.json").write_text(json.dumps(report.__dict__, indent=2, default=float))
    return JobResult(run_id, method, lineage, report, published_to)
