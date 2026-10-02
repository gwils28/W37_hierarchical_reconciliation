"""Bloc 3 W37 - un run complet de l'option A (batch materialise) sur la hierarchie du bloc 2.

hash_check -> choose_method -> reconcile -> quality_gate -> artefact immuable (ou blocage).
Execution : .venv/bin/python 03_system_design/demo_job.py
"""
import json
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS
from hierarchicalforecast.utils import aggregate
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace

from reconciliation_job import choose_method, hash_check, load_contract, mase, quality_gate

HERE = Path(__file__).parent
OUT = HERE / "outputs"
cfg = load_contract(HERE / "reconciliation.yaml")
# La hierarchie de demo (region x canal) n'a pas de niveau famille/national : on adapte les poids.
cfg["quality_gate"]["level_weights"] = {"feuille": 0.7, "region": 0.3}

# --- donnees : memes que le bloc 2
rng = np.random.default_rng(0)
n, h, season = 80, 8, 4
dates = pd.date_range("2015-01-01", periods=n, freq="QS")
rows = []
for region in ["Bretagne", "PaysLoire"]:
    for canal in ["GMS", "RHD"]:
        base = 100 + 30 * (region == "Bretagne") + 20 * (canal == "GMS")
        y = base + 10 * np.sin(2 * np.pi * np.arange(n) / 4) + np.cumsum(rng.normal(0, 1.5, n))
        rows.append(pd.DataFrame({"ds": dates, "Region": region, "Canal": canal, "y": y}))
Y_df, S_df, tags = aggregate(df=pd.concat(rows), spec=[["Region"], ["Region", "Canal"]])
train, test = Y_df[Y_df.ds < dates[-h]], Y_df[Y_df.ds >= dates[-h]]

# --- 1. hash de S (refus si la source a change sans bump)
OUT.mkdir(exist_ok=True)
source = "demo_region_canal@v1"
sh = hash_check(S_df, source, OUT / cfg["hierarchy"]["hash_registry"])

# --- 2. base forecasts + residus in-sample (fenetre strictement anterieure)
sf = StatsForecast(models=[AutoETS(season_length=season)], freq="QS", n_jobs=1)
Y_hat = sf.forecast(df=train, h=h, fitted=True)
Y_fit = sf.forecast_fitted_values()
window = Y_fit.ds.sort_values().unique()[-cfg["reconciler"]["covariance_window"]:]
Y_fit = Y_fit[Y_fit.ds.isin(window)]
assert Y_fit.ds.max() < test.ds.min()

order = S_df.unique_id.tolist()
wide = lambda d, col: d.pivot(index="ds", columns="unique_id", values=col)[order]  # noqa: E731
E = (wide(Y_fit, "y") - wide(Y_fit, "AutoETS")).dropna().values
method, why = choose_method(E, cfg)

# --- 3. reconciliation : methode retenue + baseline bottom-up
rec = MinTrace(method=method)
hrec = HierarchicalReconciliation(reconcilers=[BottomUp(), rec])
Y_rec = hrec.reconcile(Y_hat_df=Y_hat, Y_df=Y_fit, S_df=S_df, tags=tags)
col_rec = [c for c in Y_rec.columns if c.startswith("AutoETS/MinTrace")][0]

# --- 4. quality gate (MASE par niveau vs bottom-up, sur le test : backtest d'une origine)
S = S_df.set_index("unique_id").values
y_true, y_train = wide(test, "y").values, wide(train, "y").values
y_rec, y_bu = wide(Y_rec, col_rec).values, wide(Y_rec, "AutoETS/BottomUp").values
levels = {"region": [order.index(u) for u in tags["Region"]],
          "feuille": [order.index(u) for u in tags["Region/Canal"]]}
report = quality_gate(y_rec, S, levels, cfg,
                      mase_rec=mase(y_true, y_rec, y_train, season),
                      mase_baseline=mase(y_true, y_bu, y_train, season))

# --- 5. publication immuable, ou blocage
run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
lineage = {"run_id": run_id, "s_hash": sh, "source": source, "method_used": method,
           "method_reason": why, "w_version": f"window={len(window)}:{pd.Timestamp(window[-1]).date()}",
           "lib_versions": {p: version(p) for p in ["hierarchicalforecast", "statsforecast", "numpy"]}}
print(json.dumps({"lineage": lineage, "gate": report.__dict__}, indent=2, default=float))

if report.passed:
    part = OUT / cfg["outputs"]["table"] / f"run_id={run_id}"
    part.mkdir(parents=True)
    out = Y_rec[["unique_id", "ds", col_rec]].rename(columns={col_rec: "y_reconciled"})
    out.assign(forecast_date=str(test.ds.min().date())).to_csv(part / "part-0.csv", index=False)
    (part / "_lineage.json").write_text(json.dumps(lineage, indent=2, default=float))
    print(f"PUBLIE -> {part.relative_to(HERE)}")
else:
    print(f"BLOQUE ({cfg['quality_gate']['fail_action']}) : {report.failures}")
