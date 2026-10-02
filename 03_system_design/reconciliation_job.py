"""Bloc 3 W37 - briques executables du contrat reconciliation.yaml.

Trois responsabilites, chacune testable isolement (cf. test_reconciliation_job.py) :
  1. hash_check       : refuser un run si S a change sans bump de version de la source ;
  2. choose_method    : mint_shrink, ou repli wls_struct si W est inutilisable ;
  3. quality_gate     : coherence, part de negatifs, MASE par niveau vs bottom-up -> publier ou bloquer.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_contract(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


# --- 1. hash de la structure S ------------------------------------------------------------
def s_hash(S_df: pd.DataFrame) -> str:
    """Empreinte de S : valeurs ET libelles (une feuille renommee change la hierarchie)."""
    S = S_df.set_index("unique_id")
    payload = json.dumps({"rows": S.index.tolist(), "cols": S.columns.tolist(),
                          "values": S.values.astype(int).tolist()})
    return hashlib.sha256(payload.encode()).hexdigest()


def hash_check(S_df: pd.DataFrame, source: str, registry_path: str | Path) -> str:
    """Enregistre le hash d'une nouvelle version ; refuse si une version connue a change de S."""
    registry_path = Path(registry_path)
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    h = s_hash(S_df)
    if source in registry and registry[source] != h:
        raise RuntimeError(f"S a change pour {source} sans bump de version "
                           f"({registry[source][:12]} -> {h[:12]}) : run refuse")
    registry[source] = h
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True))
    return h


# --- 2. choix du reconciliateur -----------------------------------------------------------
def choose_method(residuals: np.ndarray, cfg: dict) -> tuple[str, dict]:
    """residuals : (T, m), one-step in-sample, fenetre strictement anterieure a l'origine."""
    rc = cfg["reconciler"]
    rules = rc.get("fallback_when", {})
    T, m = residuals.shape
    diag = {"T": T, "m": m}
    if T < rules.get("min_residuals", 0):
        return rc["fallback"], {**diag, "reason": f"T={T} < min_residuals"}
    if rules.get("residuals_fewer_than_series", False) and m > T:
        return rc["fallback"], {**diag, "reason": "m > T"}
    W1 = np.cov(residuals.T)
    sd = np.sqrt(np.diag(W1))
    if np.any(sd == 0):
        return rc["fallback"], {**diag, "reason": "residu de variance nulle"}
    X = (residuals - residuals.mean(0)) / sd
    Wk = X[:, :, None] * X[:, None, :]
    r = Wk.mean(0)
    off = ~np.eye(m, dtype=bool)
    lam = float(np.clip((T / (T - 1) ** 3 * ((Wk - r) ** 2).sum(0))[off].sum() / (r[off] ** 2).sum(), 0, 1))
    W = lam * np.diag(np.diag(W1)) + (1 - lam) * W1
    cond = float(np.linalg.cond(W))
    diag.update(lam=lam, cond=cond)
    if cond > rules.get("max_condition_number", np.inf):
        return rc["fallback"], {**diag, "reason": f"cond(W)={cond:.1e}"}
    return rc["method"], {**diag, "reason": "ok"}


# --- 3. quality gate ----------------------------------------------------------------------
def mase(y: np.ndarray, yhat: np.ndarray, y_train: np.ndarray, season: int) -> np.ndarray:
    """MASE par serie. y, yhat : (h, m) ; y_train : (T, m). Denominateur sur le TRAIN."""
    scale = np.abs(y_train[season:] - y_train[:-season]).mean(axis=0)
    return np.abs(y - yhat).mean(axis=0) / scale


@dataclass
class GateReport:
    passed: bool
    checks: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)


def quality_gate(y_rec: np.ndarray, S: np.ndarray, levels: dict[str, list[int]], cfg: dict,
                 mase_rec: np.ndarray | None = None, mase_baseline: np.ndarray | None = None) -> GateReport:
    """y_rec : (h, m) dans l'ordre des lignes de S ; levels : nom -> indices de lignes.

    Les feuilles sont les nb dernieres lignes de S (convention HierarchicalForecast).
    """
    qg = cfg["quality_gate"]
    nb = S.shape[1]
    rep = GateReport(passed=True)

    # (a) coherence : y_agg == S_agg @ b, en absolu ET relatif au niveau des valeurs
    implied = y_rec[:, -nb:] @ S.T
    gap = np.abs(y_rec - implied)
    rel = gap.max() / max(np.abs(y_rec).max(), 1e-12)
    rep.checks["coherence_abs"] = float(gap.max())
    rep.checks["coherence_rel"] = float(rel)
    if gap.max() > float(qg["coherence_tol_abs"]) and rel > float(qg.get("coherence_tol_rel", 0)):
        rep.failures.append(f"incoherence {gap.max():.2e} (rel {rel:.2e})")
    if not np.isfinite(y_rec).all():
        rep.failures.append("valeurs non finies (NaN/inf) dans les previsions reconciliees")

    # (b) part de previsions negatives
    neg = float((y_rec < 0).mean())
    rep.checks["negative_share"] = neg
    if neg > float(qg["max_negative_share"]):
        rep.failures.append(f"part de negatifs {neg:.4f} > {qg['max_negative_share']}")

    # (c) MASE par niveau vs baseline : la degradation a UN niveau suffit a bloquer
    if mase_rec is not None and mase_baseline is not None:
        tol = float(qg.get("max_mase_degradation", 0.02))
        per_level = {}
        for name, idx in levels.items():
            r, b = float(np.mean(mase_rec[idx])), float(np.mean(mase_baseline[idx]))
            per_level[name] = {"mase": r, "baseline": b, "delta": r / b - 1}
            if r > b * (1 + tol):
                rep.failures.append(f"MASE {name} degrade de {r / b - 1:+.1%} vs {qg['compare_to_baseline']}")
        rep.checks["mase_by_level"] = per_level
        w = qg.get("level_weights", {})
        if w and set(w) <= set(per_level):
            rep.checks["weighted_mase_delta"] = sum(w[k] * per_level[k]["delta"] for k in w) / sum(w.values())

    rep.passed = not rep.failures
    return rep
