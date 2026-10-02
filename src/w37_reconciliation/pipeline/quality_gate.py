"""Quality gate : décide si un run est publiable. Toute règle doit pouvoir échouer."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def mase(y: np.ndarray, y_hat: np.ndarray, y_train: np.ndarray, season: int) -> np.ndarray:
    """MASE par série. y, y_hat : (h × m) ; y_train : (T × m). Dénominateur calculé sur le TRAIN."""
    scale = np.abs(y_train[season:] - y_train[:-season]).mean(axis=0)
    return np.abs(y - y_hat).mean(axis=0) / scale


@dataclass
class GateReport:
    passed: bool
    checks: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


def quality_gate(y_rec: np.ndarray, S: np.ndarray, levels: dict[str, list[int]], cfg: dict,
                 mase_rec: np.ndarray | None = None,
                 mase_baseline: np.ndarray | None = None) -> GateReport:
    """y_rec : (h × m) dans l'ordre des lignes de S ; levels : nom -> indices de lignes.

    Les feuilles sont les nb dernières lignes de S (convention HierarchicalForecast).
    """
    qg = cfg["quality_gate"]
    nb = S.shape[1]
    rep = GateReport(passed=True)

    # (a) cohérence : passe si l'écart respecte le seuil absolu OU le seuil relatif
    gap = np.abs(y_rec - y_rec[:, -nb:] @ S.T)
    rel = gap.max() / max(np.abs(y_rec).max(), 1e-12)
    rep.checks["coherence_abs"] = float(gap.max())
    rep.checks["coherence_rel"] = float(rel)
    if gap.max() > float(qg["coherence_tol_abs"]) and rel > float(qg.get("coherence_tol_rel", 0)):
        rep.failures.append(f"incohérence {gap.max():.2e} (rel {rel:.2e})")
    if not np.isfinite(y_rec).all():
        rep.failures.append("valeurs non finies (NaN/inf) dans les prévisions réconciliées")

    # (b) part de prévisions négatives
    neg = float((y_rec < 0).mean())
    rep.checks["negative_share"] = neg
    if neg > float(qg["max_negative_share"]):
        rep.failures.append(f"part de négatifs {neg:.4f} > {qg['max_negative_share']}")

    # (c) MASE par niveau vs baseline : une dégradation à UN niveau suffit à bloquer
    if mase_rec is not None and mase_baseline is not None:
        tol = float(qg.get("max_mase_degradation", 0.02))
        per_level = {}
        for name, idx in levels.items():
            r, b = float(np.mean(mase_rec[idx])), float(np.mean(mase_baseline[idx]))
            per_level[name] = {"mase": r, "baseline": b, "delta": r / b - 1}
            if r > b * (1 + tol):
                rep.failures.append(f"MASE {name} dégradé de {r / b - 1:+.1%} vs {qg['compare_to_baseline']}")
        rep.checks["mase_by_level"] = per_level
        w = qg.get("level_weights", {})
        if w and set(w) <= set(per_level):   # rapporté seulement : la règle bloquante reste par niveau
            rep.checks["weighted_mase_delta"] = sum(w[k] * per_level[k]["delta"] for k in w) / sum(w.values())

    rep.passed = not rep.failures
    return rep
