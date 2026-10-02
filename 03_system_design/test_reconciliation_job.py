"""Tests du contrat : chaque regle du YAML doit pouvoir echouer. Lancer : .venv/bin/pytest 03_system_design"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reconciliation_job import choose_method, hash_check, load_contract, quality_gate

CFG = load_contract(Path(__file__).parent / "reconciliation.yaml")
# 1 total + 3 feuilles
S = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
LEVELS = {"national": [0], "feuille": [1, 2, 3]}
S_DF = pd.DataFrame({"unique_id": ["T", "a", "b", "c"], "a": S[:, 0], "b": S[:, 1], "c": S[:, 2]})


def coherent(h=5, seed=0):
    b = np.random.default_rng(seed).uniform(10, 50, size=(h, 3))
    return b @ S.T


def test_gate_passes_on_coherent_forecasts():
    assert quality_gate(coherent(), S, LEVELS, CFG).passed


def test_gate_blocks_incoherent_forecasts():
    y = coherent(); y[2, 0] += 1e-3                      # total qui ne somme plus
    rep = quality_gate(y, S, LEVELS, CFG)
    assert not rep.passed and "incoherence" in rep.failures[0]


def test_gate_tolerates_float_noise_on_large_values():
    y = coherent() * 1e9; y[0, 0] *= 1 + 1e-15          # ~1e-5 en absolu, 1e-15 en relatif
    assert quality_gate(y, S, LEVELS, CFG).passed


def test_gate_blocks_nan():
    y = coherent(); y[1, 2] = np.nan
    assert not quality_gate(y, S, LEVELS, CFG).passed


def test_gate_blocks_negative_share():
    b = np.full((10, 3), 20.0); b[0, 0] = -1.0          # 1 feuille negative (total = 39 > 0) -> 1/40 valeurs < 0
    rep = quality_gate(b @ S.T, S, LEVELS, CFG)
    assert not rep.passed and rep.checks["negative_share"] == pytest.approx(1 / 40)


def test_gate_blocks_mase_degradation_at_any_level():
    # MinT ameliore le national (-30 %) mais degrade les feuilles (+5 %) : cas du bloc 4
    base = np.array([1.0, 1.0, 1.0, 1.0])
    rec = np.array([0.7, 1.05, 1.05, 1.05])
    rep = quality_gate(coherent(), S, LEVELS, CFG, mase_rec=rec, mase_baseline=base)
    assert not rep.passed and "feuille" in rep.failures[0]


def test_gate_accepts_degradation_within_tolerance():
    rep = quality_gate(coherent(), S, LEVELS, CFG,
                       mase_rec=np.array([0.9, 1.01, 1.01, 1.01]), mase_baseline=np.ones(4))
    assert rep.passed


def test_hash_check_refuses_changed_structure_without_bump(tmp_path):
    reg = tmp_path / "reg.json"
    hash_check(S_DF, "dim@v7", reg)
    hash_check(S_DF, "dim@v7", reg)                      # meme S : ok, idempotent
    renamed = S_DF.assign(unique_id=["T", "a", "b", "z"])  # une feuille renommee
    with pytest.raises(RuntimeError, match="sans bump"):
        hash_check(renamed, "dim@v7", reg)
    hash_check(renamed, "dim@v8", reg)                   # avec bump : accepte


def test_choose_method_keeps_mint_shrink_when_m_exceeds_T():
    E = np.random.default_rng(1).normal(size=(30, 50))   # m = 50 > T = 30
    method, why = choose_method(E, CFG)
    assert method == "mint_shrink" and why["cond"] < 1e10


def test_choose_method_falls_back_on_short_history():
    E = np.random.default_rng(1).normal(size=(10, 4))
    assert choose_method(E, CFG)[0] == "wls_struct"


def test_choose_method_falls_back_on_degenerate_residuals():
    E = np.random.default_rng(1).normal(size=(60, 4)); E[:, 2] = 0.0   # serie constante
    assert choose_method(E, CFG)[0] == "wls_struct"
