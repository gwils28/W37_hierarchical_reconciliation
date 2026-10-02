"""hash_check, choose_method, load_contract."""
import numpy as np
import pytest

from w37_reconciliation.pipeline import choose_method, hash_check, load_contract


def test_hash_check_refuses_changed_structure_without_bump(tmp_path, S_df_small):
    reg = tmp_path / "reg.json"
    hash_check(S_df_small, "dim@v7", reg)
    hash_check(S_df_small, "dim@v7", reg)                 # même S : idempotent
    renamed = S_df_small.assign(unique_id=["T", "a", "b", "z"])
    with pytest.raises(RuntimeError, match="sans bump"):
        hash_check(renamed, "dim@v7", reg)
    hash_check(renamed, "dim@v8", reg)                    # avec bump : accepté


def test_keeps_mint_shrink_when_m_exceeds_T(contract):
    E = np.random.default_rng(1).normal(size=(30, 50))    # m = 50 > T = 30
    method, why = choose_method(E, contract)
    assert method == "mint_shrink" and why["cond"] < 1e10


def test_falls_back_on_short_history(contract):
    E = np.random.default_rng(1).normal(size=(10, 4))
    method, why = choose_method(E, contract)
    assert method == "wls_struct" and "min_residuals" in why["reason"]


def test_falls_back_on_degenerate_residuals(contract):
    E = np.random.default_rng(1).normal(size=(60, 4))
    E[:, 2] = 0.0                                         # série constante
    assert choose_method(E, contract)[0] == "wls_struct"


def test_contract_missing_section_fails_early(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("job: x\nhierarchy: {}\n")
    with pytest.raises(ValueError, match="sections manquantes"):
        load_contract(bad)
