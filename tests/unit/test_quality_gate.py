import numpy as np
import pytest

from w37_reconciliation.pipeline import mase, quality_gate

LEVELS = {"national": [0], "feuille": [1, 2, 3]}


def test_passes_on_coherent_forecasts(S_small, coherent, contract):
    assert quality_gate(coherent(), S_small, LEVELS, contract).passed


def test_blocks_incoherent_forecasts(S_small, coherent, contract):
    y = coherent()
    y[2, 0] += 1e-3                                       # le total ne somme plus
    rep = quality_gate(y, S_small, LEVELS, contract)
    assert not rep.passed and "incohérence" in rep.failures[0]


def test_tolerates_float_noise_on_large_values(S_small, coherent, contract):
    y = coherent() * 1e9
    y[0, 0] *= 1 + 1e-15                                  # ~1e-5 en absolu, 1e-15 en relatif
    assert quality_gate(y, S_small, LEVELS, contract).passed


def test_blocks_nan(S_small, coherent, contract):
    y = coherent()
    y[1, 2] = np.nan
    assert not quality_gate(y, S_small, LEVELS, contract).passed


def test_blocks_negative_share(S_small, contract):
    b = np.full((10, 3), 20.0)
    b[0, 0] = -1.0                                        # 1 feuille négative (total = 39 > 0) -> 1/40
    rep = quality_gate(b @ S_small.T, S_small, LEVELS, contract)
    assert not rep.passed and rep.checks["negative_share"] == pytest.approx(1 / 40)


def test_blocks_mase_degradation_at_any_level(S_small, coherent, contract):
    # Cas du bloc 4 : MinT améliore le national (−30 %) mais dégrade les feuilles (+5 %).
    rep = quality_gate(coherent(), S_small, LEVELS, contract,
                       mase_rec=np.array([0.7, 1.05, 1.05, 1.05]), mase_baseline=np.ones(4))
    assert not rep.passed and "feuille" in rep.failures[0]


def test_accepts_degradation_within_tolerance(S_small, coherent, contract):
    rep = quality_gate(coherent(), S_small, LEVELS, contract,
                       mase_rec=np.array([0.9, 1.01, 1.01, 1.01]), mase_baseline=np.ones(4))
    assert rep.passed


def test_mase_uses_seasonal_naive_scale_from_train():
    y_train = np.array([[1.0], [2.0], [3.0], [5.0]])      # |Δ_2| = 2, 3 -> échelle 2.5
    assert mase(np.array([[10.0]]), np.array([[15.0]]), y_train, season=2)[0] == pytest.approx(2.0)
