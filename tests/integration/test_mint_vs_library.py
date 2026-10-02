"""Bloc 2 de bout en bout : les critères « c'est fini quand… » du plan W37."""
import pytest

from w37_reconciliation.mint.experiment import gap_decomposition, m_greater_than_T_demo, run_mint_experiment

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def exp():
    return run_mint_experiment()


def test_projection_preserves_unbiasedness(exp):
    assert exp.metrics["unbiasedness"] < 1e-12


def test_reconciled_forecasts_are_coherent(exp):
    assert exp.metrics["incoherence"] < 1e-9


def test_close_to_library_but_not_identical(exp):
    assert 0 < exp.metrics["relative_gap"] < 1e-3


def test_gap_comes_from_covariance_base_not_lambda(exp):
    gaps = gap_decomposition(exp)["écart absolu"].to_numpy()
    assert gaps[1] < gaps[0] / 100          # même base que la lib -> écart divisé par > 100


def test_shrinkage_rescues_m_greater_than_T(exp):
    demo = m_greater_than_T_demo(exp.E)
    assert demo["rank_W1"] < demo["m"] and demo["cond_W"] < 1e3
