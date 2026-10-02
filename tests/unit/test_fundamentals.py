"""Bloc 4 : chaque affirmation chiffrée du README est figée par un test."""
import numpy as np
import pytest

from w37_reconciliation.fundamentals import (
    bias_allocation,
    bias_counterexample,
    bias_sweep,
    bootstrap_coverage,
    coverage_experiment,
    rho_sweep,
    toy_hierarchy,
)
from w37_reconciliation.fundamentals.diagnostics import (
    random_unbiased_G,
    residual_bias_table,
    weighted_trace_check,
)
from w37_reconciliation.mint import mint_projection

# --- n°1 : le biais --------------------------------------------------------------------------


def test_counterexample_matches_plan():
    tab = bias_counterexample()
    assert tab["erreur bottom-up"].abs().sum() == 0
    assert tab.loc["Feuille_1":, "erreur MinT"].abs().sum() == pytest.approx(14.4643, abs=1e-4)


def test_bias_spreads_along_column_of_P():
    alloc = bias_allocation(toy_hierarchy(), np.diag([1.0, 9.0, 9.0, 9.0]))
    np.testing.assert_allclose(alloc, [27 / 28, 9 / 28, 9 / 28, 9 / 28])
    np.testing.assert_allclose(15 * alloc[1:], bias_counterexample()["erreur MinT"].iloc[1:])


def test_mint_beats_bottom_up_without_bias_and_loses_with_large_bias():
    s = bias_sweep([0.0, 8.0], n=20_000).set_index(["biais", "méthode"])
    assert s.loc[(0.0, "MinT"), "MAE feuilles"] < s.loc[(0.0, "bottom-up"), "MAE feuilles"]
    assert s.loc[(8.0, "MinT"), "MAE feuilles"] > s.loc[(8.0, "bottom-up"), "MAE feuilles"]
    # vs prévisions de base : le total s'améliore, les feuilles se dégradent (la phrase du plan)
    assert s.loc[(8.0, "MinT"), "MAE total"] < s.loc[(8.0, "base"), "MAE total"]
    assert s.loc[(8.0, "MinT"), "MAE feuilles"] > s.loc[(8.0, "base"), "MAE feuilles"]


def test_residual_bias_table_flags_only_biased_series():
    E = np.random.default_rng(0).normal(size=(200, 4))
    E[:, 0] += 1.0
    flags = residual_bias_table(E, ["T", "a", "b", "c"])["biais suspect"].tolist()
    assert flags == [True, False, False, False]


# --- n°2 : les quantiles ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def coverage():
    return coverage_experiment()            # graine 37, N = 400 000 : les chiffres exacts du plan


def test_method_A_is_wrong_in_both_directions(coverage):
    assert coverage.loc["Total", "cov A"] == pytest.approx(0.922, abs=1e-3)
    assert (coverage.loc["Region_1":, "cov A"] < 0.895).all()


def test_method_B_undercovers_total(coverage):
    assert coverage.loc["Total", "cov B"] == pytest.approx(0.764, abs=1e-3)


def test_method_C_is_calibrated_everywhere(coverage):
    np.testing.assert_allclose(coverage["cov C"], 0.90, atol=2e-3)


def test_rho_sweep_errors_go_in_opposite_directions():
    sw = rho_sweep(n=100_000)
    # empiler les quantiles : de moins en moins faux ; ignorer la dépendance : de plus en plus faux
    assert sw["écart"].is_monotonic_decreasing
    assert sw["couverture B (total)"].is_monotonic_decreasing


def test_joint_bootstrap_recovers_calibration():
    cov = bootstrap_coverage(n_samples=20_000, n_truth=100_000)
    assert abs(cov.loc["Total", "cov bootstrap joint"] - 0.90) < 0.02
    assert cov.loc["Total", "cov bootstrap indépendant"] < 0.80


# --- question d'entretien : pondérer les niveaux ---------------------------------------------


def test_random_G_is_unbiased():
    S = toy_hierarchy()
    G = random_unbiased_G(S, mint_projection(S, np.eye(4))[0], np.random.default_rng(0))
    np.testing.assert_allclose(S @ G @ S, S, atol=1e-12)


@pytest.mark.parametrize("weights", [[1, 1, 1, 1], [10, 0.01, 0.01, 0.01], [0.01, 5, 0.01, 0.01]])
def test_no_level_weighting_beats_mint_when_W_is_true(weights):
    A = np.random.default_rng(1).normal(size=(4, 4))
    res = weighted_trace_check(toy_hierarchy(), A @ A.T + np.eye(4), np.array(weights, float), n_random=500)
    assert res["part des G aléatoires battant MinT"] == 0.0
