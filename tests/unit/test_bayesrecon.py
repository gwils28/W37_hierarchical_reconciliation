"""Bloc 6 : distributions, scores, exemple jouet exact et méthodes de réconciliation, sans réseau."""
import numpy as np
import pytest
from scipy import stats

from w37_reconciliation.bayesrecon import methods, scores
from w37_reconciliation.bayesrecon.example import M5Example
from w37_reconciliation.bayesrecon.pmf import Normal, Pmf
from w37_reconciliation.bayesrecon.toy import Toy

# --- distributions -------------------------------------------------------------------------------------


def test_pmf_moments_quantiles_and_cdf():
    p = Pmf.of([0.5, 0.3, 0.2])
    assert p.mean == pytest.approx(0.7)
    assert p.var == pytest.approx(0.3 + 0.8 - 0.49)
    assert p.quantile(0.5) == 0 and p.quantile(0.6) == 1 and p.quantile(0.95) == 2
    np.testing.assert_allclose(p.cdf(np.array([-1, 0, 1, 5])), [0, 0.5, 0.8, 1.0])
    assert p.prob_negative() == 0


def test_pmf_is_copied_and_renormalised():
    raw = np.array([1.0, 1.0])
    p = Pmf.of(raw)
    raw[0] = 99
    np.testing.assert_allclose(p.p, [0.5, 0.5])
    with pytest.raises(ValueError):
        Pmf.of([0.5, -0.5])


def test_normal_puts_mass_on_negative_sales_near_zero():
    d = Normal(0.8, 1.0)
    assert d.prob_negative() == pytest.approx(stats.norm.cdf(-1.3))
    assert d.quantile(0.05) < 0


# --- scores --------------------------------------------------------------------------------------------


def test_rps_is_zero_for_a_certain_and_correct_forecast():
    assert scores.rps(Pmf.of([0, 0, 1.0]), 2) == 0
    assert scores.rps(Pmf.of([1.0, 0, 0]), 2) == pytest.approx(2.0)      # F = 1 à k = 0 et 1 au lieu de 0


def test_interval_score_rewards_width_and_penalises_misses():
    d = Pmf.of([0.0, 0.05, 0.9, 0.05])          # F(1) = 0,05 et F(2) = 0,95 : intervalle 90 % = [1, 2]
    assert scores.interval_score(d, 2, alpha=0.1) == pytest.approx(1.0)
    assert scores.interval_score(d, 5, alpha=0.1) == pytest.approx(1.0 + 2 / 0.1 * 3)


def test_negatives_counts_impossible_forecasts():
    out = scores.negatives([Normal(-0.2, 1), Normal(0.5, 1), Normal(3, 1), Pmf.of([1.0])], alpha=0.1)
    assert out["moyennes < 0"] == 1 and out["bornes basses < 0"] == 2


def test_skill_score_is_symmetric_and_zero_when_equal():
    assert scores.skill([1, 2], [1, 2]) == 0
    assert scores.skill([2.0], [1.0]) == pytest.approx(-scores.skill([1.0], [2.0]))
    assert scores.skill([0.0], [0.0]) == 0


# --- exemple jouet ------------------------------------------------------------------------------------


def test_toy_conditioning_stays_on_non_negative_integers_and_moves_towards_the_total():
    t = Toy()
    ex = t.exact()
    assert all(d.p.sum() == pytest.approx(1) for d in ex.values())
    assert ex["b1"].mean < t.lam1 and ex["b2"].mean < t.lam2               # le total prévu est plus bas
    assert ex["total"].mean == pytest.approx(ex["b1"].mean + ex["b2"].mean, abs=1e-9)


def test_toy_gaussian_mint_is_coherent_but_puts_mass_below_zero():
    g = Toy().gaussian_mint()
    assert g["total"].mu == pytest.approx(g["b1"].mu + g["b2"].mu)
    assert g["b1"].prob_negative() > 0.05


def test_bayesreconpy_buis_recovers_the_exact_conditional_distribution():
    t = Toy()
    exact, buis = t.exact(), t.buis(num_samples=200_000)
    for k in ("b1", "b2", "total"):
        n = min(len(exact[k].p), len(buis[k].p))
        np.testing.assert_allclose(buis[k].p[:n], exact[k].p[:n], atol=0.01)


def test_buis_by_hand_matches_the_exact_conditional_distribution():
    t = Toy()
    exact, hand = t.exact(), t.buis_by_hand(num_samples=200_000)
    for k in ("b1", "b2", "total"):
        n = min(len(exact[k].p), len(hand[k].p))
        np.testing.assert_allclose(hand[k].p[:n], exact[k].p[:n], atol=0.01)


# --- méthodes sur une petite hiérarchie synthétique ---------------------------------------------------------


@pytest.fixture
def small():
    rng = np.random.default_rng(0)
    # total + deux sous-groupes : TD-cond exige que chaque article soit dans UN groupe du niveau le plus bas
    A = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)
    lam = np.array([0.3, 0.6, 1.5, 0.2])
    k = np.arange(30)
    pmfs = [Pmf.of(stats.nbinom.pmf(k, 1.0, 1 / (1 + m))) for m in lam]    # surdispersées
    return M5Example(A=A, upper_names=["T", "G1", "G2"], bottom_names=list("abcd"),
                     mu_u=np.array([1.8, 0.6, 1.3]), sd_u=np.array([0.9, 0.5, 0.7]),
                     residuals_u=rng.normal(size=(300, 3)) * [0.9, 0.5, 0.7], pmf_b=pmfs,
                     actual_u=np.array([2.0, 1.0, 1.0]), actual_b=np.array([0.0, 1.0, 1.0, 0.0]),
                     Q_u=np.ones(3), Q_b=np.ones(4))


def test_conditioning_methods_do_not_mutate_their_inputs(small):
    before = [p.p.copy() for p in small.pmf_b]
    methods.td_cond(small, 2000, seed=1)
    methods.mix_cond(small, 2000, seed=1)
    for p, b in zip(small.pmf_b, before, strict=True):
        np.testing.assert_array_equal(p.p, b)


def test_every_method_is_coherent_and_only_the_gaussian_one_goes_negative(small):
    recs = methods.run_all(small, 4000, seed=2)
    table = scores.summary(recs, small)
    for name in ("MinT gaussien", "MixCond", "TD-cond"):
        assert table.loc[name, "incohérence max (ventes)"] < 1e-6
    for name in ("bottom-up", "MixCond", "TD-cond"):
        assert table.loc[name, "articles · bornes basses < 0"] == 0
    assert table.loc["MinT gaussien", "articles · masse sous zéro (moyenne)"] > 0
