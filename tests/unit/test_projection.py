import numpy as np

from w37_reconciliation.mint import incoherence, mint_projection, reconcile, unbiasedness_error


def random_spd(m: int, seed: int = 0) -> np.ndarray:
    A = np.random.default_rng(seed).normal(size=(m, m))
    return A @ A.T + m * np.eye(m)


def test_projection_is_idempotent_and_unbiased(S_small):
    _, P = mint_projection(S_small, random_spd(4))
    np.testing.assert_allclose(P @ P, P, atol=1e-12)
    assert unbiasedness_error(P, S_small) < 1e-12


def test_reconciled_forecasts_are_coherent(S_small):
    _, P = mint_projection(S_small, random_spd(4, seed=1))
    y_hat = np.random.default_rng(2).normal(50, 10, size=(8, 4))    # incohérentes
    assert incoherence(y_hat, S_small) > 1
    assert incoherence(reconcile(y_hat, P), S_small) < 1e-10


def test_coherent_input_is_left_unchanged(S_small, coherent):
    _, P = mint_projection(S_small, random_spd(4, seed=3))
    y = coherent()
    np.testing.assert_allclose(reconcile(y, P), y, atol=1e-10)


def test_ols_is_orthogonal_projection(S_small):
    _, P = mint_projection(S_small, np.eye(4))
    np.testing.assert_allclose(P, P.T, atol=1e-12)


def test_trusting_leaves_approaches_bottom_up(S_small):
    # Feuilles quasi parfaites, total très bruité -> G tend vers [0 | I].
    G, _ = mint_projection(S_small, np.diag([1e6, 1e-6, 1e-6, 1e-6]))
    np.testing.assert_allclose(G, np.hstack([np.zeros((3, 1)), np.eye(3)]), atol=1e-6)
