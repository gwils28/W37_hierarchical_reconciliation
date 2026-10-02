import numpy as np
import pytest

from w37_reconciliation.mint import schafer_strimmer_lambda, shrunk_covariance


def correlated_residuals(T: int, m: int, rho: float, seed: int = 0) -> np.ndarray:
    R = (1 - rho) * np.eye(m) + rho * np.ones((m, m))
    return np.random.default_rng(seed).normal(size=(T, m)) @ np.linalg.cholesky(R).T


def test_lambda_is_bounded():
    for seed in range(5):
        lam = schafer_strimmer_lambda(np.random.default_rng(seed).normal(size=(20, 6)))
        assert 0.0 <= lam <= 1.0


def test_lambda_high_when_correlations_are_noise():
    # Séries indépendantes : les corrélations empiriques sont du bruit -> rétrécir fort.
    assert schafer_strimmer_lambda(correlated_residuals(40, 6, rho=0.0)) > 0.5


def test_lambda_low_when_correlations_are_real():
    # Forte corrélation réelle et beaucoup d'observations : peu de shrinkage.
    assert schafer_strimmer_lambda(correlated_residuals(500, 6, rho=0.8)) < 0.05


def test_shrinkage_keeps_variances_and_shrinks_covariances():
    cov = shrunk_covariance(correlated_residuals(50, 4, rho=0.5))
    np.testing.assert_allclose(np.diag(cov.W), np.diag(cov.W1))
    off = ~np.eye(4, dtype=bool)
    np.testing.assert_allclose(cov.W[off], (1 - cov.lam) * cov.W1[off])


def test_shrinkage_makes_singular_covariance_invertible():
    E = correlated_residuals(T=4, m=6, rho=0.3)          # m > T
    cov = shrunk_covariance(E)
    assert np.linalg.matrix_rank(cov.W1) < 6
    assert np.all(np.linalg.eigvalsh(cov.W) > 0)
    assert cov.condition_number < 1e3


def test_library_base_is_centered_ddof1():
    E = correlated_residuals(30, 3, rho=0.2) + 5.0       # résidus biaisés : le centrage compte
    np.testing.assert_allclose(shrunk_covariance(E, base="library").W1, np.cov(E.T))
    np.testing.assert_allclose(shrunk_covariance(E, base="plan").W1, E.T @ E / 30)


def test_unknown_base_raises():
    with pytest.raises(ValueError):
        shrunk_covariance(np.ones((5, 2)), base="autre")
