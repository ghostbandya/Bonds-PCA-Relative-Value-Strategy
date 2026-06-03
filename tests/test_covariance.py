"""Tests for covariance estimators."""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.covariance import estimate_cov, pca_from_cov


def make_data(T=300, N=10, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, N))


class TestEstimateCov:
    def test_sample_shape(self):
        X = make_data()
        mean_, Sigma, _ = estimate_cov(X, method="sample")
        assert Sigma.shape == (10, 10)
        assert mean_.shape == (10,)

    def test_sample_symmetric(self):
        X = make_data()
        _, Sigma, _ = estimate_cov(X, method="sample")
        assert np.allclose(Sigma, Sigma.T, atol=1e-10)

    def test_sample_positive_semidefinite(self):
        X = make_data()
        _, Sigma, _ = estimate_cov(X, method="sample")
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals >= -1e-10)

    def test_ewma_shape(self):
        X = make_data()
        _, Sigma, extra = estimate_cov(X, method="ewma", ewma_halflife=63)
        assert Sigma.shape == (10, 10)
        assert "ewma_halflife" in extra

    def test_lw_shape(self):
        X = make_data()
        _, Sigma, extra = estimate_cov(X, method="ledoit_wolf")
        assert Sigma.shape == (10, 10)
        assert "shrinkage" in extra

    def test_lw_shrinkage_in_01(self):
        X = make_data()
        _, _, extra = estimate_cov(X, method="ledoit_wolf")
        assert 0.0 <= extra["shrinkage"] <= 1.0

    def test_unknown_method_raises(self):
        X = make_data()
        with pytest.raises(ValueError):
            estimate_cov(X, method="unknown")


class TestPcaFromCov:
    def test_loading_shape(self):
        X = make_data()
        V, lam, Sigma = pca_from_cov(X, n_pc=3)
        assert V.shape == (3, 10)
        assert lam.shape == (3,)

    def test_orthonormal_rows(self):
        X = make_data()
        V, _, _ = pca_from_cov(X, n_pc=3)
        assert np.allclose(V @ V.T, np.eye(3), atol=1e-10), "V rows should be orthonormal"

    def test_eigenvalue_ordering(self):
        X = make_data()
        _, lam, _ = pca_from_cov(X, n_pc=3)
        assert lam[0] >= lam[1] >= lam[2], "Eigenvalues should be in descending order"

    def test_sign_stabilisation(self):
        """Max-abs element of each loading row should be positive."""
        X = make_data()
        V, _, _ = pca_from_cov(X, n_pc=3)
        for j in range(3):
            max_abs_val = V[j, np.argmax(np.abs(V[j]))]
            assert max_abs_val > 0, f"PC{j+1} largest element should be positive"
