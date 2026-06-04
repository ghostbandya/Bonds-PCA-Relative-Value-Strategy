"""Tests for regime detection — 80/20 split and no-lookahead in HMM fitting.

Adapted from Jay's test_backtest.py no-lookahead philosophy: the HMM must be
fitted only on training data; mutating future features should not change the
HMM parameters (transition matrix, means) learned from training data alone.
"""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src._04_regime_detection import (
    build_stability_features, classify_hmm, classify_rule_based,
    normalize_features_percentile,
)
from src.config import get_split_dates, get_train_end


def _make_fake_pca_results(T=500, seed=42):
    """Synthetic PCA stability features that mimic the real output shape."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-03", periods=T, freq="B")
    # Simulate three regime-like periods
    cum_var = np.concatenate([
        rng.uniform(0.78, 0.84, T // 3),
        rng.uniform(0.72, 0.78, T // 3),
        rng.uniform(0.69, 0.74, T - 2 * (T // 3)),
    ])
    cos_sim = rng.uniform(0.995, 1.000, T)
    r2 = rng.uniform(0.70, 0.90, T)

    var_exp = pd.DataFrame({
        "PC1": cum_var * 0.6,
        "PC2": cum_var * 0.25,
        "PC3": cum_var * 0.15,
        "cumulative": cum_var,
    }, index=dates)
    stab = pd.DataFrame({
        "PC1_cos": cos_sim,
        "PC2_cos": cos_sim * 0.999,
        "PC3_cos": cos_sim * 0.998,
    }, index=dates)
    r2_series = pd.Series(r2, index=dates, name="r_squared")

    return {"var_explained": var_exp, "ev_stability": stab, "r_squared": r2_series}


class TestRegimeDetection:
    def test_rule_based_returns_valid_labels(self):
        pca = _make_fake_pca_results()
        features = build_stability_features(pca, smooth_window=5)
        regime = classify_rule_based(features)
        assert set(regime.unique()).issubset({0, 1, 2})
        assert len(regime) == len(features)

    def test_hmm_returns_correct_keys(self):
        pca = _make_fake_pca_results()
        features = build_stability_features(pca, smooth_window=5)
        idx = features.index
        train_end = get_train_end(idx)
        regime, proba, model = classify_hmm(features, train_end=train_end)
        assert set(regime.unique()).issubset({0, 1, 2})
        assert proba.shape[1] == 3
        # regime index is a subset of features index (percentile norm drops early rows)
        assert regime.index.isin(features.index).all()
        assert len(regime) > 0
        assert proba.shape[0] == len(regime)

    def test_hmm_only_trains_on_train_data(self):
        """Mutating features strictly after train_end does not change HMM parameters.

        The HMM's transition matrix and emission means must be identical whether
        or not future (test) feature data is perturbed, because the model is
        fitted ONLY on features[:train_end].
        """
        pca = _make_fake_pca_results(T=500)
        features = build_stability_features(pca, smooth_window=5)
        idx = features.index
        train_end = get_train_end(idx)

        # Fit on original features
        _, _, model_orig = classify_hmm(features, train_end=train_end)

        # Perturb all features strictly after train_end
        features_mut = features.copy()
        future_mask = features_mut.index > train_end
        rng = np.random.default_rng(1)
        features_mut.loc[future_mask] += rng.normal(
            0, 5.0, size=features_mut.loc[future_mask].shape
        )

        _, _, model_mut = classify_hmm(features_mut, train_end=train_end)

        # HMM params should be unchanged (fitted only on training data)
        assert np.allclose(
            model_orig.transmat_, model_mut.transmat_, atol=1e-10
        ), "HMM transition matrix changed after mutating test-period features"

        assert np.allclose(
            model_orig.means_, model_mut.means_, atol=1e-10
        ), "HMM emission means changed after mutating test-period features"

    def test_hmm_produces_predictions_on_train_and_test(self):
        """HMM fitted on training data produces regime labels for both splits."""
        pca = _make_fake_pca_results(T=600)
        features = build_stability_features(pca, smooth_window=5)
        train_end = get_train_end(features.index)
        regime, proba, _ = classify_hmm(features, train_end=train_end)
        train_regime = regime.loc[regime.index <= train_end]
        test_regime  = regime.loc[regime.index >  train_end]
        assert len(train_regime) > 0, "No training-period regime labels produced"
        assert len(test_regime)  > 0, "No test-period regime labels produced"
        # Both segments use the same integer label set
        assert set(regime.unique()).issubset({0, 1, 2})
        # Proba rows sum to 1
        assert np.allclose(proba.values.sum(axis=1), 1.0, atol=1e-6)

    def test_features_shape(self):
        pca = _make_fake_pca_results(T=300)
        features = build_stability_features(pca, smooth_window=5)
        assert "cum_var" in features.columns
        assert "mean_cos_sim" in features.columns
        assert "r_squared" in features.columns
        assert len(features) > 0

    def test_percentile_normalisation_range(self):
        """Expanding percentile rank should map values to [0, 1]."""
        pca = _make_fake_pca_results(T=300)
        features = build_stability_features(pca, smooth_window=5)
        normed = normalize_features_percentile(features, min_periods=10).dropna()
        assert normed.min().min() >= 0.0 - 1e-9
        assert normed.max().max() <= 1.0 + 1e-9
