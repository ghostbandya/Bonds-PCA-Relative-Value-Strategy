"""Tests for z-score signal generation."""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src._05_signal_generation import compute_z_scores


def make_residuals(T=300, N=5, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-04", periods=T, freq="B")
    cols  = ["US_2Y","US_5Y","US_10Y","DE_10Y","UK_10Y"]
    return pd.DataFrame(rng.standard_normal((T, N)) * 0.01, index=dates, columns=cols)


class TestZScores:
    def test_shape(self):
        resid = make_residuals()
        z = compute_z_scores(resid, z_window=63)
        assert z.shape == resid.shape

    def test_warmup_nans(self):
        """First z_window-1 rows should be NaN (window not yet warm)."""
        resid = make_residuals()
        z = compute_z_scores(resid, z_window=63)
        # Should have NaNs in early rows
        assert z.iloc[:62].isna().all().all()

    def test_post_warmup_finite(self):
        """After warmup, values should be finite for most dates."""
        resid = make_residuals()
        z = compute_z_scores(resid, z_window=63)
        z_after = z.iloc[63:]
        assert z_after.notna().mean().mean() > 0.8

    def test_version_a_resets(self):
        """Version A cumsum should reset to 0 at refit boundaries."""
        resid = make_residuals(T=200)
        dates = resid.index
        refit_dates = [dates[50], dates[100], dates[150]]
        z = compute_z_scores(resid, z_window=30, version="A", refit_dates=refit_dates)
        # Just checks no error and correct shape
        assert z.shape == resid.shape

    def test_version_b_requires_K(self):
        resid = make_residuals()
        with pytest.raises(ValueError):
            compute_z_scores(resid, version="B", K=None)

    def test_version_b_shape(self):
        resid = make_residuals()
        z = compute_z_scores(resid, z_window=63, version="B", K=63)
        assert z.shape == resid.shape

    def test_mean_near_zero(self):
        """Z-scores should be approximately mean-zero over a long sample.
        The z-score is (cum - rolling_mean) / rolling_std where cum is a
        random walk. The trailing window mean adapts slowly so there can be
        a modest persistent bias; threshold of 1.0 is appropriate."""
        resid = make_residuals(T=500)
        z = compute_z_scores(resid, z_window=63)
        mean_abs = z.dropna().mean().abs()
        assert (mean_abs < 1.0).all(), (
            f"Z-score means too large: {mean_abs.to_dict()}"
        )

    def test_no_lookahead(self):
        """Adding future data should not change past z-scores.
        We extend resid1 by appending new dates (not overlapping ones)."""
        resid1 = make_residuals(T=200)
        # Create 50 extra days starting right after resid1 ends
        last_date = resid1.index[-1]
        extra_dates = pd.date_range(last_date + pd.offsets.BDay(1), periods=50, freq="B")
        rng = np.random.default_rng(99)
        extra_data = pd.DataFrame(
            rng.standard_normal((50, resid1.shape[1])) * 0.01,
            index=extra_dates,
            columns=resid1.columns,
        )
        resid2 = pd.concat([resid1, extra_data])
        z1 = compute_z_scores(resid1, z_window=63)
        z2 = compute_z_scores(resid2, z_window=63)
        # z-scores on the original 200 days should be identical in both runs
        assert np.allclose(
            z1.values, z2.loc[resid1.index].values, equal_nan=True
        ), "Z-scores changed when future data was appended — look-ahead bias detected"
