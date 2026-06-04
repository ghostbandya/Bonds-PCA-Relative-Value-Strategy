"""Tests for StrategyConfig and split-date utilities.

Adapted from Jay's test_config.py, extended for the multi-country universe
and 80/20 train/test split.
"""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    StrategyConfig, get_split_dates, get_train_end,
    TRAIN_FRAC, TEST_FRAC, tenor_to_years,
)


# ---------------------------------------------------------------------------
# StrategyConfig — frozen / hashable / validation
# ---------------------------------------------------------------------------

class TestStrategyConfig:
    def test_frozen(self):
        """StrategyConfig is frozen: attribute assignment raises FrozenInstanceError."""
        cfg = StrategyConfig()
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError (or AttributeError)
            cfg.method = 2  # type: ignore

    def test_hashable(self):
        """Frozen dataclass must be hashable so it can be used as a dict/set key."""
        cfg = StrategyConfig()
        d = {cfg: "ok"}
        assert d[cfg] == "ok"

    def test_default_is_valid(self):
        """Default config (method=3, lw, 252/63/63, version A) should not raise."""
        cfg = StrategyConfig()
        assert cfg.method == 3
        assert cfg.covariance == "lw"
        assert cfg.pca_window == 252
        assert cfg.refit == 63
        assert cfg.z_window == 63
        assert cfg.version == "A"

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            StrategyConfig(method=4)

    def test_invalid_covariance(self):
        with pytest.raises(ValueError, match="covariance"):
            StrategyConfig(covariance="glasso")

    def test_invalid_version(self):
        with pytest.raises(ValueError, match="version"):
            StrategyConfig(version="C")

    def test_version_b_requires_K(self):
        with pytest.raises(ValueError, match="K"):
            StrategyConfig(version="B", K=None)

    def test_version_b_with_K_valid(self):
        cfg = StrategyConfig(version="B", K=63)
        assert cfg.K == 63

    def test_invalid_cost_model(self):
        with pytest.raises(ValueError, match="cost_model"):
            StrategyConfig(cost_model="percent")

    def test_no_trade_band_range(self):
        with pytest.raises(ValueError, match="no_trade_band"):
            StrategyConfig(no_trade_band=1.5)

    def test_no_trade_band_zero_valid(self):
        cfg = StrategyConfig(no_trade_band=0.0)
        assert cfg.no_trade_band == 0.0

    def test_cov_method_property(self):
        """cov_method property maps label to engine name."""
        assert StrategyConfig(covariance="sample").cov_method == "sample"
        assert StrategyConfig(covariance="ewma").cov_method == "ewma"
        assert StrategyConfig(covariance="lw").cov_method == "ledoit_wolf"

    def test_two_identical_configs_are_equal(self):
        c1 = StrategyConfig(method=2, covariance="sample")
        c2 = StrategyConfig(method=2, covariance="sample")
        assert c1 == c2
        assert hash(c1) == hash(c2)

    def test_different_configs_not_equal(self):
        c1 = StrategyConfig(method=1)
        c2 = StrategyConfig(method=2)
        assert c1 != c2


# ---------------------------------------------------------------------------
# 80/20 split utilities
# ---------------------------------------------------------------------------

class TestSplitDates:
    def _make_index(self, n=1000):
        return pd.date_range("2005-01-03", periods=n, freq="B")

    def test_split_keys(self):
        """get_split_dates returns exactly 'train' and 'test' keys (not val)."""
        idx = self._make_index()
        splits = get_split_dates(idx)
        assert set(splits.keys()) == {"train", "test"}, (
            f"Expected only 'train'/'test' keys, got {set(splits.keys())}"
        )

    def test_no_val_key(self):
        """There must be no 'val' key — we use 80/20, not 60/20/20."""
        idx = self._make_index()
        splits = get_split_dates(idx)
        assert "val" not in splits

    def test_train_fraction(self):
        """Train split should be approximately TRAIN_FRAC of the total."""
        n = 1000
        idx = self._make_index(n)
        splits = get_split_dates(idx)
        train_start, train_end = splits["train"]
        n_train = int(((idx >= train_start) & (idx <= train_end)).sum())
        expected = int(n * TRAIN_FRAC)
        assert abs(n_train - expected) <= 1, (
            f"Expected ~{expected} train days, got {n_train}"
        )

    def test_test_fraction(self):
        """Test split should be approximately TEST_FRAC of the total."""
        n = 1000
        idx = self._make_index(n)
        splits = get_split_dates(idx)
        test_start, test_end = splits["test"]
        n_test = int(((idx >= test_start) & (idx <= test_end)).sum())
        expected = int(n * TEST_FRAC)
        assert abs(n_test - expected) <= 2

    def test_splits_cover_full_range(self):
        """Train end + 1 day = Test start (no gap in dates)."""
        idx = self._make_index()
        splits = get_split_dates(idx)
        train_end   = splits["train"][1]
        test_start  = splits["test"][0]
        # test_start should be the very next index entry after train_end
        pos = idx.get_loc(train_end)
        assert idx[pos + 1] == test_start, (
            "Gap between train_end and test_start — dates not contiguous"
        )

    def test_splits_no_overlap(self):
        """Training and test periods must not overlap."""
        idx = self._make_index()
        splits = get_split_dates(idx)
        train_end  = splits["train"][1]
        test_start = splits["test"][0]
        assert test_start > train_end

    def test_train_end_is_80_percent(self):
        """get_train_end returns same value as splits['train'][1]."""
        idx = self._make_index()
        assert get_train_end(idx) == get_split_dates(idx)["train"][1]

    def test_fracs_sum_to_one(self):
        assert abs(TRAIN_FRAC + TEST_FRAC - 1.0) < 1e-12

    def test_train_frac_is_80(self):
        assert TRAIN_FRAC == 0.80

    def test_test_frac_is_20(self):
        assert TEST_FRAC == 0.20


# ---------------------------------------------------------------------------
# tenor_to_years helper
# ---------------------------------------------------------------------------

class TestTenorToYears:
    def test_standard_tenors(self):
        assert tenor_to_years("US_2Y")  == 2.0
        assert tenor_to_years("DE_10Y") == 10.0
        assert tenor_to_years("UK_30Y") == 30.0
        assert tenor_to_years("JP_5Y")  == 5.0

    def test_monotone(self):
        tenors = ["US_1Y", "US_2Y", "US_5Y", "US_10Y", "US_20Y", "US_30Y"]
        years = [tenor_to_years(t) for t in tenors]
        assert years == sorted(years)
