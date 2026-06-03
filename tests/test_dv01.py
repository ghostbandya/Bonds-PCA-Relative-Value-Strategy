"""Tests for DV01 engine — adapted from Jay's test_dv01.py."""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dv01 import dv01_per_100, build_dv01_matrix, notionals_from_g
import pandas as pd

class TestDv01PerInstrument:
    """Verify DV01 formulas against documented worked examples (Jay §2)."""

    def test_10y_par_bond(self):
        # 10Y par bond at 4.47% yield: doc ~0.0800
        result = dv01_per_100(4.47, 10.0)
        assert abs(result - 0.0800) < 0.001, f"Expected ~0.0800, got {result:.5f}"

    def test_1y_zero(self):
        # 1Y zero at 4.9%: doc ~0.0093
        result = dv01_per_100(4.9, 1.0)
        assert abs(result - 0.0093) < 0.001, f"Expected ~0.0093, got {result:.5f}"

    def test_2y_par_bond(self):
        # 2Y par bond at ~4%: should be ~0.019
        result = dv01_per_100(4.0, 2.0)
        assert 0.015 < result < 0.022, f"2Y DV01 out of range: {result}"

    def test_30y_par_bond(self):
        # 30Y par bond: DV01 should be much higher than 10Y
        dv_10 = dv01_per_100(4.0, 10.0)
        dv_30 = dv01_per_100(4.0, 30.0)
        assert dv_30 > dv_10 * 1.5, "30Y DV01 should be > 1.5x 10Y"

    def test_nan_propagation(self):
        result = dv01_per_100(float("nan"), 10.0)
        assert np.isnan(result), "NaN yield should return NaN DV01"

    def test_zirp_guard(self):
        # Near-zero yield should not blow up
        result = dv01_per_100(0.001, 10.0)
        assert np.isfinite(result) and result > 0

    def test_monotone_in_tenor(self):
        # Higher tenor -> higher DV01 (holding yield constant)
        y = 4.0
        dv2  = dv01_per_100(y, 2.0)
        dv5  = dv01_per_100(y, 5.0)
        dv10 = dv01_per_100(y, 10.0)
        dv30 = dv01_per_100(y, 30.0)
        assert dv2 < dv5 < dv10 < dv30

    def test_monotone_in_yield(self):
        # Higher yield -> lower DV01 (inverse relationship for par bonds)
        assert dv01_per_100(2.0, 10.0) > dv01_per_100(5.0, 10.0)


class TestBuildDv01Matrix:
    def test_shape(self):
        dates = pd.date_range("2020-01-02", periods=10, freq="B")
        cols = pd.MultiIndex.from_tuples(
            [("US","10Y"),("US","2Y"),("DE","10Y")], names=["country","tenor"])
        yields = pd.DataFrame(4.0, index=dates, columns=cols)
        mat = build_dv01_matrix(yields)
        assert mat.shape == (10, 3)

    def test_values_positive(self):
        dates = pd.date_range("2020-01-02", periods=5, freq="B")
        cols = pd.MultiIndex.from_tuples(
            [("US","10Y"),("US","30Y")], names=["country","tenor"])
        yields = pd.DataFrame(4.0, index=dates, columns=cols)
        mat = build_dv01_matrix(yields)
        assert (mat > 0).all().all()


class TestNotionals:
    def test_sign_convention(self):
        # Positive g (long yield) -> negative notional (short bond)
        g    = np.array([1.0, 0.0])
        dv01 = np.array([0.08, 0.04])
        q    = notionals_from_g(g, dv01)
        assert q[0] < 0, "Long yield should be short bond notional"

    def test_zero_g(self):
        g    = np.zeros(3)
        dv01 = np.array([0.08, 0.04, 0.02])
        q    = notionals_from_g(g, dv01)
        assert np.allclose(q, 0)
