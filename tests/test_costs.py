"""Tests for transaction cost models."""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.costs import compute_cost_flat, compute_cost_dv01bp


class TestFlatCost:
    def test_zero_cost_per_contract(self):
        delta_g = np.array([1.0, -0.5])
        dv01    = np.array([0.08, 0.04])
        cols    = ["US_10Y", "US_5Y"]
        c = compute_cost_flat(delta_g, dv01, cols, cost_per_contract_usd=0.0)
        assert c == 0.0

    def test_positive_cost(self):
        delta_g = np.array([1.0, 0.0])
        dv01    = np.array([0.08, 0.04])
        cols    = ["US_10Y", "US_5Y"]
        c = compute_cost_flat(delta_g, dv01, cols, cost_per_contract_usd=8.5)
        assert c > 0

    def test_symmetric_in_sign(self):
        dv01 = np.array([0.08, 0.04])
        cols = ["US_10Y", "US_5Y"]
        c_pos = compute_cost_flat(np.array([1.0, 0.5]),  dv01, cols, 8.5)
        c_neg = compute_cost_flat(np.array([-1.0, -0.5]), dv01, cols, 8.5)
        assert abs(c_pos - c_neg) < 1e-10, "Cost should not depend on sign of trade"


class TestDv01BpCost:
    def test_zero_delta_zero_cost(self):
        delta_g = np.zeros(2)
        dv01    = np.array([0.08, 0.04])
        cols    = ["US_10Y", "US_5Y"]
        c = compute_cost_dv01bp(delta_g, dv01, cols)
        assert c == 0.0

    def test_dv01_cancels_cost_driven_by_spread(self):
        """In DV01_bp model: cost = half_spread * delta_g (DV01 cancels out).
        2Y has wider half-spread (0.20bp) than 10Y (0.12bp), so for the same
        delta_g, 2Y costs more — the opposite of the flat $/contract model."""
        dv01 = np.array([0.02, 0.08])  # 2Y vs 10Y
        cols = ["US_2Y", "US_10Y"]
        delta_2y  = np.array([1.0, 0.0])
        delta_10y = np.array([0.0, 1.0])
        c2  = compute_cost_dv01bp(delta_2y,  dv01, cols)
        c10 = compute_cost_dv01bp(delta_10y, dv01, cols)
        # 2Y half-spread (0.20bp) > 10Y half-spread (0.12bp) -> c2 > c10
        assert c2 > c10, (
            f"2Y cost ({c2:.2e}) should exceed 10Y cost ({c10:.2e}) "
            "because 2Y has a wider bid-ask spread"
        )

    def test_spread_mult_scales_cost(self):
        delta_g = np.array([1.0])
        dv01    = np.array([0.08])
        cols    = ["US_10Y"]
        c1   = compute_cost_dv01bp(delta_g, dv01, cols, spread_mult=1.0)
        c1_5 = compute_cost_dv01bp(delta_g, dv01, cols, spread_mult=1.5)
        assert abs(c1_5 / c1 - 1.5) < 1e-9
