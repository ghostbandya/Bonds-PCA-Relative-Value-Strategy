"""Tests for the no-trade band."""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.rebalance import apply_no_trade_band


def make_book(T=50, N=5, seed=42):
    """Random iid book — each day is an independent draw."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=T, freq="B")
    cols  = [f"US_{t}" for t in ["1Y","2Y","5Y","10Y","30Y"]]
    data  = rng.standard_normal((T, N)) * 0.01
    return pd.DataFrame(data, index=dates, columns=cols)


def make_slow_book(T=100, N=5, seed=42):
    """Slowly drifting book (random walk with small steps).
    The no-trade band is designed for slowly-changing targets — when the
    optimal book drifts gradually, holding yesterday's position is close
    to optimal. With iid random data, the target jumps wildly each day
    and the band rarely triggers a hold."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=T, freq="B")
    cols  = [f"US_{t}" for t in ["1Y","2Y","5Y","10Y","30Y"]]
    increments = rng.standard_normal((T, N)) * 0.001  # very small steps
    data = np.cumsum(increments, axis=0)
    return pd.DataFrame(data, index=dates, columns=cols)


class TestNoTradeBand:
    def test_tau_zero_identity(self):
        """tau=0 should return the target unchanged."""
        G = make_book()
        Sigma_dict = {}
        V_dict     = {}
        exec_G, hold_mask = apply_no_trade_band(G, Sigma_dict, V_dict, tau=0.0)
        assert np.allclose(exec_G.values, G.values)
        assert not hold_mask.any()

    def test_high_tau_causes_holds(self):
        """tau=0.9 on slowly-drifting data should produce many hold days.
        When the target evolves gradually, yesterday's book is already close
        enough to today's target that the drift ratio stays below tau."""
        G = make_slow_book(T=100)
        Sigma_dict = {d: np.eye(5) for d in G.index}
        V_dict     = {}
        _, hold_mask = apply_no_trade_band(G, Sigma_dict, V_dict, tau=0.9)
        assert hold_mask.sum() > 10, (
            f"Expected many hold days with tau=0.9 on slow data, "
            f"got {hold_mask.sum()}"
        )

    def test_output_shape(self):
        G = make_book()
        Sigma_dict = {d: np.eye(5) for d in G.index}
        V_dict     = {}
        exec_G, hold_mask = apply_no_trade_band(G, Sigma_dict, V_dict, tau=0.5)
        assert exec_G.shape == G.shape
        assert len(hold_mask) == len(G)

    def test_hold_reduces_turnover(self):
        """Executed book should have lower total variation than target."""
        G = make_book()
        Sigma_dict = {d: np.eye(5) for d in G.index}
        V_dict     = {}
        exec_G, _ = apply_no_trade_band(G, Sigma_dict, V_dict, tau=0.7)
        tv_target = G.diff().abs().sum().sum()
        tv_exec   = exec_G.diff().abs().sum().sum()
        assert tv_exec < tv_target, "No-trade band should reduce total variation"
