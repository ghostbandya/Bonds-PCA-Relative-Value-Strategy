"""
dv01.py
=======
DV01 engine for multi-country sovereign bonds.
Adapted from Jay's dv01.py to handle COUNTRY_TENOR column labels.

Maths (per $100 face per 1bp move):
  sub-1Y bills :  DV01 = n * 100 / (1 + y*n)^2 * 1e-4
  1Y zero      :  DV01 = 100 / (1 + y/2)^3 * 1e-4
  coupon par   :  DV01 = (0.01/y) * [1 - (1 + y/2)^(-2n)]

All four countries use government par yields so these formulas apply.
Input yields must be in PERCENT (e.g. 4.25 = 4.25%) -- we divide by 100 internally.

WHY TIME-VARYING DV01?
A 10Y bond's DV01 at 2% yield (~9.1) is very different from at 5% (~7.8).
Using a fixed approximation (Jay's table or our old constant 8.60) introduces
systematic errors that compound across 23 instruments. The formula approach
computes the correct DV01 for each instrument on each date from actual yields.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import tenor_to_years

ZERO_YIELD_FLOOR: float = 1e-6   # guard for ZIRP / negative rate environments


def dv01_per_100(y_pct: float | np.ndarray, n: float) -> float | np.ndarray:
    """
    DV01 per $100 face for yield y_pct (in percent) and tenor n (years).

    Accepts scalar or array. NaN yields propagate to NaN DV01.

    Parameters
    ----------
    y_pct : yield in percent (e.g. 4.25 for 4.25%)
    n     : maturity in years

    Returns
    -------
    DV01 in dollars per $100 face per 1bp move
    """
    scalar = np.ndim(y_pct) == 0
    raw = np.asarray(y_pct, dtype=float)
    nan_mask = np.isnan(raw)

    # Convert percent -> decimal, clip at floor to handle ZIRP
    y = np.clip(raw / 100.0, ZERO_YIELD_FLOOR, None)

    if n < 1.0:
        # Sub-1Y bill: simple interest, single cash flow
        out = n * 100.0 / (1.0 + y * n) ** 2 * 1e-4
    elif np.isclose(n, 1.0):
        # 1Y zero-coupon (semi-annual basis)
        out = 100.0 / (1.0 + y / 2.0) ** 3 * 1e-4
    else:
        # Coupon par bond (semi-annual, n > 1)
        # For a par bond price = 100 always, so:
        #   DV01 = (0.01/y) * [1 - (1 + y/2)^{-2n}]
        out = (0.01 / y) * (1.0 - (1.0 + y / 2.0) ** (-2.0 * n))

    # Restore NaN where input was NaN
    if np.ndim(out) > 0:
        out = np.where(nan_mask, np.nan, out)
    elif nan_mask:
        return np.nan

    return float(out) if scalar else out


def build_dv01_matrix(
    yields_clean: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a time-varying DV01 matrix (date x instrument).

    Parameters
    ----------
    yields_clean : MultiIndex-column DataFrame (country, tenor) from _02_data_prep.
                   Yields are in percent.
    columns      : optional list of COUNTRY_TENOR columns to compute (e.g. ["US_10Y"]).
                   If None, computes for all available instruments.

    Returns
    -------
    DataFrame (date x instrument) of DV01 values ($ per $100 face per 1bp).

    Usage
    -----
    The DV01 matrix is used by:
      - weights.py: to convert yield-space book g -> tradable notionals q = -D^{-1} g
      - costs.py (dv01_bp model): cost_i = half_spread_i * DV01_i * |delta_q_i|
    """
    # Flatten MultiIndex -> COUNTRY_TENOR if needed
    if isinstance(yields_clean.columns, pd.MultiIndex):
        flat = yields_clean.copy()
        flat.columns = [f"{c}_{t}" for c, t in yields_clean.columns]
    else:
        flat = yields_clean.copy()

    if columns is None:
        columns = list(flat.columns)

    result = {}
    for col in columns:
        if col not in flat.columns:
            continue
        n = tenor_to_years(col)
        y_series = flat[col]
        result[col] = dv01_per_100(y_series.values, n)

    df = pd.DataFrame(result, index=flat.index)
    return df


def get_dv01_row(dv01_matrix: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """
    Return the DV01 row for a given date, using the nearest available date
    if the exact date is not present (e.g. weekend / holiday).
    """
    if date in dv01_matrix.index:
        return dv01_matrix.loc[date]
    # Find nearest prior date
    prior = dv01_matrix.index[dv01_matrix.index <= date]
    if len(prior) == 0:
        return dv01_matrix.iloc[0]
    return dv01_matrix.loc[prior[-1]]


def notionals_from_g(g: np.ndarray, dv01_row: np.ndarray) -> np.ndarray:
    """
    Convert yield-space book g ($/bp) to tradable notionals q.

    Canonical chain (Jay's §3.5.1):
      q = -D^{-1} g

    where D is the diagonal of per-instrument DV01 ($/bp per $100 face).
    The sign convention: positive g means we are long yield (short the bond),
    so positive q means short the bond notional.

    Parameters
    ----------
    g        : (N,) yield-exposure vector in $/bp (the portfolio book)
    dv01_row : (N,) per-instrument DV01 values ($/100 face/bp)

    Returns
    -------
    q : (N,) notional vector in $100 face units
    """
    d = np.asarray(dv01_row, dtype=float)
    # Avoid division by zero (should not happen with valid DV01s)
    d = np.where(d == 0, np.nan, d)
    return -np.asarray(g, dtype=float) / d
