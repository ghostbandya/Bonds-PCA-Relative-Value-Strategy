"""
06_backtest.py
==============
Strategy backtest engine.

The backtest translates target positions (from 05_signal_generation) into
P&L using daily yield changes. Because we trade bond futures, P&L is
computed via price returns, not yield returns.

P&L logic
---------
  For each instrument i on date t:
    raw_pnl_i(t) = position_i(t-1) × price_return_i(t)

  Where price_return can be:
    - futures_price_return : (px_t - px_{t-1}) / px_{t-1}   [preferred]
    - yield_change_pnl     : -DV01 × position × Δy_i(t)     [fallback]

  Since we normalise positions to ±1 (or ±0.5 in NEUTRAL), P&L is in
  return units per unit notional.

Portfolio P&L
-------------
  Total daily return = mean of individual instrument returns (equal-weight)
  Sharpe = annualised mean / annualised std of daily returns
  Max drawdown from cumulative return series

Regime-conditioned stats are also computed separately for GOOD/NEUTRAL/BAD.

Outputs
-------
  outputs/backtest/
    pnl_daily.csv      — daily P&L per instrument
    pnl_total.csv      — portfolio-level daily & cumulative P&L
    metrics.csv        — summary statistics
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR   = os.path.join(BASE_DIR, "data")

# ── Approximate modified duration (DV01 per 1% yield move) by tenor ───────────
# These are par-bond approximations.  e.g. a 10Y par bond has mod. duration ~9,
# so a 1% rise in yield ≈ 9% fall in price.
# Format: partial-match on the column suffix (case-insensitive).
TENOR_DV01 = {
    "0.5Y": 0.50,  "1Y":  0.98,  "2Y":  1.92,  "3Y":  2.83,
    "5Y":   4.55,  "7Y":  6.20,  "10Y": 8.60,  "15Y": 12.0,
    "20Y": 15.5,   "25Y": 18.0,  "30Y": 19.5,  "40Y": 22.0,
}


def build_dv01_series(columns: pd.Index) -> pd.Series:
    """
    Return a Series of DV01 values aligned to the given column index.
    Columns are expected in the format '<COUNTRY>_<TENOR>' (e.g. 'US_10Y').
    Falls back to 1.0 for unrecognised tenors.
    """
    dv01 = {}
    for col in columns:
        tenor = col.split("_", 1)[-1].upper()   # 'US_10Y' → '10Y'
        dv01[col] = TENOR_DV01.get(tenor, 1.0)
    return pd.Series(dv01)


# ══════════════════════════════════════════════════════════════════════════════
#  P&L computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_futures_returns(futures_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily percentage returns from front-month futures prices.
    """
    ret = futures_prices.pct_change()
    ret.columns = pd.MultiIndex.from_tuples(ret.columns) \
        if not isinstance(ret.columns, pd.MultiIndex) else ret.columns
    return ret.iloc[1:]


def compute_pnl(
    positions:       pd.DataFrame,
    price_returns:   pd.DataFrame = None,
    yield_changes:   pd.DataFrame = None,
    dv01_per_unit:   float = 1.0,
    transaction_cost: float = 0.0002,  # 2 bps round-trip per trade
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute daily instrument-level and portfolio-level P&L.

    Parameters
    ----------
    positions        : (date × instrument) target position from signal generator
    price_returns    : (date × instrument) daily futures price returns [preferred]
    yield_changes    : (date × instrument) daily yield changes (used if no prices)
    dv01_per_unit    : DV01 per unit notional (for yield-change PnL)
    transaction_cost : one-way cost per trade as fraction of notional

    Returns
    -------
    pnl_instrument : DataFrame (date × instrument)
    pnl_portfolio  : DataFrame with columns [daily_ret, cumulative_ret, drawdown]
    """
    # Align positions with returns
    if price_returns is not None:
        # Flatten MultiIndex to match position columns if needed
        if isinstance(price_returns.columns, pd.MultiIndex):
            price_returns.columns = [f"{c[0]}_{c[1]}" for c in price_returns.columns]

        common_dates = positions.index.intersection(price_returns.index)
        common_cols  = positions.columns.intersection(price_returns.columns)
        if len(common_cols) == 0:
            print("  [WARN] No column overlap between positions and futures returns.")
            print(f"    Position cols: {positions.columns[:5].tolist()}")
            print(f"    Return cols:   {price_returns.columns[:5].tolist()}")
            return pd.DataFrame(), pd.DataFrame()

        pos = positions.loc[common_dates, common_cols]
        ret = price_returns.loc[common_dates, common_cols]

        # P&L: previous position × today's return
        pnl_instr = pos.shift(1) * ret

    elif yield_changes is not None:
        common_dates = positions.index.intersection(yield_changes.index)
        common_cols  = positions.columns.intersection(yield_changes.columns)
        if len(common_cols) == 0:
            print("  [WARN] No column overlap between positions and yield changes.")
            print(f"    Position cols: {positions.columns[:5].tolist()}")
            print(f"    Yield-change cols: {yield_changes.columns[:5].tolist()}")
            return pd.DataFrame(), pd.DataFrame()
        pos = positions.loc[common_dates, common_cols]
        dy  = yield_changes.loc[common_dates, common_cols]
        # Duration-neutral P&L (equal dollar-duration per instrument):
        #
        #   pnl_i = -DV01_base × position_i(t-1) × Δresidual_i(t) / 100
        #
        # Instead of weighting by instrument-specific DV01 (which makes 30Y
        # positions 10× more impactful than 2Y), we apply a CONSTANT DV01 equal
        # to the 10Y reference (8.60).  This is equivalent to first scaling each
        # position inversely by its DV01:
        #   pos_adj_i = pos_i × (DV01_base / DV01_i)
        # and then multiplying by DV01_i → the DV01s cancel, leaving DV01_base.
        #
        # Result: all instruments contribute equal price-sensitivity per
        # unit of residual yield move — a truly duration-neutral portfolio.
        DV01_BASE  = 8.60          # 10Y modified duration (reference)
        pnl_instr  = -(pos.shift(1) * dy) * (DV01_BASE / 100.0)
    else:
        raise ValueError("Provide either price_returns or yield_changes.")

    # Transaction costs — charged on position changes (trades)
    trades     = pos.diff().abs()
    tc_instr   = trades * transaction_cost
    pnl_instr -= tc_instr

    pnl_instr  = pnl_instr.dropna(how="all")

    # Portfolio P&L — equal-weight average across active instruments
    n_active       = (pos.shift(1).abs() > 0).sum(axis=1).replace(0, np.nan)
    daily_ret      = pnl_instr.sum(axis=1) / n_active
    daily_ret      = daily_ret.fillna(0)
    cumulative_ret = (1 + daily_ret).cumprod() - 1
    drawdown       = _max_drawdown_series(cumulative_ret)

    pnl_portfolio = pd.DataFrame({
        "daily_ret":      daily_ret,
        "cumulative_ret": cumulative_ret,
        "drawdown":       drawdown,
    })

    return pnl_instr, pnl_portfolio


def _max_drawdown_series(cum_ret: pd.Series) -> pd.Series:
    """Rolling max-drawdown from peak series."""
    wealth = 1 + cum_ret
    peak   = wealth.cummax()
    dd     = (wealth - peak) / peak
    return dd


# ══════════════════════════════════════════════════════════════════════════════
#  Performance metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    pnl_portfolio: pd.DataFrame,
    regime:        pd.Series = None,
    ann_factor:    int = 252,
) -> pd.DataFrame:
    """
    Compute standard performance metrics, overall and per-regime.

    Returns
    -------
    DataFrame with rows = ['Overall', 'GOOD', 'NEUTRAL', 'BAD']
    and columns = [annualised_return, annualised_vol, sharpe, max_drawdown,
                   win_rate, n_days]
    """
    REGIME_NAMES = {0: "GOOD", 1: "NEUTRAL", 2: "BAD"}

    def _stats(daily: pd.Series, label: str) -> dict:
        daily    = daily.dropna()
        ann_ret  = daily.mean() * ann_factor
        ann_vol  = daily.std() * np.sqrt(ann_factor)
        sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
        cum      = (1 + daily).cumprod() - 1
        dd       = _max_drawdown_series(cum).min()
        win_rate = (daily > 0).mean()
        return {
            "label":            label,
            "ann_return_%":     round(ann_ret * 100, 2),
            "ann_vol_%":        round(ann_vol * 100, 2),
            "sharpe":           round(sharpe, 3),
            "max_drawdown_%":   round(dd * 100, 2),
            "win_rate_%":       round(win_rate * 100, 1),
            "n_days":           len(daily),
        }

    rows = [_stats(pnl_portfolio["daily_ret"], "Overall")]

    if regime is not None:
        common = pnl_portfolio.index.intersection(regime.index)
        for r, name in REGIME_NAMES.items():
            mask = regime.loc[common] == r
            sub  = pnl_portfolio.loc[common]["daily_ret"][mask]
            rows.append(_stats(sub, name))

    df = pd.DataFrame(rows).set_index("label")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    signal_dict:      dict,
    regime_dict:      dict,
    futures_prices:   pd.DataFrame = None,
    yield_changes:    pd.DataFrame = None,
    transaction_cost: float = 0.0002,
    save: bool = True,
) -> dict:
    """Full backtest pipeline."""
    print("=" * 50)
    print("  Backtest Pipeline")
    print("=" * 50)

    positions = signal_dict["positions"]

    # Build returns
    price_returns = None
    if futures_prices is not None and not futures_prices.empty:
        print("\n[1] Using futures price returns for P&L …")
        price_returns = compute_futures_returns(futures_prices)
    else:
        print("\n[1] Using yield-change P&L (DV01-weighted) …")

    print("[2] Computing instrument-level P&L …")
    pnl_instr, pnl_port = compute_pnl(
        positions, price_returns=price_returns,
        yield_changes=yield_changes,
        transaction_cost=transaction_cost,
    )

    if pnl_port.empty:
        print("  [ERROR] P&L computation failed — check column alignment.")
        return {}

    print("[3] Computing performance metrics …")
    metrics = compute_metrics(pnl_port, regime=regime_dict.get("regime"))

    print("\n  Performance Summary:")
    print(metrics.to_string())

    if save:
        out = os.path.join(OUTPUT_DIR, "backtest")
        os.makedirs(out, exist_ok=True)
        pnl_instr.to_csv(os.path.join(out, "pnl_daily.csv"))
        pnl_port.to_csv(os.path.join(out, "pnl_total.csv"))
        metrics.to_csv(os.path.join(out, "metrics.csv"))
        print(f"\n  Results saved to {out}/")

    return {
        "pnl_instrument": pnl_instr,
        "pnl_portfolio":  pnl_port,
        "metrics":        metrics,
    }
