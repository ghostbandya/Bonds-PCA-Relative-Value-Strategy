"""
06_backtest.py
==============
Strategy backtest engine — translates positions into P&L and performance metrics.

═══════════════════════════════════════════════════════════════
P&L CALCULATION — TWO MODES
═══════════════════════════════════════════════════════════════
Mode A — Futures price returns  (preferred if futures data available)
  pnl_i(t) = position_i(t-1) × [px_i(t) - px_i(t-1)] / px_i(t-1)
  This is a mark-to-market return on the notional invested.

Mode B — Yield-change P&L  (used in this project — no futures prices)
  Bond price ≈ -DV01 × Δyield  (first-order duration approximation)
  pnl_i(t) = -DV01_base × position_i(t-1) × Δresidual_i(t) / 100

  CRITICAL DESIGN CHOICE — CONSTANT DV01_BASE:
  Instead of each instrument's own DV01 (which varies from ~1 for 1Y
  to ~19 for 30Y), we use a single constant DV01_BASE = 8.60 (the 10Y
  reference).  This is DURATION-NEUTRAL sizing:

    Effective weight = position × DV01_base / DV01_instrument

  A long in US_30Y and a long in US_1Y both have the same price
  sensitivity to a 1 bps yield move.  Without this, 30Y positions
  would dominate P&L even if position sizes were equal.

  Note: we apply DV01 to RESIDUAL changes (not raw yield changes).
  This removes systematic factor P&L (PC1/PC2/PC3 moves) and isolates
  the idiosyncratic spread reversion — the actual source of alpha.

═══════════════════════════════════════════════════════════════
TRANSACTION COSTS
═══════════════════════════════════════════════════════════════
2 bps (0.0002) charged on every position change (one-way).
This is conservative for IG govt bonds / futures (typical bid-ask
for on-the-run Treasuries ≈ 0.25–0.5 bps; futures ≈ 0.1 bps).
The conservative estimate accounts for market impact and slippage
in a live implementation.

═══════════════════════════════════════════════════════════════
PORTFOLIO AGGREGATION
═══════════════════════════════════════════════════════════════
Daily portfolio return = sum of instrument P&Ls / number of active positions
(equal-weight, not equal-notional, since all positions are already DV01-normalised)

Regime-conditioned stats are computed separately by slicing the daily
returns into GOOD / NEUTRAL / BAD subsets.  The GOOD Sharpe is the most
meaningful metric — it measures alpha in the environment where the strategy
is designed to work.
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
        # ── Duration-neutral P&L ─────────────────────────────────────────
        # pnl_i(t) = -DV01_base × pos_i(t-1) × Δresidual_i(t) / 100
        #
        # The /100 converts yield changes from percent to decimal
        # (e.g. 0.05 % move → 0.0005 decimal → ~4.3 bps × 8.60 ≈ 0.037%).
        #
        # Why a CONSTANT DV01_BASE = 8.60 (10Y reference)?
        # If we used each instrument's own DV01, a 1 bps residual move on
        # US_30Y (DV01 ≈ 19.5) would generate 2.3× the P&L of the same
        # move on US_5Y (DV01 ≈ 4.55), even with identical position sizes.
        # The 30Y would dominate even if we had equal confidence in both signals.
        # Using DV01_BASE equalises price-sensitivity across all instruments
        # so every signal contributes proportionally to the portfolio.
        DV01_BASE  = 8.60          # 10Y modified duration (reference)
        pnl_instr  = -(pos.shift(1) * dy) * (DV01_BASE / 100.0)
    else:
        raise ValueError("Provide either price_returns or yield_changes.")

    # ── Transaction costs ────────────────────────────────────────────────
    # pos.diff().abs() captures the size of every position change:
    #   0 → no trade (hold), no cost
    #   1 → new trade opened (from 0 to ±1), or full flip (from -1 to +1)
    #   2 → rare; in practice positions go 0→±1 or ±1→0
    # Default 2 bps (0.0002) is conservative relative to actual govt bond
    # bid-ask spreads but captures realistic execution friction.
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


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST V2 — Methods 1 / 2 / 3 + vol-targeting + no-trade band
#  Adapted from Jay's backtest engine for multi-country universe
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest_v2(
    pca_results:    dict,
    signal_dict:    dict,
    yields_clean:   "pd.DataFrame",
    yield_changes:  "pd.DataFrame",
    regime_dict:    dict,
    config:         "StrategyConfig" = None,
    save:           bool = True,
    out_subdir:     str  = "backtest_v2",
) -> dict:
    """
    Full backtest using Methods 1/2/3 portfolio construction + vol-targeting
    + no-trade band + proper DV01-based costs.

    This is the multi-country adaptation of Jay's backtest pipeline.

    How it differs from run_backtest (v1):
    ──────────────────────────────────────
    v1: Uses binary ±1/±0 positions from S-score thresholds.
        P&L = -DV01_BASE (constant 8.60) × position × Δresidual
    v2: Computes a continuous yield-space book g_t via Method 1/2/3.
        P&L = g_t · Δy_t  (dot product over all instruments)
        Costs via DV01_bp half-spread model.
        Vol-targeting scales the book daily.
        No-trade band reduces turnover.

    Parameters
    ----------
    pca_results   : dict from rolling_pca() — must include 'residuals', 'loadings'
    signal_dict   : dict from run_signals() — must include 's_scores'
    yields_clean  : MultiIndex DataFrame of yield levels (for DV01 computation)
    yield_changes : flat-column DataFrame of daily yield changes (COUNTRY_TENOR)
    regime_dict   : dict from detect_regimes()
    config        : StrategyConfig; defaults to StrategyConfig() (Method 3, LW)
    save          : save outputs to disk
    out_subdir    : subdirectory under outputs/

    Returns
    -------
    dict: pnl_instrument, pnl_portfolio, metrics, book_g, hold_mask
    """
    from .config import StrategyConfig, get_split_dates, TRADE_COUNTRIES
    from .covariance import estimate_cov, build_rolling_cov
    from .dv01 import build_dv01_matrix, get_dv01_row
    from .weights import (method1_geometric, method2_minvar, method3_meanvar,
                          calibrate_gamma, vol_scale, projection_matrix)
    from .costs import compute_daily_costs
    from .rebalance import apply_no_trade_band

    if config is None:
        config = StrategyConfig()

    print("=" * 60)
    print(f"  Backtest V2 — Method {config.method} | Cov: {config.covariance} "
          f"| Band: {config.no_trade_band} | Vol-target: {config.vol_target}")
    print("=" * 60)

    # ── 1. Align instruments ────────────────────────────────────────────────
    residuals  = pca_results["residuals"]
    s_scores   = signal_dict["s_scores"]
    regime     = regime_dict["regime"]

    # Filter to tradeable instruments
    trade_cols = [c for c in residuals.columns
                  if c.split("_")[0] in config.trade_countries]
    residuals  = residuals[trade_cols]
    s_scores   = s_scores.reindex(columns=trade_cols)

    # Align yield changes to same columns
    yc = yield_changes.copy()
    if isinstance(yc.columns, pd.MultiIndex):
        yc.columns = [f"{c}_{t}" for c, t in yc.columns]
    trade_yc = yc[[c for c in trade_cols if c in yc.columns]]

    common_dates = (residuals.index
                    .intersection(s_scores.index)
                    .intersection(regime.index)
                    .intersection(trade_yc.index))
    common_dates = common_dates.sort_values()

    residuals = residuals.loc[common_dates]
    s_scores  = s_scores.loc[common_dates]
    trade_yc  = trade_yc.loc[common_dates]
    N         = len(trade_cols)

    # ── 2. 80/20 train / test split ─────────────────────────────────────────
    # NOTE: splits are computed here only to get train_end for gamma calibration
    # reporting; the authoritative split was already set by main.py before PCA.
    splits = get_split_dates(common_dates)
    train_end = splits["train"][1]
    print(f"\n  Split: Train → {splits['train'][1].date()} | "
          f"Test  → {splits['test'][1].date()}")

    # ── 3. Build DV01 matrix ────────────────────────────────────────────────
    print("\n[1] Building DV01 matrix ...")
    dv01_matrix = build_dv01_matrix(yields_clean, columns=trade_cols)
    dv01_matrix = dv01_matrix.reindex(common_dates, method="ffill")
    print(f"    DV01 matrix: {dv01_matrix.shape}")

    # ── 4. Rolling covariance matrices ─────────────────────────────────────
    print(f"\n[2] Building rolling covariance ({config.covariance}, "
          f"window={config.pca_window}) ...")
    sigma_dict = build_rolling_cov(
        trade_yc, window=config.pca_window, method=config.cov_method)
    print(f"    Covariance available from: "
          f"{min(sigma_dict.keys()).date() if sigma_dict else 'N/A'}")

    # ── 5. Rolling PCA loadings ─────────────────────────────────────────────
    # Use saved loadings from pca_results if available; otherwise re-derive
    # from the factor_scores + residuals relationship.
    loadings_dict = pca_results.get("loadings", {})
    # Build V_dict: date -> (k, N) array for tradeable instruments only
    V_dict = {}
    for date, beta_df in loadings_dict.items():
        if date not in common_dates:
            continue
        # beta_df columns = PC1..PCk, index = instruments
        available = [c for c in trade_cols if c in beta_df.index]
        if len(available) < N:
            continue
        V = beta_df.loc[available].values.T  # (k, N)
        V_dict[date] = V

    print(f"    Loadings available for {len(V_dict)} dates")

    # ── 6. Calibrate gamma (Method 3 only) ─────────────────────────────────
    gamma = config.gamma
    if config.method == 3 and gamma is None:
        print("\n[3] Calibrating gamma on TRAIN split ...")
        train_mask = common_dates <= train_end
        train_dates = common_dates[train_mask]
        if len(train_dates) > 100 and len(sigma_dict) > 0:
            # Use mean Sigma and mean V over training period for calibration
            sig_train = np.mean([sigma_dict[d] for d in train_dates
                                  if d in sigma_dict], axis=0)
            V_train = np.mean([V_dict[d] for d in train_dates
                               if d in V_dict], axis=0) if V_dict else None
            if V_train is not None and sig_train is not None:
                gamma = calibrate_gamma(
                    trade_yc.loc[train_dates],
                    V_train,
                    s_scores.loc[train_dates],
                    sig_train,
                )
                print(f"    Calibrated gamma = {gamma:.3f}")
            else:
                gamma = 500.0
                print(f"    Insufficient data for calibration; using gamma = {gamma}")
        else:
            gamma = 500.0
            print(f"    Using default gamma = {gamma}")

    # ── 7. Build daily portfolio book g_t ───────────────────────────────────
    print(f"\n[4] Computing Method-{config.method} portfolio book ...")

    # Get a representative V for dates without loadings (use nearest prior)
    # Fallback V: identity projection (no factor hedge) — should rarely be used
    V_fallback = np.zeros((3, N))
    Sigma_fallback = np.eye(N) * 1e-4

    book_g = pd.DataFrame(0.0, index=common_dates, columns=trade_cols)

    for date in common_dates:
        s_row = s_scores.loc[date].values
        if np.all(np.isnan(s_row)):
            continue

        # Regime filter
        r = regime.loc[date] if date in regime.index else 0
        if r == 2:  # BAD: flat
            continue

        size = 1.0 if r == 0 else 0.5  # NEUTRAL: half size

        # Get loadings and covariance for this date
        V     = V_dict.get(date, V_fallback)
        Sigma = sigma_dict.get(date, Sigma_fallback)

        if config.method == 1:
            g = method1_geometric(s_row, V)
        elif config.method == 2:
            g = method2_minvar(s_row, V, Sigma)
        else:  # method 3
            g = method3_meanvar(s_row, V, Sigma, gamma=gamma)

        book_g.loc[date] = g * size

    print(f"    Book non-zero on {(book_g.abs().sum(axis=1) > 0).mean()*100:.1f}% of days")

    # ── 8. Vol-targeting ────────────────────────────────────────────────────
    if config.vol_target:
        print("\n[5] Applying vol-targeting ...")
        # Compute proxy daily returns on unscaled book
        proxy_ret = (book_g.shift(1) * trade_yc).sum(axis=1).fillna(0)
        k_t = vol_scale(proxy_ret)
        # Apply scaling
        scaled_G = book_g.multiply(k_t, axis=0)
        print(f"    Avg leverage scalar k_t: {np.nanmean(k_t):.2f} "
              f"(max: {np.nanmax(k_t):.2f})")
    else:
        scaled_G = book_g

    # ── 9. No-trade band ────────────────────────────────────────────────────
    if config.no_trade_band > 0:
        print(f"\n[6] Applying no-trade band (tau={config.no_trade_band}) ...")
        exec_G, hold_mask = apply_no_trade_band(
            scaled_G, sigma_dict, V_dict, tau=config.no_trade_band)
        hold_frac = hold_mask.mean() * 100
        print(f"    Hold fraction: {hold_frac:.1f}%")
    else:
        exec_G    = scaled_G
        hold_mask = pd.Series(False, index=common_dates)

    # ── 10. P&L computation ─────────────────────────────────────────────────
    print("\n[7] Computing P&L ...")
    # P&L: g_{t-1} . Δy_t  (previous book × today's yield changes)
    # Yield changes are in percent; we keep them as-is for return units.
    pnl_instr = exec_G.shift(1) * trade_yc
    pnl_instr = pnl_instr.dropna(how="all")

    # Transaction costs
    costs = compute_daily_costs(
        exec_G,
        dv01_matrix,
        cost_model=config.cost_model,
        cost_per_contract_usd=config.cost_per_contract_usd,
        capital_usd=config.capital_usd,
    )
    costs = costs.reindex(pnl_instr.index, fill_value=0.0)

    # Portfolio P&L (equal-weight across active instruments)
    n_active  = (exec_G.shift(1).abs() > 0).sum(axis=1).replace(0, np.nan)
    daily_ret = pnl_instr.sum(axis=1) / n_active.reindex(pnl_instr.index)
    daily_ret = daily_ret.fillna(0) - costs.reindex(daily_ret.index, fill_value=0)

    cum_ret  = (1 + daily_ret).cumprod() - 1
    drawdown = _max_drawdown_series(cum_ret)

    pnl_portfolio = pd.DataFrame({
        "daily_ret":      daily_ret,
        "cumulative_ret": cum_ret,
        "drawdown":       drawdown,
    })

    # ── 11. Metrics ──────────────────────────────────────────────────────────
    metrics = compute_metrics(pnl_portfolio, regime=regime_dict.get("regime"))

    print("\n  Performance Summary (V2):")
    print(metrics.to_string())

    # Per-split metrics
    for split_name, (s_start, s_end) in splits.items():
        split_mask = (daily_ret.index >= s_start) & (daily_ret.index <= s_end)
        sub = pnl_portfolio[split_mask]
        if len(sub) > 10:
            m = compute_metrics(sub)
            print(f"\n  {split_name.upper()} split ({s_start.date()} → {s_end.date()}):")
            print(f"    Sharpe: {m.loc['Overall','sharpe']:.3f}  "
                  f"Return: {m.loc['Overall','ann_return_%']:.2f}%  "
                  f"MaxDD: {m.loc['Overall','max_drawdown_%']:.2f}%")

    # ── 12. Save ─────────────────────────────────────────────────────────────
    if save:
        out = os.path.join(OUTPUT_DIR, out_subdir)
        os.makedirs(out, exist_ok=True)
        pnl_instr.to_csv(os.path.join(out, "pnl_daily.csv"))
        pnl_portfolio.to_csv(os.path.join(out, "pnl_total.csv"))
        metrics.to_csv(os.path.join(out, "metrics.csv"))
        exec_G.to_csv(os.path.join(out, "book_g.csv"))
        hold_mask.to_csv(os.path.join(out, "hold_mask.csv"))
        print(f"\n  Results saved to outputs/{out_subdir}/")

    return {
        "pnl_instrument":  pnl_instr,
        "pnl_portfolio":   pnl_portfolio,
        "metrics":         metrics,
        "book_g":          exec_G,
        "hold_mask":       hold_mask,
        "gamma":           gamma,
        "splits":          splits,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SPLIT-AWARE METRICS — train / val / test reporting
# ══════════════════════════════════════════════════════════════════════════════

def metrics_by_split(
    pnl_portfolio: pd.DataFrame,
    splits: dict,
    regime: pd.Series = None,
) -> dict:
    """
    Compute performance metrics for each chronological split independently.

    Parameters
    ----------
    pnl_portfolio : DataFrame with 'daily_ret' column (from compute_pnl or run_backtest_v2)
    splits        : dict from get_split_dates() — keys: 'train', 'val', 'test'
    regime        : optional regime Series for within-split regime conditioning

    Returns
    -------
    dict: {'train': metrics_df, 'val': metrics_df, 'test': metrics_df}
    """
    result = {}
    for split_name, (s_start, s_end) in splits.items():
        mask = (pnl_portfolio.index >= s_start) & (pnl_portfolio.index <= s_end)
        sub  = pnl_portfolio[mask]
        if len(sub) < 20:
            continue
        sub_regime = regime.loc[mask] if regime is not None else None
        result[split_name] = compute_metrics(sub, regime=sub_regime)
    return result


def print_strategy_table(all_results: dict, splits: dict) -> None:
    """
    Print a unified Training / Test comparison table for all strategies.

    Parameters
    ----------
    all_results : dict {strategy_name -> {'pnl_portfolio': DataFrame, ...}}
    splits      : dict from get_split_dates() — keys: 'train', 'test'
    """
    split_names = list(splits.keys())   # ['train', 'test']

    # Header
    print("\n" + "=" * 72)
    print("  STRATEGY RESULTS — Training vs Test")
    print("=" * 72)
    first_pnl = list(all_results.values())[0]["pnl_portfolio"]
    for split_name, (s_start, s_end) in splits.items():
        n_days = int(((first_pnl.index >= s_start) & (first_pnl.index <= s_end)).sum())
        label  = "TRAINING" if split_name == "train" else "TEST (holdout)"
        print(f"  {label:<16}: {s_start.date()} → {s_end.date()} ({n_days:,} days)")
    print(f"  HMM regime fitted on TRAINING only (80%) → applied forward to TEST (20%)")
    print()

    # Column headers
    col_labels = ["TRAINING", "TEST"]
    header = f"  {'Strategy':<24}"
    for lbl in col_labels:
        header += f"  {lbl:<22}"
    print(header)
    print(f"  {'':<24}" + "  " +
          "  ".join([f"{'Sharpe':>6} {'MaxDD%':>7} {'Ret%':>6}"] * len(split_names)))
    print("  " + "-" * 70)

    # Rows
    for name, res in all_results.items():
        if "pnl_portfolio" not in res:
            continue
        split_metrics = metrics_by_split(res["pnl_portfolio"], splits)
        row = f"  {name:<24}"
        for s in split_names:
            if s not in split_metrics:
                row += f"  {'N/A':>6} {'N/A':>7} {'N/A':>6}"
                continue
            m = split_metrics[s].loc["Overall"]
            row += (f"  {m['sharpe']:>6.3f} "
                    f"{m['max_drawdown_%']:>7.2f} "
                    f"{m['ann_return_%']:>6.2f}")
        print(row)

    print("=" * 72)
    print("  Columns: Sharpe | Max Drawdown (%) | Annualised Return (%)")
    print("  HMM regime model fitted on TRAINING only — no look-ahead bias.")
    print("  TEST is the honest evaluation — the model never saw this data.")
    print("=" * 72 + "\n")
