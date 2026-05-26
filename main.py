"""
main.py
=======
Orchestrator — runs the full pipeline end-to-end.

Steps
-----
  1. Fetch multi-country sovereign yields  (01_data_fetch)
  2. Clean & align data                    (02_data_prep)
  3. Rolling PCA                           (03_rolling_pca)
  4. Regime detection (3 regimes)          (04_regime_detection)
  5. S-score signal generation             (05_signal_generation)
  6. PCA backtest                          (06_backtest)
  7. Carry + Roll-Down strategy            (08_carry_signal)
  8. Carry backtest                        (06_backtest reused)
  9. Visualisations                        (07_visualisation)

Usage
-----
  python main.py
  python main.py --skip-fetch
  python main.py --mode country --country US --skip-fetch
  python main.py --regime-method rules --skip-fetch
  python main.py --start 2010-01-01 --skip-fetch
"""

import os
import sys
import argparse
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src._01_data_fetch        import fetch_all, build_combined
from src._02_data_prep         import run_prep
from src._03_rolling_pca       import rolling_pca, save_pca_results
from src._04_regime_detection  import detect_regimes, save_regime_results
from src._05_signal_generation import run_signals
from src._06_backtest          import run_backtest, compute_pnl, compute_metrics
from src._07_visualisation     import run_all_plots
from src._08_carry_signal      import run_carry


def parse_args():
    p = argparse.ArgumentParser(description="PCA Bond Strategy -- full pipeline")
    p.add_argument("--start",           default="2000-01-01")
    p.add_argument("--end",             default=None)
    p.add_argument("--countries",       nargs="+", default=["US", "DE", "UK", "JP"],
                   help="Countries in PCA (all 4 for better factors)")
    p.add_argument("--trade-countries", nargs="+", default=["US", "DE", "UK"],
                   help="Countries to trade (JP excluded: BoJ YCC suppresses mean reversion)")
    p.add_argument("--skip-fetch",      action="store_true")
    p.add_argument("--skip-backtest",   action="store_true")
    p.add_argument("--mode",            choices=["cross", "country"], default="cross")
    p.add_argument("--country",         default="US")
    p.add_argument("--corr-window",     type=int, default=252)
    p.add_argument("--resid-window",    type=int, default=60)
    p.add_argument("--k",               type=int, default=3)
    p.add_argument("--regime-method",   choices=["hmm", "rules"], default="hmm")
    p.add_argument("--smooth",          type=int, default=21)
    p.add_argument("--s-bo",            type=float, default=1.25)
    p.add_argument("--s-bc",            type=float, default=0.75)
    p.add_argument("--s-so",            type=float, default=1.25)
    p.add_argument("--s-sc",            type=float, default=0.50)
    p.add_argument("--tc",              type=float, default=0.0002)
    p.add_argument("--no-plots",        action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  PCA Bond Strategy -- Applied Quant Macro")
    print("=" * 60)
    print(f"  Countries (PCA)   : {args.countries}")
    print(f"  Countries (trade) : {args.trade_countries}")
    print(f"  Mode        : {args.mode}")
    print(f"  Corr window : {args.corr_window} | Resid window : {args.resid_window}")
    print(f"  Regime      : {args.regime_method} | Smooth : {args.smooth}")
    print(f"  k (PCs)     : {args.k}")
    print("=" * 60 + "\n")

    # Step 1: Data fetch
    if not args.skip_fetch:
        print("\n[STEP 1] Fetching multi-country sovereign yields ...")
        results  = fetch_all(countries=args.countries, start=args.start, end=args.end)
        combined = build_combined(results)
    else:
        print("\n[STEP 1] Skipping data fetch (--skip-fetch).")

    # Step 2: Data prep
    print("\n[STEP 2] Preparing data ...")
    prep = run_prep(start=args.start, end=args.end, countries=args.countries)

    if args.mode == "cross":
        changes = prep["cross_market"]
    else:
        changes_multi = prep["yield_changes"]
        changes = changes_multi.xs(args.country, axis=1, level="country").dropna(how="all")

    print(f"  Input to PCA: {changes.shape[0]} rows x {changes.shape[1]} instruments")

    # Step 3: Rolling PCA
    print("\n[STEP 3] Running rolling PCA ...")
    pca_results = rolling_pca(
        changes, k=args.k,
        corr_window=args.corr_window, resid_window=args.resid_window,
    )
    save_pca_results(pca_results)

    # Step 4: Regime detection
    print("\n[STEP 4] Detecting regimes ...")
    regime_dict = detect_regimes(
        pca_results, method=args.regime_method, smooth_window=args.smooth,
    )
    save_regime_results(regime_dict)

    # Step 5: Signal generation
    # Filter residuals to tradeable countries only.
    # JP stays in PCA for better factor estimation but is excluded from trading
    # because BoJ YCC 2012-2024 suppressed free-float mean reversion in JGBs.
    print("\n[STEP 5] Generating signals ...")
    pca_for_signals = dict(pca_results)
    trade_cols = [c for c in pca_results["residuals"].columns
                  if c.split("_")[0] in args.trade_countries]
    pca_for_signals["residuals"] = pca_results["residuals"][trade_cols]
    print(f"  Tradeable instruments: {trade_cols}")

    signal_dict = run_signals(
        pca_for_signals, regime_dict,
        resid_window=args.resid_window,
        s_bo=args.s_bo, s_bc=args.s_bc,
        s_so=args.s_so, s_sc=args.s_sc,
    )

    # Step 6: PCA backtest
    backtest_dict = {}
    if not args.skip_backtest:
        print("\n[STEP 6] Running PCA backtest ...")
        # Factor-neutral P&L: use daily diffs of cumulated residuals, not raw yield changes.
        # This removes systematic factor moves and isolates the idiosyncratic spread.
        daily_residuals = pca_for_signals["residuals"].diff().iloc[1:]
        backtest_dict = run_backtest(
            signal_dict, regime_dict,
            futures_prices=None,
            yield_changes=daily_residuals,
            transaction_cost=args.tc,
        )
    else:
        print("\n[STEP 6] Skipping backtest (--skip-backtest).")

    # Step 7: Carry + Roll-Down strategy
    print("\n[STEP 7] Running Carry + Roll-Down strategy ...")
    carry_dict = run_carry(prep["yields_clean"], regime_dict)

    carry_backtest_dict = {}
    if not args.skip_backtest:
        print("\n[STEP 7b] Running Carry backtest ...")
        carry_positions = carry_dict["positions"]
        trade_instr = [c for c in carry_positions.columns
                       if c.split("_")[0] in args.trade_countries]
        carry_positions_trade = carry_positions[trade_instr]

        # Carry uses raw yield changes (carry IS the systematic level signal;
        # using factor-neutral residuals would cancel the carry itself)
        yc = prep["yield_changes"]
        if isinstance(yc.columns, pd.MultiIndex):
            yc = yc.copy()
            yc.columns = [f"{c}_{t}" for c, t in yc.columns]
        carry_yield_changes = yc[[c for c in trade_instr if c in yc.columns]]

        carry_pnl_instr, carry_pnl_port = compute_pnl(
            carry_positions_trade,
            yield_changes=carry_yield_changes,
            transaction_cost=args.tc,
        )
        carry_metrics = compute_metrics(carry_pnl_port, regime=regime_dict.get("regime"))

        print("\n  Carry Strategy Performance:")
        print(carry_metrics.to_string())

        out = os.path.join(BASE_DIR, "outputs", "carry_backtest")
        os.makedirs(out, exist_ok=True)
        carry_pnl_instr.to_csv(os.path.join(out, "carry_pnl_daily.csv"))
        carry_pnl_port.to_csv(os.path.join(out, "carry_pnl_total.csv"))
        carry_metrics.to_csv(os.path.join(out, "carry_metrics.csv"))
        print(f"  Carry results saved to {out}/")

        carry_backtest_dict = {
            "pnl_instrument": carry_pnl_instr,
            "pnl_portfolio":  carry_pnl_port,
            "metrics":        carry_metrics,
        }

    # Step 8: Visualisations
    if not args.no_plots:
        print("\n[STEP 8] Generating plots ...")
        run_all_plots(
            yields_clean=prep["yields_clean"],
            pca_results=pca_results,
            regime_dict=regime_dict,
            signal_dict=signal_dict,
            backtest_dict=backtest_dict if backtest_dict else None,
            carry_backtest_dict=carry_backtest_dict if carry_backtest_dict else None,
        )

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print(f"  Outputs in: {os.path.join(BASE_DIR, 'outputs')}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
