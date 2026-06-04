"""
main.py
=======
Orchestrator — runs the full pipeline end-to-end with a single
80/20 train / test split applied consistently to all strategies.

The split is computed ONCE from the raw yield-changes index before
any modelling, then passed into PCA (to freeze loadings after train
end) and into the HMM regime detector (to fit only on training data).
Neither the PCA nor the regime model ever re-uses the full dataset.

Steps
-----
  1. Fetch multi-country sovereign yields  (01_data_fetch)
  2. Clean & align data                    (02_data_prep)
  3. Compute 80/20 train / test split dates (ONCE, before any modelling)
  4. Rolling PCA — loadings frozen after train_end  (03_rolling_pca)
  5. Regime detection — HMM fitted on train only    (04_regime_detection)
  6. Signal generation                     (05_signal_generation)
  7. Run all strategies on the same split:
       a. PCA OU S-score   (original threshold approach)
       b. PCA Z-score M1   (geometric projection)     [--v2 only]
       c. PCA Z-score M2   (min-variance KKT)         [--v2 only]
       d. PCA Z-score M3   (mean-variance, LW cov)    [--v2 only]
       e. Carry + Roll-Down
  8. Print unified train / test comparison table
  9. Visualisations                        (07_visualisation)

Usage
-----
  python main.py --skip-fetch
  python main.py --skip-fetch --v2
  python main.py --skip-fetch --v2 --no-trade-band 0.75
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
from src._06_backtest          import (run_backtest, run_backtest_v2,
                                        compute_pnl, compute_metrics,
                                        print_strategy_table, metrics_by_split)
from src._07_visualisation     import run_all_plots
from src._08_carry_signal      import run_carry
from src.config                import StrategyConfig, get_split_dates, get_train_end


def parse_args():
    p = argparse.ArgumentParser(description="PCA Bond Strategy -- full pipeline")
    p.add_argument("--start",           default="1990-01-01")
    p.add_argument("--end",             default=None)
    p.add_argument("--countries",       nargs="+", default=["US", "DE", "UK", "JP"])
    p.add_argument("--trade-countries", nargs="+", default=["US", "DE", "UK"])
    p.add_argument("--skip-fetch",      action="store_true")
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
    # ── V2 parameters ───────────────────────────────────────────────────────
    p.add_argument("--v2",              action="store_true",
                   help="Also run Methods 1/2/3 (Jay's approach) alongside OU + Carry")
    p.add_argument("--covariance",      choices=["sample", "ewma", "lw"], default="lw")
    p.add_argument("--z-window",        type=int, default=63)
    p.add_argument("--no-trade-band",   type=float, default=0.0)
    p.add_argument("--no-vol-target",   action="store_true")
    p.add_argument("--cost-model",      choices=["flat", "dv01_bp"], default="dv01_bp")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  PCA Bond Strategy — Applied Quant Macro")
    print("=" * 60)
    print(f"  Countries (PCA)   : {args.countries}")
    print(f"  Countries (trade) : {args.trade_countries}")
    print(f"  Corr window : {args.corr_window} | Resid window : {args.resid_window}")
    print(f"  Regime      : {args.regime_method} | k (PCs) : {args.k}")
    if args.v2:
        print(f"  V2 methods  : M1, M2, M3 | Cov: {args.covariance} "
              f"| Band: {args.no_trade_band} | Cost: {args.cost_model}")
    print("=" * 60 + "\n")

    # ── Step 1: Data fetch ──────────────────────────────────────────────────
    if not args.skip_fetch:
        print("\n[STEP 1] Fetching multi-country sovereign yields ...")
        results  = fetch_all(countries=args.countries, start=args.start, end=args.end)
        combined = build_combined(results)
    else:
        print("\n[STEP 1] Skipping data fetch (--skip-fetch).")

    # ── Step 2: Data prep ───────────────────────────────────────────────────
    print("\n[STEP 2] Preparing data ...")
    prep = run_prep(start=args.start, end=args.end, countries=args.countries)

    if args.mode == "cross":
        changes = prep["cross_market"]
    else:
        changes_multi = prep["yield_changes"]
        changes = changes_multi.xs(args.country, axis=1, level="country").dropna(how="all")

    print(f"  Input to PCA: {changes.shape[0]} rows x {changes.shape[1]} instruments")

    # ── Step 3: Compute split dates ONCE — before any modelling ────────────────
    # The split boundary (train_end) is the only date the models need.
    # PCA freezes loadings after train_end; HMM fits only on data up to train_end.
    # We NEVER re-compute splits after PCA or regime detection — doing so would
    # shift the boundary to a different index length and break the train/test
    # accounting.
    print("\n[STEP 3] Computing 80/20 train / test split ...")
    splits    = get_split_dates(changes.index)
    train_end = get_train_end(changes.index)
    for name, (s, e) in splits.items():
        n = int(((changes.index >= s) & (changes.index <= e)).sum())
        print(f"  {name.upper():10s}: {s.date()} → {e.date()} ({n:,} days)")
    print(f"  PCA loadings frozen after {train_end.date()} (test period uses frozen model)")
    print(f"  HMM will be fitted on TRAIN data only (up to {train_end.date()})")

    # ── Step 4: Rolling PCA ─────────────────────────────────────────────────
    print("\n[STEP 4] Running rolling PCA ...")
    pca_results = rolling_pca(
        changes, k=args.k,
        corr_window=args.corr_window, resid_window=args.resid_window,
        train_end=train_end,
    )
    save_pca_results(pca_results)

    # ── Step 5: Regime detection (HMM fitted on train only) ─────────────────
    print("\n[STEP 5] Detecting regimes ...")
    regime_dict = detect_regimes(
        pca_results, method=args.regime_method, smooth_window=args.smooth,
        train_end=train_end,
    )
    save_regime_results(regime_dict)

    # ── Step 6: Signal generation ───────────────────────────────────────────
    print("\n[STEP 6] Generating signals (full period — signals are causal) ...")
    trade_cols = [c for c in pca_results["residuals"].columns
                  if c.split("_")[0] in args.trade_countries]
    pca_for_signals = dict(pca_results)
    pca_for_signals["residuals"] = pca_results["residuals"][trade_cols]
    print(f"  Tradeable instruments: {trade_cols}")

    # OU S-score signals (original approach)
    signal_dict_ou = run_signals(
        pca_for_signals, regime_dict,
        resid_window=args.resid_window,
        s_bo=args.s_bo, s_bc=args.s_bc,
        s_so=args.s_so, s_sc=args.s_sc,
        signal_method="ou",
        save=True,
    )

    # Z-score signals (needed for Methods 1/2/3)
    signal_dict_z = None
    if args.v2:
        signal_dict_z = run_signals(
            pca_for_signals, regime_dict,
            resid_window=args.resid_window,
            signal_method="zscore",
            z_window=args.z_window,
            version="A",
            save=False,
        )

    # ── Step 7: Run all strategies ──────────────────────────────────────────
    print("\n[STEP 7] Running all strategies ...")
    all_results = {}

    # Flatten yield changes for P&L computation
    yc = prep["yield_changes"].copy()
    if hasattr(yc.columns, "levels"):
        yc.columns = [f"{c}_{t}" for c, t in yc.columns]
    trade_yc = yc[[c for c in trade_cols if c in yc.columns]]

    # ── 7a: PCA OU S-score (original) ──────────────────────────────────────
    print("\n  [7a] PCA OU S-score ...")
    daily_residuals = pca_for_signals["residuals"].diff().iloc[1:]
    pnl_ou_i, pnl_ou_p = compute_pnl(
        signal_dict_ou["positions"],
        yield_changes=daily_residuals,
        transaction_cost=args.tc,
    )
    all_results["PCA OU S-score"] = {
        "pnl_portfolio": pnl_ou_p,
        "pnl_instrument": pnl_ou_i,
        "metrics": compute_metrics(pnl_ou_p, regime=regime_dict.get("regime")),
    }
    _save_backtest(pnl_ou_i, pnl_ou_p, all_results["PCA OU S-score"]["metrics"], "backtest_ou")

    # ── 7b-d: Methods 1 / 2 / 3 (V2) ──────────────────────────────────────
    if args.v2 and signal_dict_z is not None:
        for method in [1, 2, 3]:
            label = f"PCA M{method} ({args.covariance.upper()})"
            print(f"\n  [7{'bcd'[method-1]}] {label} ...")
            config = StrategyConfig(
                method          = method,
                covariance      = args.covariance,
                pca_window      = args.corr_window,
                refit           = 63,
                z_window        = args.z_window,
                version         = "A",
                cost_model      = args.cost_model,
                no_trade_band   = args.no_trade_band,
                vol_target      = not args.no_vol_target,
                trade_countries = tuple(args.trade_countries),
            )
            res = run_backtest_v2(
                pca_results    = pca_results,
                signal_dict    = signal_dict_z,
                yields_clean   = prep["yields_clean"],
                yield_changes  = prep["yield_changes"],
                regime_dict    = regime_dict,
                config         = config,
                save           = True,
                out_subdir     = f"backtest_m{method}",
            )
            all_results[label] = res

    # ── 7e: Carry + Roll-Down ───────────────────────────────────────────────
    print("\n  [7e] Carry + Roll-Down ...")
    carry_dict = run_carry(prep["yields_clean"], regime_dict)

    carry_positions = carry_dict["positions"]
    carry_instr = [c for c in carry_positions.columns
                   if c.split("_")[0] in args.trade_countries]
    cp_trade = carry_positions[carry_instr]
    carry_yc = yc[[c for c in carry_instr if c in yc.columns]]

    carry_pnl_i, carry_pnl_p = compute_pnl(
        cp_trade, yield_changes=carry_yc, transaction_cost=args.tc,
    )
    carry_metrics = compute_metrics(carry_pnl_p, regime=regime_dict.get("regime"))
    all_results["Carry + Roll-Down"] = {
        "pnl_portfolio":  carry_pnl_p,
        "pnl_instrument": carry_pnl_i,
        "metrics":        carry_metrics,
    }
    _save_backtest(carry_pnl_i, carry_pnl_p, carry_metrics, "backtest_carry")

    # ── Step 8: Unified comparison table ────────────────────────────────────
    print_strategy_table(all_results, splits)

    # Save carry to outputs/carry_backtest/ (kept for plot compatibility)
    out = os.path.join(BASE_DIR, "outputs", "carry_backtest")
    os.makedirs(out, exist_ok=True)
    carry_pnl_i.to_csv(os.path.join(out, "carry_pnl_daily.csv"))
    carry_pnl_p.to_csv(os.path.join(out, "carry_pnl_total.csv"))
    carry_metrics.to_csv(os.path.join(out, "carry_metrics.csv"))

    # ── Step 9: Visualisations ───────────────────────────────────────────────
    if not args.no_plots:
        print("\n[STEP 9] Generating plots ...")
        # Use OU backtest for the existing plot suite (backward compatible)
        ou_backtest_dict = {
            "pnl_instrument": pnl_ou_i,
            "pnl_portfolio":  pnl_ou_p,
            "metrics":        all_results["PCA OU S-score"]["metrics"],
        }
        carry_backtest_dict = {
            "pnl_instrument": carry_pnl_i,
            "pnl_portfolio":  carry_pnl_p,
            "metrics":        carry_metrics,
        }
        run_all_plots(
            yields_clean       = prep["yields_clean"],
            pca_results        = pca_results,
            regime_dict        = regime_dict,
            signal_dict        = signal_dict_ou,
            backtest_dict      = ou_backtest_dict,
            carry_backtest_dict= carry_backtest_dict,
        )

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print(f"  Outputs in: {os.path.join(BASE_DIR, 'outputs')}/")
    print("=" * 60 + "\n")


def _save_backtest(pnl_i, pnl_p, metrics, subdir):
    """Save backtest outputs to outputs/<subdir>/."""
    out = os.path.join(BASE_DIR, "outputs", subdir)
    os.makedirs(out, exist_ok=True)
    pnl_i.to_csv(os.path.join(out, "pnl_daily.csv"))
    pnl_p.to_csv(os.path.join(out, "pnl_total.csv"))
    metrics.to_csv(os.path.join(out, "metrics.csv"))


if __name__ == "__main__":
    main()
