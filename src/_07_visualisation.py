"""
07_visualisation.py
===================
All charts for the project — saved to outputs/plots/.

Plots generated
---------------
  1. yield_curves.png          — historical yield levels per country
  2. pca_variance_explained.png — rolling cumulative variance explained by PC1-3
  3. pca_loadings.png           — PC1/PC2/PC3 loadings at a selected date
  4. eigenvector_stability.png  — rolling cosine similarity (regime health monitor)
  5. regime_timeline.png        — GOOD/NEUTRAL/BAD regime bands on one chart
  6. s_scores.png               — S-score time series for selected instruments
  7. pnl_curve.png              — cumulative P&L with regime shading
  8. regime_pnl_breakdown.png   — per-regime performance bar chart
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR  = os.path.join(BASE_DIR, "outputs", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

REGIME_COLOURS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
REGIME_NAMES   = {0: "GOOD", 1: "NEUTRAL", 2: "BAD"}
COUNTRY_COLOURS = {"US": "#2980b9", "DE": "#e67e22", "UK": "#8e44ad", "JP": "#27ae60"}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        10,
})


# ── Helpers ────────────────────────────────────────────────────────────────────

def _shade_regimes(ax, regime: pd.Series, alpha: float = 0.15) -> None:
    """Add coloured background shading for regime periods."""
    if regime is None or regime.empty:
        return
    prev_r, start = regime.iloc[0], regime.index[0]
    for date, r in regime.items():
        if r != prev_r:
            ax.axvspan(start, date, color=REGIME_COLOURS[prev_r], alpha=alpha, lw=0)
            prev_r, start = r, date
    ax.axvspan(start, regime.index[-1], color=REGIME_COLOURS[prev_r], alpha=alpha, lw=0)


def _save(fig, name: str) -> None:
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  1. Yield curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_yield_curves(yields_clean: pd.DataFrame, tenors: list = None) -> None:
    """Plot 10Y yield levels for each country on one chart."""
    if tenors is None:
        tenors = ["10Y"]

    fig, ax = plt.subplots(figsize=(14, 5))
    for country in yields_clean.columns.get_level_values("country").unique():
        for t in tenors:
            try:
                s = yields_clean[(country, t)].dropna()
                ax.plot(s.index, s.values, label=f"{country} {t}",
                        color=COUNTRY_COLOURS.get(country, None), lw=1.2)
            except KeyError:
                pass

    ax.set_title("10Y Sovereign Yields — Multi-Country", fontsize=12, fontweight="bold")
    ax.set_ylabel("Yield (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(ncol=4, fontsize=9)
    _save(fig, "yield_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
#  2. Variance explained
# ══════════════════════════════════════════════════════════════════════════════

def plot_variance_explained(var_exp: pd.DataFrame, regime: pd.Series = None) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))

    for pc, col in zip(["PC1", "PC2", "PC3"], ["#2980b9", "#e67e22", "#8e44ad"]):
        if pc in var_exp.columns:
            ax.fill_between(var_exp.index, 0, var_exp[pc] * 100,
                            alpha=0.5, color=col, label=pc)

    ax.plot(var_exp.index, var_exp["cumulative"] * 100,
            color="black", lw=1.5, label="Cumulative (top 3)")
    ax.axhline(99, ls="--", color="red", lw=0.8, alpha=0.6, label="99% threshold")

    _shade_regimes(ax, regime)
    ax.set_title("Rolling Variance Explained by Top-3 PCs", fontsize=12, fontweight="bold")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_ylim(50, 101)
    ax.legend(ncol=5, fontsize=9)
    _save(fig, "pca_variance_explained.png")


# ══════════════════════════════════════════════════════════════════════════════
#  3. PC loadings at a specific date
# ══════════════════════════════════════════════════════════════════════════════

def plot_pca_loadings(loadings_dict: dict, date=None) -> None:
    """Plot PC1/PC2/PC3 loadings (eigenvector components) at a given date."""
    if not loadings_dict:
        print("  [SKIP] No loadings available.")
        return

    dates = sorted(loadings_dict.keys())
    if date is None:
        date = dates[-1]

    betas = loadings_dict.get(date)
    if betas is None:
        betas = loadings_dict[dates[-1]]

    k    = betas.shape[1]
    fig, axes = plt.subplots(1, k, figsize=(14, 4), sharey=False)

    pc_labels = [f"PC{i+1}" for i in range(k)]
    pc_names  = ["Level", "Slope", "Curvature"]
    colours   = ["#2980b9", "#e67e22", "#8e44ad"]

    for i, (ax, pc, name, col) in enumerate(zip(axes, pc_labels, pc_names, colours)):
        vals   = betas[pc] if pc in betas.columns else betas.iloc[:, i]
        instrs = betas.index.tolist()
        ax.bar(range(len(instrs)), vals, color=col, alpha=0.7)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"PC{i+1}: {name}", fontweight="bold")
        ax.set_xticks(range(len(instrs)))
        ax.set_xticklabels(instrs, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Loading (β)")

    fig.suptitle(f"PCA Factor Loadings — {pd.Timestamp(date).date()}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "pca_loadings.png")


# ══════════════════════════════════════════════════════════════════════════════
#  4. Eigenvector stability
# ══════════════════════════════════════════════════════════════════════════════

def plot_eigenvector_stability(ev_stability: pd.DataFrame, regime: pd.Series = None) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    colours = {"PC1_cos": "#2980b9", "PC2_cos": "#e67e22", "PC3_cos": "#8e44ad"}

    for col, c in colours.items():
        if col in ev_stability.columns:
            ax.plot(ev_stability.index, ev_stability[col], lw=1.0,
                    alpha=0.8, color=c, label=col.replace("_cos", ""))

    ax.axhline(0.97, ls="--", color="red",    lw=0.8, alpha=0.7, label="0.97 (GOOD threshold)")
    ax.axhline(0.92, ls="--", color="orange", lw=0.8, alpha=0.7, label="0.92 (NEUTRAL threshold)")
    _shade_regimes(ax, regime)
    ax.set_title("Eigenvector Stability (Cosine Similarity to Previous Window)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=5, fontsize=9)
    _save(fig, "eigenvector_stability.png")


# ══════════════════════════════════════════════════════════════════════════════
#  5. Regime timeline
# ══════════════════════════════════════════════════════════════════════════════

def plot_regime_timeline(regime: pd.Series, var_exp: pd.DataFrame = None) -> None:
    nrows = 2 if var_exp is not None else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 5 if nrows == 1 else 7),
                             sharex=True)
    if nrows == 1:
        axes = [axes]

    # Regime raster
    ax = axes[0]
    ax.scatter(regime.index, regime.values,
               c=[REGIME_COLOURS[r] for r in regime],
               s=6, marker="|", linewidths=1.5)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["GOOD", "NEUTRAL", "BAD"])
    ax.set_title("Regime Timeline", fontsize=12, fontweight="bold")
    patches = [mpatches.Patch(color=REGIME_COLOURS[r], label=REGIME_NAMES[r])
               for r in [0, 1, 2]]
    ax.legend(handles=patches, ncol=3, fontsize=9, loc="upper right")

    if var_exp is not None and nrows > 1:
        ax2 = axes[1]
        ax2.plot(var_exp.index, var_exp["cumulative"] * 100, color="#2c3e50", lw=1.2)
        _shade_regimes(ax2, regime)
        ax2.set_ylabel("Cumulative Var Explained (%)")
        ax2.set_title("Variance Explained (coloured by regime)", fontweight="bold")

    plt.tight_layout()
    _save(fig, "regime_timeline.png")


# ══════════════════════════════════════════════════════════════════════════════
#  6. S-scores
# ══════════════════════════════════════════════════════════════════════════════

def plot_s_scores(
    s_scores: pd.DataFrame,
    instruments: list = None,
    regime: pd.Series = None,
    s_bo: float = 1.25,
    s_bc: float = 0.75,
) -> None:
    if instruments is None:
        instruments = s_scores.columns[:4].tolist()

    n    = len(instruments)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, instr in zip(axes, instruments):
        s = s_scores[instr].dropna()
        ax.plot(s.index, s.values, lw=0.9, color="#2c3e50")
        ax.axhline(0,    color="black", lw=0.5, ls="-")
        ax.axhline( s_bo, color="red",   lw=0.8, ls="--", alpha=0.7, label=f"+{s_bo} (short)")
        ax.axhline(-s_bo, color="green", lw=0.8, ls="--", alpha=0.7, label=f"-{s_bo} (long)")
        ax.axhline( s_bc, color="orange", lw=0.6, ls=":", alpha=0.6)
        ax.axhline(-s_bc, color="orange", lw=0.6, ls=":", alpha=0.6)
        _shade_regimes(ax, regime, alpha=0.10)
        ax.set_ylabel("S-score")
        ax.set_title(str(instr), fontweight="bold")
        ax.set_ylim(-5, 5)
        if ax is axes[0]:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("S-Scores (Standardised Residuals)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "s_scores.png")


# ══════════════════════════════════════════════════════════════════════════════
#  7. Cumulative P&L with regime shading
# ══════════════════════════════════════════════════════════════════════════════

def plot_pnl(pnl_portfolio: pd.DataFrame, regime: pd.Series = None) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Cumulative return
    ax1 = axes[0]
    ax1.plot(pnl_portfolio.index, pnl_portfolio["cumulative_ret"] * 100,
             color="#2980b9", lw=1.5, label="Strategy")
    ax1.axhline(0, color="black", lw=0.5)
    _shade_regimes(ax1, regime)
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.set_title("Strategy Cumulative P&L", fontsize=12, fontweight="bold")
    patches = [mpatches.Patch(color=REGIME_COLOURS[r], label=REGIME_NAMES[r], alpha=0.5)
               for r in [0, 1, 2]]
    ax1.legend(handles=patches + [mpatches.Patch(color="#2980b9", label="Strategy")],
               ncol=4, fontsize=9)

    # Drawdown
    ax2 = axes[1]
    ax2.fill_between(pnl_portfolio.index,
                     pnl_portfolio["drawdown"] * 100, 0,
                     color="#e74c3c", alpha=0.6, label="Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Rolling Drawdown", fontweight="bold")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, "pnl_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
#  8. Per-regime P&L breakdown
# ══════════════════════════════════════════════════════════════════════════════

def plot_regime_pnl_breakdown(metrics: pd.DataFrame) -> None:
    if metrics is None or metrics.empty:
        return
    subset = metrics.loc[metrics.index.isin(["GOOD", "NEUTRAL", "BAD"])]
    if subset.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    stat_cols = ["ann_return_%", "sharpe", "win_rate_%"]
    titles    = ["Annualised Return (%)", "Sharpe Ratio", "Win Rate (%)"]

    for ax, col, title in zip(axes, stat_cols, titles):
        vals   = subset[col]
        colors = [REGIME_COLOURS[{"GOOD": 0, "NEUTRAL": 1, "BAD": 2}[r]] for r in vals.index]
        ax.bar(vals.index, vals.values, color=colors, alpha=0.8, edgecolor="black", lw=0.5)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(title, fontweight="bold")
        for i, v in enumerate(vals.values):
            ax.text(i, v + 0.1 * (1 if v >= 0 else -1), f"{v:.2f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=9)

    fig.suptitle("Performance Breakdown by Regime", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "regime_pnl_breakdown.png")


# ══════════════════════════════════════════════════════════════════════════════
#  9. PCA vs Carry comparison
# ══════════════════════════════════════════════════════════════════════════════



# ======================================================================
#  9. PCA vs Carry comparison
# ======================================================================

def plot_strategy_comparison(
    pca_pnl,
    carry_pnl,
    regime=None,
    pca_metrics=None,
    carry_metrics=None,
):
    """Three-panel comparison: cumulative return, drawdown, rolling Sharpe."""
    common = pca_pnl.index.intersection(carry_pnl.index)
    pca   = pca_pnl.loc[common]
    carry = carry_pnl.loc[common]

    fig = plt.figure(figsize=(15, 11))
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[3, 1.5, 1.5], hspace=0.35)

    # Panel 1: Cumulative return
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(common, pca["cumulative_ret"] * 100,
             color="#2980b9", lw=1.8, label="PCA Mean-Reversion")
    ax1.plot(common, carry["cumulative_ret"] * 100,
             color="#e67e22", lw=1.8, label="Carry + Roll-Down", linestyle="--")
    ax1.axhline(0, color="black", lw=0.5)
    _shade_regimes(ax1, regime)

    final_pca   = pca["cumulative_ret"].iloc[-1] * 100
    final_carry = carry["cumulative_ret"].iloc[-1] * 100
    ax1.annotate(f"{final_pca:+.1f}%", xy=(common[-1], final_pca),
                 xytext=(8, 0), textcoords="offset points",
                 color="#2980b9", fontsize=9, fontweight="bold",
                 va="center")
    ax1.annotate(f"{final_carry:+.1f}%", xy=(common[-1], final_carry),
                 xytext=(8, 0), textcoords="offset points",
                 color="#e67e22", fontsize=9, fontweight="bold",
                 va="center")

    regime_patches = [mpatches.Patch(color=REGIME_COLOURS[r], label=REGIME_NAMES[r], alpha=0.5)
                      for r in [0, 1, 2]]
    strat_patches  = [mpatches.Patch(color="#2980b9", label="PCA Mean-Reversion"),
                      mpatches.Patch(color="#e67e22", label="Carry + Roll-Down")]
    ax1.legend(handles=strat_patches + regime_patches, ncol=5, fontsize=8, loc="upper left")
    ax1.set_ylabel("Cumulative Return (%)", fontsize=10)
    ax1.set_title("Strategy Comparison: PCA Mean-Reversion vs Carry + Roll-Down",
                  fontsize=12, fontweight="bold")

    # Panel 2: Drawdown
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(common, pca["drawdown"] * 100, 0,
                     color="#2980b9", alpha=0.4, label="PCA DD")
    ax2.fill_between(common, carry["drawdown"] * 100, 0,
                     color="#e67e22", alpha=0.4, label="Carry DD")
    _shade_regimes(ax2, regime, alpha=0.08)
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.set_title("Drawdown", fontweight="bold", fontsize=10)
    ax2.legend(fontsize=8, loc="lower left")

    # Panel 3: Rolling 1Y Sharpe
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    roll = 252
    pca_roll_sharpe   = (pca["daily_ret"].rolling(roll).mean() /
                         pca["daily_ret"].rolling(roll).std()) * np.sqrt(252)
    carry_roll_sharpe = (carry["daily_ret"].rolling(roll).mean() /
                         carry["daily_ret"].rolling(roll).std()) * np.sqrt(252)
    ax3.plot(common, pca_roll_sharpe,   color="#2980b9", lw=1.2, label="PCA 1Y Sharpe")
    ax3.plot(common, carry_roll_sharpe, color="#e67e22", lw=1.2, linestyle="--",
             label="Carry 1Y Sharpe")
    ax3.axhline(0, color="black", lw=0.5)
    ax3.axhline(1, color="grey",  lw=0.5, linestyle=":")
    _shade_regimes(ax3, regime, alpha=0.08)
    ax3.set_ylabel("Rolling Sharpe", fontsize=10)
    ax3.set_title("Rolling 1-Year Sharpe Ratio", fontweight="bold", fontsize=10)
    ax3.legend(fontsize=8)

    # Summary stats footer
    if pca_metrics is not None and carry_metrics is not None:
        try:
            pr = pca_metrics.loc["Overall"]
            cr = carry_metrics.loc["Overall"]
            txt = (
                f"PCA:   Sharpe {pr['sharpe']:.2f}  Ann.Ret {pr['ann_return_%']:.1f}%"
                f"  Vol {pr['ann_vol_%']:.1f}%  MaxDD {pr['max_drawdown_%']:.1f}%     "
                f"Carry: Sharpe {cr['sharpe']:.2f}  Ann.Ret {cr['ann_return_%']:.1f}%"
                f"  Vol {cr['ann_vol_%']:.1f}%  MaxDD {cr['max_drawdown_%']:.1f}%"
            )
            fig.text(0.5, 0.005, txt, ha="center", fontsize=8, style="italic",
                     color="#444444",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8",
                               edgecolor="#cccccc"))
        except Exception:
            pass

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")
    _save(fig, "strategy_comparison.png")


# ======================================================================
#  Run all plots
# ======================================================================

def run_all_plots(
    yields_clean=None,
    pca_results=None,
    regime_dict=None,
    signal_dict=None,
    backtest_dict=None,
    carry_backtest_dict=None,
):
    print("=" * 50)
    print("  Generating Visualisations")
    print("=" * 50)

    regime = regime_dict.get("regime") if regime_dict else None

    if yields_clean is not None:
        print("\n[1] Yield curves ...")
        plot_yield_curves(yields_clean)

    if pca_results is not None:
        print("[2] Variance explained ...")
        plot_variance_explained(pca_results["var_explained"], regime)

        print("[3] PCA loadings ...")
        plot_pca_loadings(pca_results.get("loadings", {}))

        print("[4] Eigenvector stability ...")
        plot_eigenvector_stability(pca_results["ev_stability"], regime)

    if regime_dict is not None:
        print("[5] Regime timeline ...")
        var_exp = pca_results["var_explained"] if pca_results else None
        plot_regime_timeline(regime, var_exp)

    if signal_dict is not None:
        print("[6] S-scores ...")
        plot_s_scores(signal_dict["s_scores"], regime=regime)

    if backtest_dict is not None and "pnl_portfolio" in backtest_dict:
        print("[7] PCA P&L curve ...")
        plot_pnl(backtest_dict["pnl_portfolio"], regime)

        if "metrics" in backtest_dict:
            print("[8] Regime P&L breakdown ...")
            plot_regime_pnl_breakdown(backtest_dict["metrics"])

    if (backtest_dict is not None and "pnl_portfolio" in backtest_dict and
            carry_backtest_dict is not None and "pnl_portfolio" in carry_backtest_dict):
        print("[9] Strategy comparison (PCA vs Carry) ...")
        plot_strategy_comparison(
            pca_pnl=backtest_dict["pnl_portfolio"],
            carry_pnl=carry_backtest_dict["pnl_portfolio"],
            regime=regime,
            pca_metrics=backtest_dict.get("metrics"),
            carry_metrics=carry_backtest_dict.get("metrics"),
        )

    print(f"\n  All plots saved to {PLOT_DIR}/")
