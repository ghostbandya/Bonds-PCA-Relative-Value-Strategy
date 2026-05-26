"""
05_signal_generation.py
=======================
Ornstein-Uhlenbeck fitting and S-score signal generation.

For each instrument i and each date t:
  1. Take the cumulated residual series X_i over the inner window (60 days)
  2. Fit OU parameters by OLS on AR(1):
       ΔX_i(t) = a + b · X_i(t-1) + noise
       → κ = -log(1+b)/Δt    (mean-reversion speed)
       → m = -a/b             (long-run mean; ≈ 0 by construction)
       → σ_eq = std(noise) / sqrt(2κ)   (equilibrium std dev)
  3. Compute S-score:
       s_i(t) = X_i(t) / σ_eq,i(t)
  4. Apply trading rules gated by regime:
       GOOD    → trade at full size
       NEUTRAL → trade at 50% size
       BAD     → no new positions; close existing ones

Signal rules — YIELD-BASED convention
--------------------------------------
  NOTE: PCA is on yield CHANGES, so the residual X_i has the same sign as yield.
  High X_i → yield above model → bond price below model → bond is CHEAP → go LONG.
  Low  X_i → yield below model → bond price above model → bond is EXPENSIVE → go SHORT.
  This is the OPPOSITE of the Avellaneda & Lee equity convention
  (where high residual = stock above model = expensive = short).

  Open long    s >  s_bo    (default 1.25)   bond cheap relative to factor model
  Close long   s <  s_bc    (default 0.75)   mean reversion largely complete
  Open short   s < −s_so    (default 1.25)   bond expensive relative to factor model
  Close short  s > −s_sc    (default 0.50)   mean reversion largely complete
  Filter       κ  >  κ_min  (default 8.4 = half-life < 30 days)
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


# ══════════════════════════════════════════════════════════════════════════════
#  OU parameter estimation
# ══════════════════════════════════════════════════════════════════════════════

def fit_ou_params(x: np.ndarray, dt: float = 1/252) -> dict:
    """
    Estimate OU parameters from a residual series x via OLS on AR(1).

    Model:  ΔX(t) = a + b·X(t-1) + ε

    Parameters
    ----------
    x  : 1-D array — cumulated residual series  X_i(t)
    dt : time step in years (1/252 for daily)

    Returns
    -------
    dict: kappa, mu, sigma, sigma_eq, half_life, r_squared
    """
    if len(x) < 10:
        return dict(kappa=np.nan, mu=np.nan, sigma=np.nan,
                    sigma_eq=np.nan, half_life=np.nan, r_squared=np.nan)

    X_lag = x[:-1]
    dX    = np.diff(x)

    # OLS with intercept
    A       = np.column_stack([np.ones_like(X_lag), X_lag])
    result  = np.linalg.lstsq(A, dX, rcond=None)
    coeffs  = result[0]
    a, b    = coeffs[0], coeffs[1]

    # Residuals and their std
    fitted  = A @ coeffs
    resid   = dX - fitted
    sigma   = resid.std(ddof=2)

    # OU parameters
    kappa   = -np.log(1 + b) / dt if (1 + b) > 0 else np.nan
    mu      = -a / b if abs(b) > 1e-10 else 0.0
    half_life = np.log(2) / kappa if (kappa is not np.nan and kappa > 0) else np.nan

    # σ_eq = σ_OU / √(2κ)  — the OU equilibrium std dev.
    # For low-volatility instruments (e.g. JGB) the OU innovations σ can be
    # tiny, making σ_eq → 0 and S-scores blow up to ±100+.  We floor σ_eq at
    # 30% of the window's empirical std of X, which bounds |S| to ~10×
    # (3 empirical stds / 0.30 = 10) — well within a ±5 cap applied below.
    if kappa is not np.nan and kappa > 0:
        sigma_eq_theory = sigma / np.sqrt(2 * kappa)
        # Floor at 60% of window empirical std of X.
        # With floor = 0.60 * std(X), a 2-sigma deviation gives |S| ≈ 3.3,
        # so the ±5 cap is hit only in genuine tail events (~1-2% of obs).
        sigma_eq_floor  = np.std(x) * 0.60
        sigma_eq        = max(sigma_eq_theory, sigma_eq_floor)
    else:
        sigma_eq = np.nan

    # R² of the AR(1) regression
    ss_res  = (resid**2).sum()
    ss_tot  = ((dX - dX.mean())**2).sum()
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return dict(kappa=kappa, mu=mu, sigma=sigma,
                sigma_eq=sigma_eq, half_life=half_life, r_squared=r2)


# ══════════════════════════════════════════════════════════════════════════════
#  S-score computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_s_scores(
    residuals: pd.DataFrame,
    resid_window: int = 60,
    kappa_min:    float = 8.4,
    dt:           float = 1/252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute S-scores for all instruments across all dates.

    For each date t:
      - Slice the residual series over the last `resid_window` days
      - Fit OU parameters
      - Compute s_i(t) = X_i(t) / σ_eq,i

    Parameters
    ----------
    residuals    : DataFrame (date × instruments) — cumulated residuals from rolling PCA
    resid_window : inner window for OU estimation
    kappa_min    : filter: only trade instruments with kappa > kappa_min
    dt           : time step in years

    Returns
    -------
    s_scores : DataFrame (date × instruments) — S-scores (NaN if κ too small)
    ou_params : DataFrame — OU params per instrument per date
                            (MultiIndex index: [date, instrument])
    """
    dates       = residuals.index
    instruments = residuals.columns.tolist()
    n           = len(dates)

    s_matrix   = np.full((n, len(instruments)), np.nan)
    ou_records = []

    for i, t in enumerate(dates):
        if i < resid_window:
            continue
        window_resid = residuals.iloc[i - resid_window: i]

        for j, instr in enumerate(instruments):
            x = window_resid[instr].dropna().values
            if len(x) < 10:
                continue

            params = fit_ou_params(x, dt=dt)
            kappa  = params["kappa"]
            sig_eq = params["sigma_eq"]

            ou_records.append({"date": t, "instrument": instr, **params})

            # Only generate signal if mean-reversion is fast enough
            if (kappa is not np.nan and kappa > kappa_min
                    and sig_eq is not np.nan and sig_eq > 0):
                x_now = residuals.at[t, instr]
                if not np.isnan(x_now):
                    s_raw = (x_now - params["mu"]) / sig_eq
                    # Cap at ±5 — beyond ±5 σ the signal is no longer
                    # informative and is likely a calibration artefact
                    s_matrix[i, j] = np.clip(s_raw, -5.0, 5.0)

    s_scores = pd.DataFrame(s_matrix, index=dates, columns=instruments)
    ou_params = pd.DataFrame(ou_records).set_index(["date", "instrument"]) \
                if ou_records else pd.DataFrame()

    return s_scores, ou_params


# ══════════════════════════════════════════════════════════════════════════════
#  Trading signal generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_signals(
    s_scores:     pd.DataFrame,
    regime:       pd.Series,
    s_bo:         float = 1.25,   # open long threshold
    s_bc:         float = 0.75,   # close long threshold
    s_so:         float = 1.25,   # open short threshold
    s_sc:         float = 0.50,   # close short threshold
    size_neutral: float = 0.50,   # position size in NEUTRAL regime
) -> pd.DataFrame:
    """
    Convert S-scores + regime labels into target positions.

    Position values:
       +1.0  : full long
       +0.5  : half long (NEUTRAL regime)
        0.0  : flat
       -0.5  : half short (NEUTRAL regime)
       -1.0  : full short

    Logic (applied per instrument per date):
      BAD regime     → 0 (forced flat)
      NEUTRAL regime → same signals as GOOD but at `size_neutral` magnitude
      GOOD regime    → standard thresholds

    Position transitions are forward-filled (hold until exit signal).

    Parameters
    ----------
    s_scores : DataFrame (date × instruments) — from compute_s_scores()
    regime   : Series (date → int 0/1/2)      — from detect_regimes()

    Returns
    -------
    positions : DataFrame (date × instruments) — target position sizes
    signals   : DataFrame (date × instruments) — raw open/close signals
    """
    common  = s_scores.index.intersection(regime.index)
    scores  = s_scores.loc[common]
    reg     = regime.loc[common]
    instrs  = scores.columns.tolist()

    n = len(common)
    pos_matrix = np.zeros((n, len(instrs)))
    sig_matrix = np.zeros((n, len(instrs)))

    for j, instr in enumerate(instrs):
        s      = scores[instr].values
        pos    = 0.0

        for i in range(n):
            r = reg.iloc[i]

            if r == 2:   # BAD — force flat
                pos = 0.0
                sig_matrix[i, j] = 0
                pos_matrix[i, j] = 0
                continue

            si = s[i]
            if np.isnan(si):
                sig_matrix[i, j] = np.nan
                pos_matrix[i, j] = pos
                continue

            size = 1.0 if r == 0 else size_neutral  # scale down in NEUTRAL

            # Exit existing position first
            # YIELD convention: long when S > s_bo, exit when S falls below s_bc
            if pos > 0 and si < s_bc:
                pos = 0.0
                sig_matrix[i, j] = 0
            elif pos < 0 and si > -s_sc:
                pos = 0.0
                sig_matrix[i, j] = 0

            # Open new position (only if currently flat)
            # YIELD convention: go LONG when yield is above model (S > +s_bo = cheap bond)
            #                   go SHORT when yield is below model (S < -s_so = expensive bond)
            if pos == 0:
                if si > s_so:
                    pos = size
                    sig_matrix[i, j] = 1
                elif si < -s_bo:
                    pos = -size
                    sig_matrix[i, j] = -1

            pos_matrix[i, j] = pos

    positions = pd.DataFrame(pos_matrix, index=common, columns=instrs)
    signals   = pd.DataFrame(sig_matrix, index=common, columns=instrs)
    return positions, signals


# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_signals(
    pca_results:  dict,
    regime_dict:  dict,
    resid_window: int   = 60,
    kappa_min:    float = 8.4,
    s_bo: float = 1.25, s_bc: float = 0.75,
    s_so: float = 1.25, s_sc: float = 0.50,
    save: bool = True,
) -> dict:
    """Full signal generation pipeline."""
    print("=" * 50)
    print("  Signal Generation Pipeline")
    print("=" * 50)

    print("\n[1] Computing S-scores …")
    s_scores, ou_params = compute_s_scores(
        pca_results["residuals"],
        resid_window=resid_window,
        kappa_min=kappa_min,
    )
    print(f"    S-scores shape: {s_scores.shape}")

    print("[2] Generating positions …")
    positions, signals = generate_signals(
        s_scores, regime_dict["regime"],
        s_bo=s_bo, s_bc=s_bc, s_so=s_so, s_sc=s_sc,
    )

    # Summary stats
    active_pct = (positions != 0).mean().mean() * 100
    long_pct   = (positions > 0).mean().mean() * 100
    short_pct  = (positions < 0).mean().mean() * 100
    print(f"    Active positions: {active_pct:.1f}% of (date, instrument) cells")
    print(f"    Long / Short:     {long_pct:.1f}% / {short_pct:.1f}%")

    if save:
        out = os.path.join(OUTPUT_DIR, "signals")
        os.makedirs(out, exist_ok=True)
        s_scores.to_csv(os.path.join(out, "s_scores.csv"))
        positions.to_csv(os.path.join(out, "positions.csv"))
        signals.to_csv(os.path.join(out, "signals.csv"))
        if not ou_params.empty:
            ou_params.to_csv(os.path.join(out, "ou_params.csv"))
        print(f"    Signals saved to {out}/")

    return {
        "s_scores":  s_scores,
        "positions": positions,
        "signals":   signals,
        "ou_params": ou_params,
    }
