"""
05_signal_generation.py
=======================
Ornstein-Uhlenbeck fitting and S-score signal generation.

═══════════════════════════════════════════════════════════════
THE ORNSTEIN-UHLENBECK (OU) PROCESS
═══════════════════════════════════════════════════════════════
The cumulated PCA residual X_i(t) is modelled as an OU process:

    dX = κ(μ - X) dt + σ dW

where:
  κ  = mean-reversion speed  (how fast X snaps back to μ)
  μ  = long-run mean          (≈ 0 by PCA construction)
  σ  = diffusion coefficient  (daily innovation std dev)

The OU process is the continuous-time limit of an AR(1) model.
We fit it via OLS on the discrete AR(1):

    ΔX(t) = a + b · X(t-1) + ε

Then map AR(1) coefficients to OU parameters:
  κ = -log(1+b) / Δt        [mean-reversion speed, annualised]
  μ = -a / b                 [long-run mean]
  σ_OU = std(ε)              [daily residual std dev]

Half-life = log(2) / κ      [time for X to decay halfway to μ]

═══════════════════════════════════════════════════════════════
S-SCORE NORMALISATION
═══════════════════════════════════════════════════════════════
The S-score is X_i normalised by the OU equilibrium std dev:

    σ_eq = σ_OU / √(2κ)

σ_eq is the std dev that X_i would have if it were in steady state.
It tells us how far X_i "normally" wanders from zero.  An S-score
of ±2 means X_i is at ±2 equilibrium standard deviations — a
meaningful deviation that should revert.

WHY FLOOR σ_eq AT 60% OF EMPIRICAL STD(X)?
When κ is very large (fast mean reversion), σ_eq → 0, making S-scores
blow up to ±100.  This is a calibration artefact, not genuine signal.
We floor σ_eq at 0.60 × std(X), which caps |S| at ~3.3σ for a 2-sigma
deviation and puts the ±5 hard cap at a genuine 3.3+ sigma event.

═══════════════════════════════════════════════════════════════
YIELD CONVENTION — OPPOSITE TO EQUITY STAT-ARB
═══════════════════════════════════════════════════════════════
PCA residuals are in yield-change space, so X_i has the same sign
as the yield deviation from model:

  X_i > 0  →  yield ABOVE model  →  bond PRICE below model
             →  bond is CHEAP    →  go LONG   (buy the bond)

  X_i < 0  →  yield BELOW model  →  bond PRICE above model
             →  bond is EXPENSIVE →  go SHORT  (sell the bond)

This is the OPPOSITE of Avellaneda & Lee (2010) for equities,
where a high residual means the stock is overpriced → short.
The sign flip is because yields and prices move inversely.

═══════════════════════════════════════════════════════════════
SIGNAL THRESHOLDS (default values)
═══════════════════════════════════════════════════════════════
  s_bo = 1.25  open long   (bond cheap by 1.25 σ_eq)
  s_bc = 0.75  close long  (reversion mostly complete)
  s_so = 1.25  open short  (bond rich by 1.25 σ_eq)
  s_sc = 0.50  close short (reversion mostly complete)

  Asymmetric exit thresholds (0.75 / 0.50 vs. entry 1.25):
  We close positions before full reversion to capture most of the
  move while avoiding reversal risk if the spread re-widens.

  κ_min = 8.4  corresponds to half-life < 30 days.
  We only trade instruments where the OU model predicts reversion
  within ~1 month.  Slower mean reversion is too slow to monetise
  before positions get hit by factor moves.

═══════════════════════════════════════════════════════════════
REGIME FILTER
═══════════════════════════════════════════════════════════════
  GOOD    → full position size (±1.0)
  NEUTRAL → half size (±0.5) — PCA structure shifting, less confidence
  BAD     → flat — PCA broken, residuals no longer mean-revert reliably
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
    # ── Map AR(1) → OU parameters ────────────────────────────────────────
    # AR(1):  X_t = (1+b)X_{t-1} + a + ε
    # OU discrete:  X_t = e^{-κΔt} X_{t-1} + μ(1-e^{-κΔt}) + noise
    # Matching coefficients: e^{-κΔt} = (1+b)  →  κ = -log(1+b)/Δt
    # Require (1+b) > 0 i.e. b > -1, otherwise the AR(1) is explosive.
    kappa   = -np.log(1 + b) / dt if (1 + b) > 0 else np.nan
    mu      = -a / b if abs(b) > 1e-10 else 0.0
    half_life = np.log(2) / kappa if (kappa is not np.nan and kappa > 0) else np.nan

    # ── σ_eq: equilibrium standard deviation of the OU process ───────────
    # σ_eq = σ_OU / √(2κ)
    # In steady state, Var(X) = σ²/(2κ), so σ_eq is the "natural" spread.
    # An S-score of ±1 means X is at ±1 σ_eq from its long-run mean.
    #
    # FLOOR RATIONALE:
    # When κ is very large (very fast mean reversion), σ_eq → 0, which would
    # make S-scores blow up to ±100+ for even tiny movements.  This is a
    # calibration artefact, not real signal.  We floor at 60% of the window's
    # empirical std(X), so the largest |S| from a 2-sigma event is:
    #   |S| = 2·std(X) / (0.60·std(X)) = 3.33 — safely within the ±5 cap.
    if kappa is not np.nan and kappa > 0:
        sigma_eq_theory = sigma / np.sqrt(2 * kappa)
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

            # BAD regime: force all positions to zero.
            # The PCA factor structure has broken down → residuals no longer
            # mean-revert predictably → any open trade is noise-driven.
            if r == 2:
                pos = 0.0
                sig_matrix[i, j] = 0
                pos_matrix[i, j] = 0
                continue

            si = s[i]
            if np.isnan(si):
                # No valid S-score this date — hold current position, no new signal
                sig_matrix[i, j] = np.nan
                pos_matrix[i, j] = pos
                continue

            # NEUTRAL: same signal logic but half position size.
            # Factor structure is shifting — we still trade but reduce risk.
            size = 1.0 if r == 0 else size_neutral

            # ── EXIT logic (check before entry to avoid flip in one step) ──
            # Close long when S has fallen back toward zero (reversion done).
            # Close short when S has risen back toward zero.
            # Asymmetric thresholds: we close at 0.75/0.50, not 1.25.
            # This locks in most of the mean-reversion move without waiting
            # for full convergence (which may never arrive perfectly).
            if pos > 0 and si < s_bc:
                pos = 0.0
                sig_matrix[i, j] = 0
            elif pos < 0 and si > -s_sc:
                pos = 0.0
                sig_matrix[i, j] = 0

            # ── ENTRY logic (only when flat) ───────────────────────────────
            # YIELD convention (opposite of equity stat-arb):
            #   S > +s_bo → yield above model → bond cheap → BUY (long)
            #   S < -s_so → yield below model → bond rich  → SELL (short)
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


# ══════════════════════════════════════════════════════════════════════════════
#  Z-SCORE SIGNAL — Jay's approach (simpler and more robust than OU)
# ══════════════════════════════════════════════════════════════════════════════
#
# COMPARISON: OU S-score vs Rolling Z-score
# ─────────────────────────────────────────
# OU S-score: fits an AR(1) model to estimate σ_eq from mean-reversion speed.
#   + Theoretically grounded (OU process)
#   - σ_eq can blow up when κ is large (needs floor)
#   - OU fitting needs ~30+ obs to be stable
#
# Rolling Z-score: (z - trailing_mean) / trailing_std over a window.
#   + Simpler: no model fitting required
#   + No blow-up risk
#   + Strictly causal (trailing window only)
#   - No explicit link to mean-reversion speed
#
# In Jay's results, the z-score approach with Method 3 achieves IR 1.318
# (gross) — significantly better than the OU S-score baseline.
# Both are available; choose via signal_method parameter in run_signals().

def compute_z_scores(
    residuals: pd.DataFrame,
    z_window: int = 63,
    version: str = "A",
    refit_dates: list = None,
    K: int = None,
) -> pd.DataFrame:
    """
    Compute rolling z-scores on cumulated PCA residuals.

    Version A: cumulative sum resets to zero at each refit boundary.
               Cleaner — doesn't mix residuals from different PCA vintages.
    Version B: rolling sum over the last K residuals.

    Parameters
    ----------
    residuals    : DataFrame (date x instruments) — daily PCA residuals
    z_window     : trailing window for mean/std normalisation
    version      : "A" (block-reset) | "B" (rolling-K sum)
    refit_dates  : list of refit boundary dates (required for Version A)
    K            : rolling window for Version B cumsum

    Returns
    -------
    z_scores : DataFrame (date x instruments) — z-score per instrument
               NaN where window is not yet warm.
    """
    if version == "A":
        # Block-reset cumulative residual (Version A).
        # At each PCA refit boundary, the cumsum restarts from zero. Without
        # this, residuals from different PCA vintages (potentially with rotated
        # eigenvectors) would be accumulated together, mixing signals from
        # structurally different factor models.
        if refit_dates is None:
            # No refit dates supplied: treat the whole series as one block.
            cum = residuals.cumsum()
        else:
            cum = pd.DataFrame(np.nan, index=residuals.index, columns=residuals.columns)
            refit_pos = sorted(
                {residuals.index.get_loc(d) for d in refit_dates
                 if d in residuals.index}
            )
            bounds = refit_pos + [len(residuals)]
            for k_idx in range(len(bounds) - 1):
                s, e = bounds[k_idx], bounds[k_idx + 1]
                if e <= s:
                    continue
                cum.iloc[s:e] = residuals.iloc[s:e].cumsum().values

    elif version == "B":
        if K is None:
            raise ValueError("version='B' requires K to be specified")
        cum = residuals.rolling(K, min_periods=K).sum()

    else:
        raise ValueError(f"version must be 'A' or 'B'; got '{version}'")

    # Strictly-causal trailing z-score (no centering; ddof=1)
    # min_periods=z_window ensures NaN until the window is warm.
    mu = cum.rolling(z_window, min_periods=z_window).mean()
    sd = cum.rolling(z_window, min_periods=z_window).std(ddof=1)
    sd = sd.replace(0.0, np.nan)

    z_scores = (cum - mu) / sd
    return z_scores


def run_signals(
    pca_results:  dict,
    regime_dict:  dict,
    resid_window: int   = 60,
    kappa_min:    float = 8.4,
    s_bo: float = 1.25, s_bc: float = 0.75,
    s_so: float = 1.25, s_sc: float = 0.50,
    save: bool = True,
    signal_method: str = "ou",       # "ou" (original) | "zscore" (Jay's approach)
    z_window: int = 63,               # z-score window (used when signal_method="zscore")
    version: str = "A",               # "A" | "B" (z-score cumulation style)
    refit_dates: list = None,         # for Version A block-reset
    K: int = None,                    # for Version B rolling sum
) -> dict:
    """
    Full signal generation pipeline.

    signal_method="ou"     : original OU S-score approach
    signal_method="zscore" : Jay's rolling z-score approach (recommended)
    """
    print("=" * 50)
    print("  Signal Generation Pipeline")
    print(f"  Method: {signal_method.upper()}")
    print("=" * 50)

    if signal_method == "ou":
        print("\n[1] Computing S-scores (OU method) …")
        s_scores, ou_params = compute_s_scores(
            pca_results["residuals"],
            resid_window=resid_window,
            kappa_min=kappa_min,
        )
        print(f"    S-scores shape: {s_scores.shape}")
    else:
        print(f"\n[1] Computing Z-scores (window={z_window}, version={version}) …")
        s_scores = compute_z_scores(
            pca_results["residuals"],
            z_window=z_window,
            version=version,
            refit_dates=refit_dates,
            K=K,
        )
        ou_params = pd.DataFrame()
        print(f"    Z-scores shape: {s_scores.shape}")

    # Cap z-scores at ±5 (same as OU S-score convention)
    s_scores = s_scores.clip(-5.0, 5.0)

    print("[2] Generating positions …")
    positions, signals = generate_signals(
        s_scores, regime_dict["regime"],
        s_bo=s_bo, s_bc=s_bc, s_so=s_so, s_sc=s_sc,
    )

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
        "s_scores":     s_scores,
        "positions":    positions,
        "signals":      signals,
        "ou_params":    ou_params,
        "signal_method": signal_method,
    }
