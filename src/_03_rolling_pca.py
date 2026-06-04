"""
03_rolling_pca.py
=================
Rolling PCA engine — the analytical core of the strategy.

═══════════════════════════════════════════════════════════════
WHY PCA ON YIELD CURVES?
═══════════════════════════════════════════════════════════════
With 28 yield series (7 tenors × 4 countries) we would normally
need to track a 28×28 covariance matrix.  PCA collapses this to
3 uncorrelated "meta-drivers" that explain ~75% of total variance
in a cross-country panel (and ~99% within a single country).

The 3 factors have well-known interpretations:
  PC1 — Level    : parallel shift, all yields move together
                   (~90% of variance in single-country PCA)
  PC2 — Slope    : curve steepens / flattens (2s10s spread)
  PC3 — Curvature: belly bows up or down relative to wings
                   (5Y moves opposite to 2Y+30Y)

═══════════════════════════════════════════════════════════════
WHAT THIS MODULE DOES — STEP BY STEP
═══════════════════════════════════════════════════════════════
For each date t (once we have at least corr_window days of history):

  1. CORRELATION WINDOW  [t - 252 : t]
     Build the standardised yield-change matrix Z ∈ ℝ^{252 × N}.
     Compute the empirical correlation matrix C = Z'Z / 251.
     Eigen-decompose: C = V Λ V',  keep top k=3 columns of V.

  2. STABILITY CHECK
     Measure how much the eigenvectors rotated vs. yesterday via
     cosine similarity  cos θ = |v_t · v_{t-1}|.
     Values near 1.0 = stable factor structure.
     Sudden drop → regime feature used downstream.

  3. RESIDUAL WINDOW  [t - 60 : t]
     Using the loadings from step 1, project the inner 60-day
     window of yield changes onto the 3 PCs:
       F = Z · V_k          (factor scores, T×k)
     OLS-regress each instrument on F to get betas β_i:
       Δy_i = β_i1 F1 + β_i2 F2 + β_i3 F3 + ε_i
     Cumulate the residuals: X_i(t) = Σ ε_i
     This running integral of the idiosyncratic component is
     what the OU / S-score model then fits.

     Why 60 days for residuals but 252 for correlation?
       - 252 days gives a stable covariance estimate (more data = better)
       - 60 days keeps the OU calibration window short so κ reflects
         current mean-reversion speed, not stale dynamics from a year ago

  4. R² diagnostic
     R²_i = 1 - SS_res_i / SS_tot_i across the residual window.
     High R² (>0.80) means the 3 PCs explain the instrument well —
     a good condition for the residual to mean-revert predictably.

═══════════════════════════════════════════════════════════════
KEY DESIGN CHOICE — CORRELATION vs COVARIANCE MATRIX
═══════════════════════════════════════════════════════════════
We standardise (divide by vol) before PCA, i.e. use the correlation
matrix.  Alternative: covariance matrix (no standardisation).

Covariance PCA would let high-volatility instruments (e.g. 30Y in
a sell-off) dominate the first eigenvector simply because they move
more in absolute bps terms.  We want PC1 to represent a *structural*
parallel shift, not just "whichever tenor is most volatile today."
Correlation PCA is the practitioner standard (CS PCA Unleashed, 2012).

═══════════════════════════════════════════════════════════════
OUTPUT FILES  (outputs/pca/)
═══════════════════════════════════════════════════════════════
  residuals.csv       — cumulated OU residuals X_i(t), one column per instrument
  factor_scores.csv   — daily PC1/PC2/PC3 time series
  var_explained.csv   — fraction of variance each PC explains, plus cumulative
  eigenvalues.csv     — raw eigenvalues λ1, λ2, λ3 per date
  ev_stability.csv    — cosine similarity of each PC to previous day's eigenvector
  r_squared.csv       — mean R² of factor model across instruments
"""

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Single-window PCA
# ══════════════════════════════════════════════════════════════════════════════

def run_pca_on_window(
    window: pd.DataFrame,
    k: int = 3,
) -> dict:
    """
    Run PCA on a single time window of yield changes.

    Parameters
    ----------
    window : DataFrame (T_window × N_instruments), standardised values
    k      : number of components to retain

    Returns
    -------
    dict with:
      'eigenvalues'   : array (k,)      — top k eigenvalues (variance explained)
      'eigenvectors'  : array (N × k)   — loadings matrix, columns = PCs
      'var_explained' : array (k,)      — fraction of total variance per PC
      'cum_var'       : float           — cumulative variance of top-k PCs
      'factor_scores' : DataFrame (T × k) — PC time series over the window
    """
    # ── Step 1: Standardise ──────────────────────────────────────────────────
    # Each column is zero-meaned and divided by its std dev.
    # This turns the covariance matrix into a correlation matrix, preventing
    # high-vol tenors (e.g. 30Y in a sell-off) from dominating PC1.
    scaler    = StandardScaler()
    Z         = scaler.fit_transform(window.values)  # shape (T, N)
    T, N      = Z.shape

    # ── Step 2: Correlation matrix  C = Z'Z / (T-1) ──────────────────────
    # Equivalent to np.corrcoef(window.T) but faster when called thousands of times.
    C         = (Z.T @ Z) / (T - 1)

    # ── Step 3: Eigendecomposition ────────────────────────────────────────
    # np.linalg.eigh is for symmetric matrices (faster + numerically stable
    # vs. np.linalg.eig). Returns eigenvalues in *ascending* order → reverse.
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx          = np.argsort(eigenvalues)[::-1]   # descending order
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]   # (N × N), columns = eigenvectors

    # ── Step 4: Retain top k ──────────────────────────────────────────────
    # We keep k=3 by default (level, slope, curvature).
    # The fraction of total variance explained by PC_j = λ_j / Σλ_i.
    top_eigenvalues  = eigenvalues[:k]
    top_eigenvectors = eigenvectors[:, :k]   # (N × k)

    var_explained = top_eigenvalues / eigenvalues.sum()
    cum_var       = var_explained.sum()

    # ── Step 5: Factor scores  F = Z · V_k   shape (T × k) ───────────────
    # Each row of F is the projection of that day's yield-change vector
    # onto the k principal directions. These are the "factor realisations."
    factor_scores = pd.DataFrame(
        Z @ top_eigenvectors,
        index   = window.index,
        columns = [f"PC{i+1}" for i in range(k)],
    )

    return {
        "eigenvalues":   top_eigenvalues,
        "eigenvectors":  top_eigenvectors,
        "var_explained": var_explained,
        "cum_var":       cum_var,
        "factor_scores": factor_scores,
        "scaler":        scaler,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Residual extraction (inner window)
# ══════════════════════════════════════════════════════════════════════════════

def extract_residuals(
    changes_window: pd.DataFrame,
    factor_scores:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each instrument, regress its yield changes on the k factor scores.
    Return the cumulated residuals X_i(t) and the OLS betas.

    Regression:  Δy_i = β_i1 F1 + β_i2 F2 + β_i3 F3 + ε_i
    (no intercept — factors are zero-mean by construction)

    Parameters
    ----------
    changes_window : DataFrame (T × N)  — yield changes over inner window
    factor_scores  : DataFrame (T × k)  — PC scores over the same dates

    Returns
    -------
    residuals  : DataFrame (T × N)   — raw daily residuals  ε_i(t)
    cumresid   : DataFrame (T × N)   — cumulated residuals  X_i(t) = Σ ε
    betas      : DataFrame (N × k)   — factor loadings
    r_squared  : Series   (N,)       — R² of each instrument's factor model
    """
    # Align on common dates
    common     = changes_window.index.intersection(factor_scores.index)
    dy         = changes_window.loc[common].values    # (T × N)
    F          = factor_scores.loc[common].values     # (T × k)

    # ── OLS:  β = (F'F)^{-1} F' dy ──────────────────────────────────────
    # For each instrument i, solve:  Δy_i = β_{i1}F_1 + β_{i2}F_2 + β_{i3}F_3
    # No intercept because the PCA factors are zero-mean by construction.
    # lstsq handles the matrix case (all N instruments simultaneously).
    betas      = np.linalg.lstsq(F, dy, rcond=None)[0]   # (k × N)
    fitted     = F @ betas                                 # (T × N) — PCA-implied moves
    residuals  = dy - fitted                               # (T × N) — idiosyncratic moves

    # R² per instrument
    ss_res = (residuals ** 2).sum(axis=0)
    ss_tot = ((dy - dy.mean(axis=0)) ** 2).sum(axis=0)
    r2     = np.where(ss_tot > 0, 1 - ss_res / ss_tot, np.nan)

    col_names  = changes_window.columns.tolist()
    resid_df   = pd.DataFrame(residuals,   index=common, columns=col_names)
    # Cumulate: X_i(t) = Σ_{s≤t} ε_i(s)
    # This integral of the daily idiosyncratic residuals is the "spread" of
    # the instrument's yield from its PCA-implied fair value.
    # It is stationary (mean-reverting) and is the input to the OU model.
    cumresid   = resid_df.cumsum()
    betas_df   = pd.DataFrame(betas.T, index=col_names,
                              columns=[f"PC{i+1}" for i in range(betas.shape[0])])
    r2_series  = pd.Series(r2, index=col_names)

    return resid_df, cumresid, betas_df, r2_series


# ══════════════════════════════════════════════════════════════════════════════
#  Rolling PCA — main function
# ══════════════════════════════════════════════════════════════════════════════

def rolling_pca(
    changes:       pd.DataFrame,
    k:             int   = 3,
    corr_window:   int   = 252,
    resid_window:  int   = 60,
    step:          int   = 1,
    verbose:       bool  = True,
    train_end:     "pd.Timestamp | None" = None,
) -> dict:
    """
    Run rolling PCA across the full history of yield changes.

    For each date t (with sufficient history):
      - Fit PCA on changes[t - corr_window : t]
      - Extract factor scores and residuals on changes[t - resid_window : t]

    train_end — prevents test-set leakage into PCA fitting
    ─────────────────────────────────────────────────────
    When train_end is set:
      - Dates ≤ train_end: PCA loadings are fitted normally (rolling window)
      - Dates > train_end: the last training-period PCA (fitted at train_end)
        is FROZEN and applied forward to generate test-period residuals.

    This means the PCA model never refits using test-period yield data.
    The test-period residuals are therefore genuinely out-of-sample — they
    measure how each bond deviates from a model estimated only on past data.

    Parameters
    ----------
    changes       : DataFrame (T × N) — daily yield changes, all instruments
    k             : number of PCs to retain
    corr_window   : lookback window for correlation matrix (default 252 = 1 year)
    resid_window  : inner window for residual estimation (default 60 days)
    step          : roll every `step` days (1 = daily, 5 = weekly)
    verbose       : print progress
    train_end     : last date of training split; PCA freezes after this date.
                    None = refit rolling PCA on all dates (original behaviour).

    Returns
    -------
    dict of DataFrames — see module docstring for keys.
    """
    n_dates = len(changes)
    dates   = changes.index
    cols    = changes.columns.tolist()
    N       = len(cols)

    if train_end is not None:
        n_train = int((dates <= train_end).sum())
        print(f"  PCA will fit rolling windows up to {train_end.date()} "
              f"({n_train:,} days), then freeze loadings for test period.")

    # Output containers
    ev_records     = []
    ve_records     = []
    stab_records   = []
    score_records  = []
    resid_records  = []
    r2_records     = []
    loadings_dict  = {}
    prev_vecs      = None

    # State for frozen-loadings mode (used after train_end)
    frozen_pca_res  = None   # last training-period PCA result
    frozen_cols     = None   # columns used in last training window

    n_processed = 0
    for i in range(corr_window, n_dates, step):
        t = dates[i]

        # ── Decide whether to refit PCA or use frozen loadings ───────────────
        # Once t crosses train_end, the PCA model is frozen. This prevents
        # test-period yield data from influencing the eigenvectors used to
        # generate signals — a key no-lookahead requirement. Any date t in the
        # test period uses the SAME loadings as the last training-period window.
        past_train_end = (train_end is not None) and (t > train_end)

        if past_train_end and frozen_pca_res is not None:
            # FROZEN MODE: apply last training-period PCA to the current window
            pca_res  = frozen_pca_res
            win_corr_clean_cols = frozen_cols
        # ── Correlation-window PCA (or frozen loadings for test period) ─────────
        win_corr = changes.iloc[i - corr_window : i]

        if past_train_end and frozen_pca_res is not None:
            # FROZEN MODE — reuse last training-period PCA; do not refit.
            # cos_sims = 1.0 because the eigenvectors haven't changed.
            win_corr_clean = changes.iloc[i - corr_window : i][frozen_cols].dropna(axis=1)
            pca_res        = frozen_pca_res
            cos_sims       = [1.0] * k   # loadings unchanged by definition
        else:
            # NORMAL MODE — fit a new PCA on the current correlation window.
            if win_corr.dropna(axis=1, how="any").shape[1] < k + 1:
                continue
            win_corr_clean = win_corr.dropna(axis=1)
            if win_corr_clean.shape[1] < k + 1:
                continue

            pca_res = run_pca_on_window(win_corr_clean, k=k)

            # Eigenvector stability: cos θ_j = |v_t · v_{t-1}| ∈ [0, 1].
            # Near 1 = loadings barely rotated (stable factor structure).
            # A sudden drop is the key trigger for a BAD regime.
            vecs = pca_res["eigenvectors"]
            cos_sims = [np.nan] * k
            if prev_vecs is not None and prev_vecs.shape == vecs.shape:
                for j in range(k):
                    cos_sims[j] = abs(float(np.dot(vecs[:, j], prev_vecs[:, j])))
            prev_vecs = vecs.copy()

            # Keep updating frozen_pca_res until train_end is reached.
            # After train_end, this value is never overwritten, so the last
            # training-period PCA is carried forward for all test-period dates.
            if train_end is not None and t <= train_end:
                frozen_pca_res = pca_res
                frozen_cols    = list(win_corr_clean.columns)

        # ── Residual-window extraction ────────────────────────────────────────
        # Use whichever columns survived the correlation window (train or frozen).
        available_cols = list(win_corr_clean.columns) if not past_train_end \
                         else frozen_cols
        win_resid = changes.iloc[max(0, i - resid_window) : i][available_cols]
        win_resid = win_resid.dropna(how="all")

        # Recompute factor scores for the residual window using current/frozen loadings
        # (the stored pca_res["factor_scores"] covers the corr_window, not resid_window)
        if len(win_resid) >= 2:
            scaler = pca_res["scaler"]
            Z_resid = scaler.transform(win_resid.values)
            V_k     = pca_res["eigenvectors"]  # (N_clean × k)
            F_resid = pd.DataFrame(
                Z_resid @ V_k,
                index=win_resid.index,
                columns=[f"PC{j+1}" for j in range(k)],
            )
        else:
            F_resid = pca_res["factor_scores"].loc[
                pca_res["factor_scores"].index.isin(win_resid.index)
            ]
        scores_in_resid = F_resid

        resid_daily, cumresid, betas_df, r2 = extract_residuals(
            win_resid, scores_in_resid
        )

        # Latest values (at date t)
        last_cum = cumresid.iloc[-1] if len(cumresid) else pd.Series(np.nan, index=cols)
        last_r2  = r2.mean() if not r2.isna().all() else np.nan
        last_score = scores_in_resid.iloc[-1] if len(scores_in_resid) else \
                     pd.Series(np.nan, index=[f"PC{j+1}" for j in range(k)])

        # ── Record ────────────────────────────────────────────────────────────
        ev_records.append([t] + pca_res["eigenvalues"].tolist())
        ve_records.append([t] + pca_res["var_explained"].tolist() +
                          [pca_res["cum_var"]])
        stab_records.append([t] + cos_sims)
        score_records.append(pd.Series([t] + last_score.tolist(),
                             index=["date"] + [f"PC{j+1}" for j in range(k)]))
        resid_records.append(last_cum.rename(t))
        r2_records.append((t, last_r2))
        loadings_dict[t] = betas_df

        n_processed += 1
        if verbose and n_processed % 100 == 0:
            print(f"  Processed {n_processed} windows … (date: {t.date()})")

    print(f"  Rolling PCA complete — {n_processed} windows processed.")

    # ── Assemble output DataFrames ────────────────────────────────────────────
    pc_labels = [f"PC{j+1}" for j in range(k)]

    eigenvalues_df = pd.DataFrame(ev_records, columns=["date"] + pc_labels
                     ).set_index("date")

    var_exp_cols   = pc_labels + ["cumulative"]
    var_exp_df     = pd.DataFrame(ve_records,
                     columns=["date"] + var_exp_cols).set_index("date")

    stab_df        = pd.DataFrame(stab_records,
                     columns=["date"] + [f"{p}_cos" for p in pc_labels]
                     ).set_index("date")

    score_df       = pd.DataFrame(score_records).set_index("date")

    resid_df       = pd.DataFrame(resid_records)
    resid_df.index.name = "date"

    r2_series      = pd.Series(dict(r2_records), name="r_squared")
    r2_series.index.name = "date"

    return {
        "eigenvalues":   eigenvalues_df,
        "var_explained": var_exp_df,
        "ev_stability":  stab_df,
        "factor_scores": score_df,
        "residuals":     resid_df,
        "r_squared":     r2_series,
        "loadings":      loadings_dict,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Save / Load helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_pca_results(results: dict, out_dir: str = None) -> None:
    if out_dir is None:
        out_dir = os.path.join(OUTPUT_DIR, "pca")
    os.makedirs(out_dir, exist_ok=True)

    for key, val in results.items():
        if key == "loadings":
            continue  # too many files; loadings accessed from dict in memory
        if isinstance(val, (pd.DataFrame, pd.Series)):
            val.to_csv(os.path.join(out_dir, f"{key}.csv"))

    print(f"  PCA results saved to {out_dir}/")


def load_pca_results(out_dir: str = None) -> dict:
    if out_dir is None:
        out_dir = os.path.join(OUTPUT_DIR, "pca")

    keys = ["eigenvalues", "var_explained", "ev_stability",
            "factor_scores", "residuals", "r_squared"]
    results = {}
    for key in keys:
        path = os.path.join(out_dir, f"{key}.csv")
        if os.path.exists(path):
            results[key] = pd.read_csv(path, index_col=0, parse_dates=True)
    results["loadings"] = {}  # not persisted
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from src.data_prep import load_cross_market, load_yield_changes

    parser = argparse.ArgumentParser(description="Run rolling PCA on multi-country yields.")
    parser.add_argument("--mode", choices=["country", "cross"], default="cross",
                        help="'country' = per-country PCA | 'cross' = global panel PCA")
    parser.add_argument("--country",      default="US",
                        help="Country code for --mode country (e.g. US, DE, UK, JP)")
    parser.add_argument("--corr-window",  type=int, default=252)
    parser.add_argument("--resid-window", type=int, default=60)
    parser.add_argument("--k",            type=int, default=3)
    args = parser.parse_args()

    if args.mode == "cross":
        print("Loading cross-market panel …")
        changes = load_cross_market()
    else:
        print(f"Loading per-country yield changes for {args.country} …")
        changes_multi = load_yield_changes()
        changes = changes_multi.xs(args.country, axis=1, level="country").dropna(how="all")

    print(f"Running rolling PCA | mode={args.mode} | k={args.k} | "
          f"corr_window={args.corr_window} | resid_window={args.resid_window}")
    results = rolling_pca(
        changes,
        k=args.k,
        corr_window=args.corr_window,
        resid_window=args.resid_window,
    )
    save_pca_results(results)

    print("\nVariance explained (last 5 rows):")
    print(results["var_explained"].tail(5).to_string())
    print("\n✓ Done.")
