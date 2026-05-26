"""
03_rolling_pca.py
=================
Rolling PCA engine — the analytical core of the project.

What this module does
---------------------
For each date t in the dataset:
  1. Takes a lookback window of `corr_window` days of yield changes
  2. Standardises the changes (zero-mean, unit-variance)
  3. Computes the empirical correlation matrix
  4. Eigen-decomposes it, keeping top-k eigenvectors
  5. Records:
       - eigenvalues (λ₁, λ₂, λ₃) and cumulative variance explained
       - eigenvector stability: cosine similarity to previous window
       - factor scores (PC₁, PC₂, PC₃) for each day in the window
       - residuals for each maturity after projecting out factors

Key design choices
------------------
- PCA is run on *changes* (stationary), not levels
- We use the *correlation* matrix (not covariance) so high-vol tenors
  don't dominate; this is the standard approach in PCA Unleashed
- Rolling window of 252 business days (≈ 1 year)
- We record eigenvector angles between consecutive windows:
    cos θ = v_t · v_{t-1}  (should be close to 1 when stable)

Outputs
-------
  results dict:
    'loadings'         : dict {date → np.array (n_tenors × k)}
    'eigenvalues'      : DataFrame (date × k)
    'var_explained'    : DataFrame (date × ['PC1','PC2','PC3','cumulative'])
    'ev_stability'     : DataFrame (date × ['PC1_cos','PC2_cos','PC3_cos'])
    'factor_scores'    : DataFrame (date × ['PC1','PC2','PC3'])
    'residuals'        : DataFrame (date × n_instruments) — cumulated residuals
    'r_squared'        : Series (date) — R² of factor model each day
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
    # Standardise within window
    scaler    = StandardScaler()
    Z         = scaler.fit_transform(window.values)  # shape (T, N)
    T, N      = Z.shape

    # Correlation matrix  C = Z'Z / (T-1)
    C         = (Z.T @ Z) / (T - 1)

    # Eigen-decomposition — numpy returns in ascending order, so reverse
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]   # (N × N), columns = eigenvectors

    # Retain top k
    top_eigenvalues  = eigenvalues[:k]
    top_eigenvectors = eigenvectors[:, :k]   # (N × k)

    var_explained = top_eigenvalues / eigenvalues.sum()
    cum_var       = var_explained.sum()

    # Factor scores  F = Z · V_k   shape (T × k)
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

    # OLS:  β = (F'F)^{-1} F' dy
    betas      = np.linalg.lstsq(F, dy, rcond=None)[0]   # (k × N)
    fitted     = F @ betas                                 # (T × N)
    residuals  = dy - fitted                               # (T × N)

    # R² per instrument
    ss_res = (residuals ** 2).sum(axis=0)
    ss_tot = ((dy - dy.mean(axis=0)) ** 2).sum(axis=0)
    r2     = np.where(ss_tot > 0, 1 - ss_res / ss_tot, np.nan)

    col_names  = changes_window.columns.tolist()
    resid_df   = pd.DataFrame(residuals,   index=common, columns=col_names)
    cumresid   = resid_df.cumsum()
    betas_df   = pd.DataFrame(betas.T, index=col_names,
                              columns=[f"PC{i+1}" for i in range(betas.shape[0])])
    r2_series  = pd.Series(r2, index=col_names)

    return resid_df, cumresid, betas_df, r2_series


# ══════════════════════════════════════════════════════════════════════════════
#  Rolling PCA — main function
# ══════════════════════════════════════════════════════════════════════════════

def rolling_pca(
    changes: pd.DataFrame,
    k:             int   = 3,
    corr_window:   int   = 252,
    resid_window:  int   = 60,
    step:          int   = 1,
    verbose:       bool  = True,
) -> dict:
    """
    Run rolling PCA across the full history of yield changes.

    For each date t (with sufficient history):
      - Fit PCA on changes[t - corr_window : t]
      - Extract factor scores and residuals on changes[t - resid_window : t]

    Parameters
    ----------
    changes       : DataFrame (T × N) — daily yield changes, all instruments
    k             : number of PCs to retain
    corr_window   : lookback window for correlation matrix (default 252 = 1 year)
    resid_window  : inner window for OU residual estimation (default 60 days)
    step          : roll every `step` days (1 = daily, 5 = weekly)
    verbose       : print progress

    Returns
    -------
    dict of DataFrames — see module docstring for keys.
    """
    n_dates = len(changes)
    dates   = changes.index
    cols    = changes.columns.tolist()
    N       = len(cols)

    # Output containers
    ev_records     = []    # eigenvalues per date
    ve_records     = []    # variance explained per date
    stab_records   = []    # eigenvector stability per date
    score_records  = []    # latest factor score per date
    resid_records  = []    # last residual per date (X_i at t)
    r2_records     = []    # R² per date
    loadings_dict  = {}    # date → eigenvector matrix
    prev_vecs      = None  # for computing angle stability

    n_processed = 0
    for i in range(corr_window, n_dates, step):
        t = dates[i]

        # ── Correlation-window PCA ────────────────────────────────────────────
        win_corr   = changes.iloc[i - corr_window : i]
        if win_corr.dropna(axis=1, how="any").shape[1] < k + 1:
            continue   # not enough instruments

        # Drop columns with any NaN in this window
        win_corr_clean = win_corr.dropna(axis=1)
        if win_corr_clean.shape[1] < k + 1:
            continue

        pca_res = run_pca_on_window(win_corr_clean, k=k)

        # ── Eigenvector stability (cosine similarity to previous window) ──────
        vecs = pca_res["eigenvectors"]   # (N × k) — may be subset if columns dropped
        cos_sims = [np.nan] * k
        if prev_vecs is not None and prev_vecs.shape == vecs.shape:
            for j in range(k):
                cos_sims[j] = abs(float(np.dot(vecs[:, j], prev_vecs[:, j])))
        prev_vecs = vecs.copy()

        # ── Residual-window extraction ────────────────────────────────────────
        win_resid = changes.iloc[max(0, i - resid_window) : i][win_corr_clean.columns]
        win_resid = win_resid.dropna(how="all")
        scores_in_resid = pca_res["factor_scores"].loc[
            pca_res["factor_scores"].index.isin(win_resid.index)
        ]

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
