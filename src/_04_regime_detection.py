"""
04_regime_detection.py
======================
3-regime classifier on top of the rolling PCA stability metrics.

Regimes
-------
  0 — GOOD      : PCA structure is stable; top-3 PCs explain the curve well;
                  eigenvectors are nearly unchanged from the previous window.
                  → Mean reversion expected to work; trade at full size.

  1 — NEUTRAL   : Moderate instability; factor structure is shifting but not
                  broken. Variance explained is in the borderline zone or
                  eigenvector angles are drifting.
                  → Trade at reduced size (50%) or hold existing positions.

  2 — BAD       : Regime shift / structural break; top-3 PCs fail to capture
                  the curve; eigenvectors have rotated significantly.
                  → Exit all positions; do not open new ones.

Two detection methods are available (select via `method` parameter):

A. Rule-based (method='rules') — fast, interpretable, no training data needed.
   Thresholds on three PCA stability metrics:
     - cum_var_explained  (target > 0.99)
     - mean eigenvector cosine similarity  (target > 0.97)
     - rolling R² of the factor model  (target > 0.85)

B. Hidden Markov Model (method='hmm') — data-driven, probabilistic.
   Fits a 3-state Gaussian HMM on the stability feature vector.
   States are relabelled by their cumulative-variance rank:
     highest cum_var → GOOD (0), middle → NEUTRAL (1), lowest → BAD (2).

Outputs
-------
  'regime'          : Series (date → int 0/1/2) — raw label
  'regime_label'    : Series (date → str)       — 'GOOD'/'NEUTRAL'/'BAD'
  'regime_proba'    : DataFrame (date × 3)      — HMM state probabilities (hmm only)
  'features'        : DataFrame — the stability features fed into the classifier
"""

import os
import warnings
from typing import Literal

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

REGIME_NAMES = {0: "GOOD", 1: "NEUTRAL", 2: "BAD"}
REGIME_COLOURS = {"GOOD": "#2ecc71", "NEUTRAL": "#f39c12", "BAD": "#e74c3c"}


# ══════════════════════════════════════════════════════════════════════════════
#  Feature engineering from PCA results
# ══════════════════════════════════════════════════════════════════════════════

def build_stability_features(
    pca_results: dict,
    smooth_window: int = 21,
) -> pd.DataFrame:
    """
    Construct the feature DataFrame used by both regime methods.

    Features
    --------
    cum_var          : cumulative variance explained by top-k PCs
    mean_cos_sim     : average cosine similarity of eigenvectors across PCs
    min_cos_sim      : minimum (worst-case) cosine similarity across PCs
    r_squared        : average R² of factor model across instruments
    var_pc1          : fraction of variance explained by PC1 alone
    var_slope_change : absolute 21-day change in cum_var (detects rapid drops)

    All features are smoothed with a rolling median (smooth_window) to reduce
    noise before classification.
    """
    var_exp  = pca_results["var_explained"]   # cols: PC1, PC2, PC3, cumulative
    stab     = pca_results["ev_stability"]    # cols: PC1_cos, PC2_cos, PC3_cos
    r2       = pca_results["r_squared"]       # Series

    # Align on common dates
    common = var_exp.index.intersection(stab.index)
    if isinstance(r2, pd.DataFrame):
        r2 = r2.iloc[:, 0]
    common = common.intersection(r2.index)

    feats = pd.DataFrame(index=common)
    feats["cum_var"]      = var_exp.loc[common, "cumulative"]
    feats["var_pc1"]      = var_exp.loc[common, "PC1"]
    feats["mean_cos_sim"] = stab.loc[common].mean(axis=1)
    feats["min_cos_sim"]  = stab.loc[common].min(axis=1)
    feats["r_squared"]    = r2.loc[common]

    # Pace of change — large negative value = rapid structural shift
    feats["var_slope_change"] = (
        feats["cum_var"]
        .diff(smooth_window)
        .abs()
    )

    # Smooth with rolling median to reduce daily noise
    feats = feats.rolling(smooth_window, min_periods=max(1, smooth_window // 2)).median()
    feats = feats.dropna()

    return feats


# ══════════════════════════════════════════════════════════════════════════════
#  Method A — Rule-based regime classification
# ══════════════════════════════════════════════════════════════════════════════

# Tunable thresholds — calibrated for CROSS-MARKET PCA (15 instruments,
# 4 countries).  Cross-market PCA typically explains ~75-80% with top-3 PCs
# (not 99%+ as in single-country PCA), so thresholds are set relative to the
# observed distribution: cum_var ≈ 0.69–0.84, mean_cos_sim ≈ 0.999.
RULE_THRESHOLDS = {
    "cum_var_good":    0.800,   # top-3 PCs explain ≥ 80.0% → potential GOOD
    "cum_var_neutral": 0.740,   # 74.0% – 80.0% → NEUTRAL
    # below 74.0% → BAD (structural break / heavy turbulence)

    "cos_good":    0.9990,      # avg eigenvector cos-sim ≥ 0.999 → very stable
    "cos_neutral": 0.9950,      # 0.995 – 0.999 → moderate stability
    # below 0.995 → bad eigenvector signal

    "r2_good":    0.80,         # R² ≥ 0.80 → factor model fits well
    "r2_neutral": 0.60,         # 0.60 – 0.80 → borderline
    # below 0.60 → bad signal
}


def _rule_score(row: pd.Series, thresholds: dict) -> int:
    """
    Compute a regime vote (0 = GOOD, 2 = BAD) from a single feature row.
    Uses majority-voting across three sub-signals (cum_var, cos_sim, r2).
    """
    votes = []

    # Variance explained signal
    if row["cum_var"] >= thresholds["cum_var_good"]:
        votes.append(0)
    elif row["cum_var"] >= thresholds["cum_var_neutral"]:
        votes.append(1)
    else:
        votes.append(2)

    # Eigenvector stability signal
    if row["mean_cos_sim"] >= thresholds["cos_good"]:
        votes.append(0)
    elif row["mean_cos_sim"] >= thresholds["cos_neutral"]:
        votes.append(1)
    else:
        votes.append(2)

    # R² signal
    if row["r_squared"] >= thresholds["r2_good"]:
        votes.append(0)
    elif row["r_squared"] >= thresholds["r2_neutral"]:
        votes.append(1)
    else:
        votes.append(2)

    # Rapid variance drop → override to BAD regardless of other signals
    if row.get("var_slope_change", 0) > 0.03:
        return 2

    # Majority vote: if any two signals agree, use that
    from collections import Counter
    count = Counter(votes)
    return count.most_common(1)[0][0]


def classify_rule_based(
    features: pd.DataFrame,
    thresholds: dict = None,
) -> pd.Series:
    """
    Classify each date into regime 0/1/2 using rule-based thresholds.
    Returns integer Series.
    """
    if thresholds is None:
        thresholds = RULE_THRESHOLDS
    return features.apply(lambda row: _rule_score(row, thresholds), axis=1)


# ══════════════════════════════════════════════════════════════════════════════
#  Feature normalisation helper
# ══════════════════════════════════════════════════════════════════════════════

def normalize_features_percentile(
    features: pd.DataFrame,
    min_periods: int = 60,
) -> pd.DataFrame:
    """
    Convert each feature column to its expanding-window percentile rank [0, 1].

    WHY PERCENTILE RANKS INSTEAD OF RAW VALUES?
    ─────────────────────────────────────────────
    A Gaussian HMM assumes each feature is drawn from one of n_states Gaussian
    distributions.  It finds states by separating the data into clusters of
    similar feature values.

    Cross-market PCA features live in a narrow absolute range:
      cum_var    ≈ 0.69 – 0.84   (total range of only 0.15!)
      mean_cos   ≈ 0.998 – 1.000 (range of 0.002!)

    In absolute terms, the HMM cannot meaningfully separate regimes because
    all 6,000 data points look nearly identical.  But in *relative* terms,
    the top 30% of cum_var days are clearly different from the bottom 30%.

    Expanding percentile rank maps each day to its position in the historical
    distribution so far: 0 = lowest ever seen, 1 = highest ever seen.
    This is look-ahead-bias-free (expanding, not full-sample) and gives the
    HMM well-separated [0,1] inputs regardless of the absolute feature level.

    The same approach is standard in cross-sectional equity factor models
    (e.g. ranking stocks by P/E within a universe rather than using raw P/E).
    """
    return features.expanding(min_periods=min_periods).rank(pct=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Method B — Gaussian Hidden Markov Model
# ══════════════════════════════════════════════════════════════════════════════

def classify_hmm(
    features: pd.DataFrame,
    n_states:     int   = 3,
    n_iter:       int   = 200,
    random_state: int   = 42,
    feature_cols: list  = None,
    normalize:    bool  = True,
) -> tuple[pd.Series, pd.DataFrame, object]:
    """
    Fit a Gaussian HMM on the stability features and decode the most likely
    state sequence.

    States are relabelled so that:
      state with highest mean cum_var → regime 0 (GOOD)
      middle state                    → regime 1 (NEUTRAL)
      state with lowest  mean cum_var → regime 2 (BAD)

    Parameters
    ----------
    features     : DataFrame from build_stability_features()
    n_states     : number of HMM states (should be 3)
    n_iter       : EM training iterations
    random_state : reproducibility seed
    feature_cols : subset of feature columns to feed to HMM
                   (default: all except var_slope_change which is noisy)

    Returns
    -------
    regime_series : Series (date → int 0/1/2)
    regime_proba  : DataFrame (date × n_states)  — posterior probabilities
    model         : fitted GaussianHMM object
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        raise ImportError("hmmlearn is required. Run: pip install hmmlearn")

    if feature_cols is None:
        feature_cols = ["cum_var", "mean_cos_sim", "r_squared", "var_pc1"]

    feat_input = features.copy()
    if normalize:
        # Convert absolute levels to expanding percentile ranks [0,1].
        # This is essential for cross-market PCA where cum_var lives in a
        # narrow range (0.69–0.84) that would confuse a Gaussian HMM trained
        # on absolute values.  Percentile ranks expose *relative* variation
        # and let the HMM find genuine structural regimes.
        feat_input = normalize_features_percentile(feat_input).dropna()
        # Re-align index with features (dropna may shorten)
        feat_input = feat_input.loc[feat_input.index.intersection(features.index)]

    X = feat_input[feature_cols].values.astype(float)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
    )
    model.fit(X)

    # Viterbi decoding — most likely state sequence
    raw_states = model.predict(X)

    # Posterior probabilities
    log_proba   = model.predict_proba(X)
    proba_df    = pd.DataFrame(
        log_proba,
        index   = feat_input.index,
        columns = [f"state_{s}" for s in range(n_states)],
    )

    # ── Relabel states by cum_var rank ────────────────────────────────────────
    # HMM states are arbitrary integers (0,1,2) with no inherent ordering.
    # We assign semantic meaning by ranking the states by their average
    # raw (un-normalised) cum_var:
    #   highest mean cum_var → regime 0 = GOOD  (PCA structure intact)
    #   middle               → regime 1 = NEUTRAL
    #   lowest mean cum_var  → regime 2 = BAD   (PCA structure broken)
    # Using raw (not normalised) cum_var ensures the label is stable across
    # different datasets and time periods.
    raw_cum_var = features.loc[feat_input.index, "cum_var"]
    state_means = {}
    for s in range(n_states):
        mask = raw_states == s
        if mask.any():
            state_means[s] = raw_cum_var.iloc[mask].mean()
        else:
            state_means[s] = 0.0

    # Sort: highest cum_var → GOOD (0), middle → NEUTRAL (1), lowest → BAD (2)
    sorted_states = sorted(state_means, key=state_means.get, reverse=True)
    state_map     = {orig: new for new, orig in enumerate(sorted_states)}
    relabelled    = np.vectorize(state_map.get)(raw_states)

    regime_series = pd.Series(relabelled, index=feat_input.index, name="regime")

    # Relabel proba columns accordingly
    proba_df.columns = [f"state_{state_map[int(c.split('_')[1])]}"
                        for c in proba_df.columns]
    proba_df = proba_df.sort_index(axis=1)

    return regime_series, proba_df, model


# ══════════════════════════════════════════════════════════════════════════════
#  Main detection function
# ══════════════════════════════════════════════════════════════════════════════

def detect_regimes(
    pca_results:   dict,
    method:        Literal["rules", "hmm"] = "hmm",
    smooth_window: int  = 21,
    thresholds:    dict = None,
    hmm_feature_cols: list = None,
) -> dict:
    """
    End-to-end regime detection from PCA results.

    Parameters
    ----------
    pca_results   : dict returned by rolling_pca()
    method        : 'rules' (fast) or 'hmm' (data-driven)
    smooth_window : rolling median window for feature smoothing
    thresholds    : custom thresholds for rule-based method
    hmm_feature_cols : custom feature subset for HMM

    Returns
    -------
    dict:
      'features'      : DataFrame — smoothed stability features
      'regime'        : Series (int 0/1/2)
      'regime_label'  : Series (str 'GOOD'/'NEUTRAL'/'BAD')
      'regime_proba'  : DataFrame — posterior probs (HMM only; else None)
      'model'         : fitted HMM object (HMM only; else None)
      'method'        : str — which method was used
      'stats'         : dict — regime frequency counts and durations
    """
    print(f"  Building stability features (smooth_window={smooth_window}) …")
    features = build_stability_features(pca_results, smooth_window=smooth_window)
    print(f"  Features shape: {features.shape}")

    regime_proba = None
    model        = None

    if method == "rules":
        print("  Classifying regimes (rule-based) …")
        regime = classify_rule_based(features, thresholds=thresholds)

    elif method == "hmm":
        print("  Classifying regimes (Gaussian HMM) …")
        regime, regime_proba, model = classify_hmm(
            features,
            feature_cols=hmm_feature_cols,
        )
    else:
        raise ValueError(f"method must be 'rules' or 'hmm', got '{method}'")

    regime_label = regime.map(REGIME_NAMES).rename("regime_label")

    # ── Statistics ────────────────────────────────────────────────────────────
    counts    = regime.value_counts().sort_index()
    stats     = {
        "counts":      counts.to_dict(),
        "frequencies": (counts / len(regime)).to_dict(),
        "avg_duration": _avg_run_length(regime),
    }

    print("\n  Regime summary:")
    for r, name in REGIME_NAMES.items():
        pct = stats["frequencies"].get(r, 0) * 100
        dur = stats["avg_duration"].get(r, 0)
        print(f"    {name:8s} (regime {r}): {pct:5.1f}%  |  avg run {dur:.0f} days")

    return {
        "features":     features,
        "regime":       regime,
        "regime_label": regime_label,
        "regime_proba": regime_proba,
        "model":        model,
        "method":       method,
        "stats":        stats,
    }


# ── Helper ─────────────────────────────────────────────────────────────────────

def _avg_run_length(s: pd.Series) -> dict:
    """Compute average consecutive run length per state."""
    runs = {}
    current, count = s.iloc[0], 1
    all_runs = {v: [] for v in s.unique()}
    for val in s.iloc[1:]:
        if val == current:
            count += 1
        else:
            all_runs[current].append(count)
            current, count = val, 1
    all_runs[current].append(count)
    return {k: (np.mean(v) if v else 0) for k, v in all_runs.items()}


def save_regime_results(regime_dict: dict, out_dir: str = None) -> None:
    if out_dir is None:
        out_dir = os.path.join(OUTPUT_DIR, "regimes")
    os.makedirs(out_dir, exist_ok=True)

    regime_dict["regime"].to_csv(os.path.join(out_dir, "regime.csv"))
    regime_dict["regime_label"].to_csv(os.path.join(out_dir, "regime_label.csv"))
    regime_dict["features"].to_csv(os.path.join(out_dir, "features.csv"))
    if regime_dict["regime_proba"] is not None:
        regime_dict["regime_proba"].to_csv(os.path.join(out_dir, "regime_proba.csv"))
    print(f"  Regime results saved to {out_dir}/")


def load_regime_results(out_dir: str = None) -> dict:
    if out_dir is None:
        out_dir = os.path.join(OUTPUT_DIR, "regimes")
    out = {}
    for fname, key in [("regime.csv", "regime"), ("regime_label.csv", "regime_label"),
                       ("features.csv", "features"), ("regime_proba.csv", "regime_proba")]:
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            out[key] = df.iloc[:, 0] if df.shape[1] == 1 else df
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, BASE_DIR)
    from src.rolling_pca import load_pca_results

    parser = argparse.ArgumentParser(description="Detect PCA-based market regimes.")
    parser.add_argument("--method", choices=["rules", "hmm"], default="hmm")
    parser.add_argument("--smooth", type=int, default=21,
                        help="Rolling median window for feature smoothing")
    args = parser.parse_args()

    print("Loading PCA results …")
    pca_results = load_pca_results()

    regime_dict = detect_regimes(
        pca_results,
        method=args.method,
        smooth_window=args.smooth,
    )
    save_regime_results(regime_dict)
    print("\n✓ Done.")
