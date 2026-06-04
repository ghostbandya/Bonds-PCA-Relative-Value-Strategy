"""
config.py
=========
Single source of truth for the multi-country PCA relative-value strategy.
Adapted from Jay's US-only config to the 4-country cross-market universe.

All load-bearing conventions are pinned here so the grid, backtest and tests
cannot drift apart.

Key differences from Jay's config
------------------------------------
- Universe spans US, DE, UK, JP across 7 common tenors
- JP is included in PCA for better factor estimation but EXCLUDED from trading
  (BoJ YCC 2012-2024 suppressed free-float mean reversion in JGBs)
- DV01_TENOR_YEARS maps COUNTRY_TENOR -> years (e.g. "US_10Y" -> 10.0)
- Contract face values and half-spreads are approximated for non-US markets
- GAMMA is NOT hardcoded — use calibrate_gamma() in weights.py on the TRAIN split
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
# All tenors present in the cross-market panel (country_tenor format).
# Exact set is determined at runtime from available data; this is the target.
COMMON_TENORS: tuple[str, ...] = ("1Y", "2Y", "3Y", "5Y", "10Y", "20Y", "30Y")
PCA_COUNTRIES:   tuple[str, ...] = ("US", "DE", "UK", "JP")
TRADE_COUNTRIES: tuple[str, ...] = ("US", "DE", "UK")   # JP excluded (BoJ YCC)
N_PC: int = 3   # number of principal components to retain

# Map COUNTRY_TENOR -> maturity in years; used by DV01 engine and carry signal.
def tenor_to_years(col: str) -> float:
    """Extract maturity in years from a column label like "US_10Y" or "DE_2Y"."""
    tenor = col.split("_", 1)[-1]   # "US_10Y" -> "10Y"
    return float(tenor.replace("Y", ""))

# ---------------------------------------------------------------------------
# Vol-targeting (mirrors Jay's §3.5.4)
# ---------------------------------------------------------------------------
TRADING_DAYS: int = 252
VOL_TARGET_ANN: float = 0.10          # 10% annualised vol target
VOL_WINDOW:     int   = 63            # trailing realised-vol window (days)
L_MAX:          float = 10.0          # max leverage scalar k_t
# Floor: caps leverage at L_MAX even when measured vol -> 0
SIGMA_FLOOR: float = 0.1 * (VOL_TARGET_ANN / np.sqrt(TRADING_DAYS))

# ---------------------------------------------------------------------------
# Factor-neutrality tolerance
# ---------------------------------------------------------------------------
NEUTRAL_TOL: float = 1e-8    # ||V g||_inf assertion threshold
RHO_MIN:     float = 1e-3    # residual-room guard (skip degenerate tenors)
RIDGE_EPS:   float = 1e-10   # ridge fallback for ill-conditioned Sigma / KKT

# ---------------------------------------------------------------------------
# Transaction cost assumptions (adapted for multi-country futures)
# ---------------------------------------------------------------------------
# Contract face values by tenor (approximate, in USD equivalent).
# US values match CME spec; DE/UK/JP use approximate DV01-equivalent sizing.
FUTURES_CONTRACT_FACE: dict[str, float] = {
    "1Y":  200_000.0,
    "2Y":  200_000.0,
    "3Y":  200_000.0,
    "5Y":  100_000.0,
    "10Y": 100_000.0,
    "20Y": 100_000.0,
    "30Y": 100_000.0,
}

# Default capital (USD) for contract sizing.
DEFAULT_CAPITAL_USD: float = 100_000_000.0

# Pre-registered one-way yield half-spreads (bp) by tenor.
# US values from CME futures tick/DV01; non-US slightly wider.
# These are ONE-WAY half-spreads (already halved; no extra 0.5 factor).
PREREGISTERED_HALF_SPREADS_BP: dict[str, dict[str, float]] = {
    "US": {"1Y": 0.30, "2Y": 0.20, "3Y": 0.25, "5Y": 0.10, "10Y": 0.12, "20Y": 0.20, "30Y": 0.25},
    "DE": {"1Y": 0.40, "2Y": 0.25, "3Y": 0.30, "5Y": 0.15, "10Y": 0.15, "20Y": 0.25, "30Y": 0.30},
    "UK": {"1Y": 0.40, "2Y": 0.25, "3Y": 0.30, "5Y": 0.15, "10Y": 0.15, "20Y": 0.25, "30Y": 0.35},
    "JP": {"1Y": 0.50, "2Y": 0.30, "3Y": 0.35, "5Y": 0.20, "10Y": 0.20, "20Y": 0.30, "30Y": 0.40},
}

def get_half_spread_bp(col: str, spreads: dict | None = None) -> float:
    """Return the one-way yield half-spread (bp) for a COUNTRY_TENOR column."""
    parts = col.split("_", 1)
    country = parts[0] if len(parts) == 2 else "US"
    tenor   = parts[-1]
    src = spreads or PREREGISTERED_HALF_SPREADS_BP
    return src.get(country, src["US"]).get(tenor, 0.25)

# ---------------------------------------------------------------------------
# Train / Val / Test chronological split
# Cross-market panel starts ~2004-09-07 (when ECB data becomes available).
# Approximate boundaries (exact dates set dynamically in data.py):
#   Train : 2004-09 -> ~2017-05  (60 %)
#   Val   : ~2017-06 -> ~2021-10  (20 %)
#   Test  : ~2021-11 -> present   (20 %)
# ---------------------------------------------------------------------------
# Train / Test split — 80 / 20
#
# WHY 80/20 (not 60/20/20):
# The HMM regime detector must be fitted ONLY on training data, then applied
# forward to the test period. Fitting on the full dataset leaks future
# information — the model "knows" about 2022 stress when labelling 2005.
# 80/20 keeps the evaluation clean and the narrative simple:
#   "Trained on 80% of history. Tested on the 20% the model never saw."
#
# Cross-market panel: ~5,400 days from 2004-09
#   Train : 2004-09 → ~2021     (80% ≈ 4,300 days)
#   Test  : ~2021   → 2026-05   (20% ≈ 1,100 days)
TRAIN_FRAC: float = 0.80
TEST_FRAC:  float = 0.20

def get_split_dates(index) -> dict[str, tuple]:
    """
    Return chronological 80/20 train / test boundary dates for a DatetimeIndex.

    Returns dict with keys "train" and "test", each a (start, end) tuple.
    The train end date is the single boundary passed to rolling_pca() (to
    freeze PCA loadings) and to detect_regimes() (to fit the HMM on training
    data only). It is computed ONCE from the raw yield-changes index before
    any modelling begins and must not be recomputed afterwards.

    Intentionally has no "val" key. We use 80/20, not 60/20/20:
    hyperparameters are set by economic reasoning, not validation-set search,
    so a separate validation period adds no value and shrinks training data.
    """
    n       = len(index)
    n_train = int(n * TRAIN_FRAC)
    return {
        "train": (index[0],       index[n_train - 1]),
        "test":  (index[n_train], index[-1]),
    }

def get_train_end(index):
    """Return the last date of the training period (pd.Timestamp)."""
    return get_split_dates(index)["train"][1]

# ---------------------------------------------------------------------------
# Covariance estimator map (mirrors Jay's COV_METHOD_MAP)
# ---------------------------------------------------------------------------
COV_METHOD_MAP: dict[str, str] = {
    "sample": "sample",
    "ewma":   "ewma",
    "lw":     "ledoit_wolf",
}

# ---------------------------------------------------------------------------
# StrategyConfig — one point in the parameter grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyConfig:
    """
    Immutable strategy configuration. Frozen so it is hashable / usable as a
    cache key. Mirrors Jay's StrategyConfig but adapted for multi-country.

    Parameters
    ----------
    method          : 1=geometric, 2=min-variance, 3=mean-variance
    covariance      : "sample" | "ewma" | "lw"
    pca_window      : rolling PCA correlation window (days)
    refit           : PCA refit cadence (days); also z-score block-reset boundary
    z_window        : trailing window for rolling z-score normalisation
    version         : "A" (block-reset cumulative residual) | "B" (rolling-K sum)
    K               : Version B rolling sum window (required if version="B")
    cost_model      : "flat" ($/contract) | "dv01_bp" (half-spread × DV01)
    cost_per_contract_usd : one-way cost per contract (flat model only)
    no_trade_band   : tau ∈ [0,1); 0 = always rebalance (off)
    vol_target      : True = scale positions to hit VOL_TARGET_ANN
    capital_usd     : portfolio size for contract sizing
    trade_countries : which countries to trade (JP excluded by default)
    gamma           : Method-3 risk-aversion scalar; None = auto-calibrate on train
    """
    method:               int   = 3
    covariance:           str   = "lw"
    pca_window:           int   = 252
    refit:                int   = 63
    z_window:             int   = 63
    version:              str   = "A"
    K:                    int | None = None
    cost_model:           str   = "dv01_bp"
    cost_per_contract_usd: float = 0.0
    no_trade_band:        float = 0.0
    vol_target:           bool  = True
    capital_usd:          float = DEFAULT_CAPITAL_USD
    trade_countries:      tuple = TRADE_COUNTRIES
    gamma:                float | None = None    # calibrated on train if None

    def __post_init__(self):
        if self.method not in {1, 2, 3}:
            raise ValueError(f"method must be 1, 2 or 3; got {self.method}")
        if self.covariance not in COV_METHOD_MAP:
            raise ValueError(f"covariance must be sample/ewma/lw; got {self.covariance}")
        if self.version not in {"A", "B"}:
            raise ValueError(f"version must be A or B; got {self.version}")
        if self.version == "B" and self.K is None:
            raise ValueError("version='B' requires K to be set")
        if self.cost_model not in {"flat", "dv01_bp"}:
            raise ValueError(f"cost_model must be flat or dv01_bp; got {self.cost_model}")
        if not (0.0 <= self.no_trade_band < 1.0):
            raise ValueError(f"no_trade_band must be in [0, 1); got {self.no_trade_band}")

    @property
    def cov_method(self) -> str:
        return COV_METHOD_MAP[self.covariance]

    @property
    def is_default(self) -> bool:
        """True if this is the recommended out-of-the-box config."""
        return (self.method == 3 and self.covariance == "lw"
                and self.pca_window == 252 and self.refit == 63
                and self.z_window == 63 and self.version == "A"
                and self.no_trade_band == 0.0)


# ---------------------------------------------------------------------------
# DataConfig — paths
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    base_dir:   Path = Path(".")
    yield_dir:  Path = Path("data/yields")
    output_dir: Path = Path("outputs")
    cache_dir:  Path = Path("outputs/cache")
