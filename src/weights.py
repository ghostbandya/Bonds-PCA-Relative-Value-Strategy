"""
weights.py
==========
Factor-neutral portfolio construction — Methods 1, 2, 3.
Adapted from Jay's weights.py for the multi-country universe.

Factor neutrality convention
----------------------------
V is the (k x N) loading matrix (k PCs, N instruments) with orthonormal rows.
M = I - V^T V  projects onto the orthogonal complement of the PC span.
Any vector in the column space of M satisfies V*g = 0 exactly.

Method 1 — Geometric (single-tenor)
    g = -s_m * M[:,m]  then L1-normalised
    Trade only the most dislocated tenor; hedge factor exposure geometrically.
    Simplest: one degree of freedom, parsimony.

Method 2 — Min-Variance (KKT)
    min  1/2 g^T Sigma g
    s.t. V g = 0,  e_m^T g = -s_m
    Pin the most dislocated tenor; minimise variance of the hedging tail.

Method 3 — Mean-Variance (soft, all tenors)
    max  alpha^T g - (gamma/2) g^T Sigma g
    s.t. V g = 0
    alpha_i = -s_i * sigma_i  (signal weighted by instrument vol)
    Incorporate the full signal vector into a portfolio; gamma controls risk.

Key results from Jay's US study:
  Method 1: IR 0.505  (baseline proof of concept)
  Method 2: IR 0.871  (min-var hedge adds ~0.4 IR)
  Method 3: IR 1.318  (MV soft tilt, Ledoit-Wolf cov — best gross)
"""
from __future__ import annotations
import numpy as np
import scipy.linalg
from .config import NEUTRAL_TOL, RHO_MIN, RIDGE_EPS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def projection_matrix(V: np.ndarray) -> np.ndarray:
    """M = I - V^T V  for loadings V (k x N, orthonormal rows)."""
    V = np.atleast_2d(np.asarray(V, dtype=float))
    N = V.shape[1]
    return np.eye(N) - V.T @ V


def assert_factor_neutral(
    V: np.ndarray,
    g: np.ndarray,
    tol: float = NEUTRAL_TOL,
    label: str = "",
) -> None:
    """Raise AssertionError unless ||V g||_inf < tol."""
    V = np.atleast_2d(np.asarray(V, dtype=float))
    residual = float(np.max(np.abs(V @ np.asarray(g, dtype=float)))) if V.size else 0.0
    if residual > tol:
        raise AssertionError(
            f"Not factor-neutral{f' [{label}]' if label else ''}: "
            f"||Vg||_inf = {residual:.3e} > {tol:.1e}"
        )


def _spd_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve A X = B for symmetric PD A; ridge fallback for ill-conditioning."""
    try:
        return scipy.linalg.solve(A, B, assume_a="pos")
    except (scipy.linalg.LinAlgError, np.linalg.LinAlgError):
        A_ridge = A + RIDGE_EPS * np.eye(A.shape[0])
        return scipy.linalg.solve(A_ridge, B, assume_a="sym")


# ---------------------------------------------------------------------------
# Method 1 — Geometric projection
# ---------------------------------------------------------------------------

def method1_geometric(
    s_row: np.ndarray,
    V: np.ndarray,
    m: int | None = None,
) -> np.ndarray:
    """
    Single-tenor geometric projection, L1-normalised.

    g = -s_m * M[:,m]  then  g /= ||g||_1

    Under L1 normalisation the direction depends only on sign(s_m); the
    magnitude cancels.  This is the 'parsimony' baseline: one degree of
    freedom, exact factor neutrality by construction.

    Parameters
    ----------
    s_row : (N,) z-score vector for this date
    V     : (k, N) PCA loadings
    m     : index of tenor to trade; defaults to argmax|s|

    Returns
    -------
    g : (N,) yield-space book (zero vector if tenor is degenerate)
    """
    s = np.asarray(s_row, dtype=float)
    M = projection_matrix(V)

    if m is None:
        m = int(np.nanargmax(np.abs(s)))

    # Residual room: rho_m = e_m^T M e_m = M[m,m]
    # If near zero, this tenor is well-explained by the 3 PCs -> skip.
    if M[m, m] < RHO_MIN:
        return np.zeros_like(s)

    g  = -s[m] * M[:, m]
    l1 = float(np.abs(g).sum())
    return g / l1 if l1 > 0 else np.zeros_like(s)


# ---------------------------------------------------------------------------
# Method 2 — Min-Variance (KKT)
# ---------------------------------------------------------------------------

def method2_minvar(
    s_row: np.ndarray,
    V: np.ndarray,
    Sigma: np.ndarray,
    m: int | None = None,
) -> np.ndarray:
    """
    Minimum-variance factor-neutral book, pinning the most dislocated tenor.

    Solves:
        min  1/2 g^T Sigma g
        s.t. V g = 0
             e_m^T g = -s_m   (pin the target tenor's yield exposure)

    The KKT system is:
        [Sigma  V^T  e_m]   [g ]   [0  ]
        [V      0    0  ] * [nu] = [0  ]
        [e_m^T  0    0  ]   [mu]   [-s_m]

    Parameters
    ----------
    s_row : (N,) z-score row
    V     : (k, N) PCA loadings
    Sigma : (N, N) covariance matrix
    m     : tenor index to pin (defaults to argmax|s|)

    Returns
    -------
    g : (N,) factor-neutral yield-space book
    """
    s     = np.asarray(s_row, dtype=float)
    V     = np.atleast_2d(np.asarray(V, dtype=float))
    Sigma = np.asarray(Sigma, dtype=float)
    k, N  = V.shape

    if m is None:
        m = int(np.nanargmax(np.abs(s)))

    s_m = float(s[m])
    if s_m == 0 or np.isnan(s_m):
        return np.zeros(N)

    # Build constraint matrix C = [V; e_m^T]  shape (k+1, N)
    e_m = np.zeros(N); e_m[m] = 1.0
    C   = np.vstack([V, e_m.reshape(1, -1)])   # (k+1, N)
    n_c = C.shape[0]

    # KKT system: [Sigma  C^T] [g ] = [0      ]
    #             [C      0  ] [nu]   [rhs    ]
    # rhs = [0,...,0, -s_m]  (k zeros for neutrality, then pin)
    KKT = np.block([[Sigma, C.T], [C, np.zeros((n_c, n_c))]])
    rhs = np.zeros(N + n_c)
    rhs[-1] = -s_m

    try:
        sol = _spd_solve(KKT, rhs)
    except Exception:
        return np.zeros(N)

    g = sol[:N]
    assert_factor_neutral(V, g, label="M2")
    return g


# ---------------------------------------------------------------------------
# Method 3 — Mean-Variance (soft, all tenors)
# ---------------------------------------------------------------------------

def method3_meanvar(
    s_row: np.ndarray,
    V: np.ndarray,
    Sigma: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Mean-variance portfolio incorporating all tenor signals simultaneously.

    Solves:
        max  alpha^T g - (gamma/2) g^T Sigma g
        s.t. V g = 0

    where  alpha_i = -s_i * sigma_i  (signal * per-instrument vol).

    The factor-neutral MV solution is (derived via KKT):
        g* = (1/gamma) * M_Sigma * alpha

    where M_Sigma is the Sigma-weighted projection onto the factor-neutral
    subspace.

    This is the highest-performing method in Jay's study (IR 1.318 gross).
    The gain over M2 comes from:
      (a) using all signals simultaneously rather than just the top one, and
      (b) weighting each signal by its risk-adjusted attractiveness.

    Parameters
    ----------
    s_row : (N,) z-score row
    V     : (k, N) PCA loadings
    Sigma : (N, N) covariance matrix
    gamma : risk-aversion scalar (calibrate_gamma() on train split)

    Returns
    -------
    g : (N,) factor-neutral mean-variance book
    """
    s     = np.asarray(s_row, dtype=float)
    V     = np.atleast_2d(np.asarray(V, dtype=float))
    Sigma = np.asarray(Sigma, dtype=float)
    k, N  = V.shape

    # Handle NaN signals: replace with 0 (no view on that instrument)
    s = np.where(np.isnan(s), 0.0, s)

    # Per-instrument vol (sqrt of diagonal of Sigma)
    sig_i  = np.sqrt(np.maximum(np.diag(Sigma), 0.0))
    alpha  = -s * sig_i   # signal * vol -> alpha vector

    if np.all(alpha == 0):
        return np.zeros(N)

    # KKT for constrained MV:
    # [Sigma  V^T] [g ] = [alpha/gamma]
    # [V      0  ] [nu]   [0          ]
    KKT = np.block([[Sigma, V.T], [V, np.zeros((k, k))]])
    rhs = np.zeros(N + k)
    rhs[:N] = alpha / gamma

    try:
        sol = _spd_solve(KKT, rhs)
    except Exception:
        return np.zeros(N)

    g = sol[:N]
    assert_factor_neutral(V, g, label="M3")
    return g


# ---------------------------------------------------------------------------
# Vol-targeting
# ---------------------------------------------------------------------------

def vol_scale(
    g_series: pd.Series | np.ndarray,
    vol_target_ann: float = 0.10,
    vol_window: int = 63,
    trading_days: int = 252,
    l_max: float = 10.0,
    sigma_floor: float | None = None,
) -> np.ndarray:
    """
    Compute the daily leverage scalar k_t for vol-targeting.

    k_t = vol_target_daily / max(realised_vol_daily, sigma_floor)
    k_t is clipped to [0, l_max].

    Parameters
    ----------
    g_series       : 1-D daily portfolio returns (used to estimate realised vol)
    vol_target_ann : annualised vol target (default 10%)
    vol_window     : trailing window for realised vol estimate
    trading_days   : trading days per year (252)
    l_max          : max leverage
    sigma_floor    : minimum vol (default: 0.1 * daily_target)

    Returns
    -------
    k : array of same length as g_series
    """
    import pandas as pd
    vol_target_daily = vol_target_ann / np.sqrt(trading_days)
    if sigma_floor is None:
        sigma_floor = 0.1 * vol_target_daily

    if isinstance(g_series, np.ndarray):
        g_series = pd.Series(g_series)

    realised_vol = g_series.rolling(vol_window, min_periods=max(1, vol_window // 2)).std()
    realised_vol = realised_vol.fillna(method="bfill").fillna(vol_target_daily)
    realised_vol = realised_vol.clip(lower=sigma_floor)

    k = (vol_target_daily / realised_vol).clip(upper=l_max)
    return k.values


# ---------------------------------------------------------------------------
# Gamma calibration
# ---------------------------------------------------------------------------

def calibrate_gamma(
    changes_train: "pd.DataFrame",
    V_mean: np.ndarray,
    s_train: "pd.DataFrame",
    Sigma_train: np.ndarray,
    vol_target_ann: float = 0.10,
    trading_days: int = 252,
) -> float:
    """
    Calibrate gamma so that the Method-3 unconstrained book's realised
    annualised vol on the TRAIN split ≈ vol_target_ann.

    Uses a simple bisection: compute the portfolio return series for a
    grid of gamma values and find the one where ann_vol ≈ target.

    Parameters
    ----------
    changes_train : (T, N) DataFrame of yield changes on the training set
    V_mean        : (k, N) mean PCA loadings over the training period
    s_train       : (T, N) DataFrame of z-scores on the training set
    Sigma_train   : (N, N) mean covariance matrix over the training period
    vol_target_ann : target annualised vol (default 10%)
    trading_days  : days per year

    Returns
    -------
    gamma : float — calibrated risk-aversion scalar
    """
    import pandas as pd

    def _portfolio_vol(gamma_try):
        pnl_list = []
        for date in s_train.index:
            if date not in changes_train.index:
                continue
            s_row = s_train.loc[date].values
            g = method3_meanvar(s_row, V_mean, Sigma_train, gamma=gamma_try)
            dy = changes_train.loc[date].values
            pnl = float(np.dot(g, dy))
            pnl_list.append(pnl)
        if len(pnl_list) < 10:
            return np.nan
        return np.std(pnl_list) * np.sqrt(trading_days)

    # Bisection over log-space gamma
    lo, hi = 1.0, 1e6
    for _ in range(50):
        mid = np.sqrt(lo * hi)
        v   = _portfolio_vol(mid)
        if np.isnan(v) or v <= 0:
            break
        if v > vol_target_ann:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


# deferred import to avoid circular at module level
import pandas as pd  # noqa: E402
