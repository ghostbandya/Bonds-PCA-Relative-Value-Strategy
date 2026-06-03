"""
rebalance.py
============
Trade-to-edge no-trade band on the scaled portfolio.
Adapted from Jay's rebalance.py for the multi-country universe.

WHY A NO-TRADE BAND?
Methods 2 and 3 recalculate the optimal portfolio every day.  Even tiny
covariance updates cause daily rebalancing -> enormous turnover -> costs
destroy net performance.  Jay found that at $8.5/contract, net IR collapses
from 1.3 to -2.11 without the band.

The band (width τ) says: only trade if the current portfolio has drifted far
enough from the target that it's worth paying costs to fix it.  Drift is
measured in Mahalanobis (risk-weighted) distance:

    r_t = ||G_t - H_{t-1}||_Sigma / ||G_t||_Sigma

  r_t > τ  ->  trade to the edge:  H_t = H_{t-1} + (1 - τ/r_t)(G_t - H_{t-1})
  r_t ≤ τ  ->  hold:               H_t = H_{t-1}

Key property: "trade to the edge" means after trading, the executed book H_t
is exactly τ away from the target G_t in Mahalanobis space.  We never
over-trade.

In Jay's sweep, τ=0.75 reduced cost drag by 83% while losing only 16% of
gross IR -> net IR went from -0.184 to +0.403.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import NEUTRAL_TOL
from .weights import projection_matrix


def _sigma_norm(x: np.ndarray, Sigma: np.ndarray) -> float:
    """Mahalanobis norm sqrt(x^T Sigma x), clipped at 0."""
    quad = float(x @ Sigma @ x)
    return float(np.sqrt(max(quad, 0.0)))


def _reproject_neutral(
    H: np.ndarray,
    V: np.ndarray,
    tol: float = NEUTRAL_TOL,
) -> tuple[np.ndarray, bool]:
    """
    If H has drifted away from factor-neutrality under new loadings V,
    re-project onto the neutral subspace M = I - V^T V.

    Returns (H_reprojected, was_reprojected).
    """
    V = np.atleast_2d(V)
    residual = float(np.max(np.abs(V @ H)))
    if residual <= tol:
        return H, False
    M = projection_matrix(V)
    return M @ H, True


def apply_no_trade_band(
    target_G: pd.DataFrame,
    Sigma_dict: dict,
    V_dict: dict,
    tau: float,
    neutral_tol: float = NEUTRAL_TOL,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply trade-to-edge no-trade band to the target book G.

    Parameters
    ----------
    target_G   : (T x N) DataFrame — desired scaled book G_t = k_t * g_t
    Sigma_dict : dict {date -> (N,N) Sigma} — rolling covariance matrices
    V_dict     : dict {date -> (k,N) V}     — rolling PCA loadings
    tau        : band width ∈ [0, 1); 0 = always rebalance (band off)
    neutral_tol: factor-neutrality tolerance for re-projection check

    Returns
    -------
    exec_G    : (T x N) DataFrame — executed book H_t (band-filtered)
    hold_mask : Series (bool)     — True on genuine hold days (no trade)
    """
    cols      = list(target_G.columns)
    N         = len(cols)
    hold_mask = pd.Series(False, index=target_G.index)

    if tau <= 0.0:
        return target_G.copy(), hold_mask

    exec_G = pd.DataFrame(0.0, index=target_G.index, columns=cols)
    H_prev = np.zeros(N)

    for t in target_G.index:
        G_t = target_G.loc[t].to_numpy(dtype=float)

        # Fetch current covariance and loadings (fall back to identity/empty)
        Sigma = Sigma_dict.get(t, np.eye(N))
        V     = V_dict.get(t, np.zeros((1, N)))  # dummy if not available

        # Re-project if new loadings made the held book non-neutral
        H_prev, _ = _reproject_neutral(H_prev, V, neutral_tol)

        # Mahalanobis drift ratio
        diff = G_t - H_prev
        num  = _sigma_norm(diff, Sigma)
        den  = _sigma_norm(G_t, Sigma)

        if den > 0.0 and (num / den) > tau:
            # Breach: trade to the edge
            # H_t = H_prev + (1 - tau/r) * (G_t - H_prev)
            scale = 1.0 - tau / (num / den)
            H_new = H_prev + scale * diff
            hold  = False
        else:
            # Hold: carry the current book
            H_new = H_prev.copy()
            hold  = True

        exec_G.loc[t] = H_new
        hold_mask[t]  = hold
        H_prev        = H_new.copy()

    return exec_G, hold_mask
