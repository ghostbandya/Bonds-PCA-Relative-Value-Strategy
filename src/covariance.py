"""
covariance.py
=============
Covariance estimators and PCA-from-covariance for the strategy.
Ported from Jay's covariance.py and adapted for multi-country use.

Three estimators:
  sample      — standard unbiased sample covariance
  ewma        — exponentially weighted moving average (half-life weighted)
  ledoit_wolf — shrinkage estimator; most stable with limited history

All estimators accept a (T x N) matrix of yield changes and return (mean, Sigma).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from .config import N_PC


def estimate_cov(
    X_fit: np.ndarray,
    method: str = "ledoit_wolf",
    ewma_halflife: int = 63,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Return (mean_, Sigma, diagnostics) for the chosen estimator.

    Parameters
    ----------
    X_fit         : (T, N) array of yield changes over the estimation window
    method        : "sample" | "ewma" | "ledoit_wolf"
    ewma_halflife : half-life in days for EWMA decay (ignored for other methods)

    Returns
    -------
    mean_  : (N,) array — weighted mean
    Sigma  : (N, N) array — covariance estimate
    extra  : dict — diagnostics (shrinkage coeff for LW, halflife for EWMA)

    Notes
    -----
    WHY LEDOIT-WOLF?
    With N=23 instruments and a rolling window of 252 days, the sample
    covariance matrix has ~276 free parameters estimated from 252 observations.
    It is close to singular and noisy.  Ledoit-Wolf shrinks the sample matrix
    toward a structured target (scaled identity), dramatically reducing
    estimation error and producing a more stable, invertible Sigma.
    In Jay's results, LW lifted IR from 1.249 (sample) to 1.318 — meaningful
    even with only 6 instruments.  With 23 instruments the benefit is larger.
    """
    X = np.asarray(X_fit, dtype=np.float64)
    T = len(X)

    if method == "sample":
        mean_ = X.mean(axis=0)
        Sigma = np.cov(X, rowvar=False, ddof=1)
        return mean_, Sigma, {}

    if method == "ewma":
        # Exponential weights: newest row gets highest weight.
        # lam = per-day decay factor; w is then normalised to sum to 1.
        lam = np.exp(-np.log(2.0) / ewma_halflife)
        w   = lam ** np.arange(T)[::-1]   # shape (T,), index 0 = oldest
        w   = w / w.sum()
        mean_ = w @ X
        Xc    = X - mean_
        # Reliability-weight debias (analogous to Bessel's correction)
        Sigma = (Xc * w[:, None]).T @ Xc / (1.0 - np.sum(w ** 2))
        return mean_, Sigma, {"ewma_halflife": ewma_halflife}

    if method == "ledoit_wolf":
        lw = LedoitWolf(assume_centered=False).fit(X)
        mean_ = X.mean(axis=0)
        return mean_, lw.covariance_, {"shrinkage": float(lw.shrinkage_)}

    raise ValueError(f"Unknown covariance method: {method!r}. "
                     f"Choose from: sample, ewma, ledoit_wolf")


def pca_from_cov(
    X_fit: np.ndarray,
    n_pc: int = N_PC,
    method: str = "ledoit_wolf",
    ewma_halflife: int = 63,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA loadings from the covariance matrix with sign stabilisation.

    Sign stabilisation: the sign of each eigenvector is flipped if needed so
    that the largest absolute element is always positive.  This prevents the
    PC1 loading from arbitrarily flipping sign between refit windows, which
    would make the factor scores discontinuous.

    Parameters
    ----------
    X_fit  : (T, N) yield-change matrix
    n_pc   : number of PCs to return
    method : covariance estimator
    ewma_halflife : for EWMA estimator

    Returns
    -------
    V      : (n_pc, N) array — loadings (rows are PCs, unit-norm)
    lam    : (n_pc,) array — eigenvalues
    Sigma  : (N, N) array — the covariance matrix used
    """
    _, Sigma, _ = estimate_cov(X_fit, method=method, ewma_halflife=ewma_halflife)
    N = Sigma.shape[0]

    # eigh: symmetric / Hermitian eigendecomposition (faster + stable)
    # Returns eigenvalues in ascending order -> reverse
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]   # (N, N); columns = eigenvectors

    V   = eigenvectors[:, :n_pc].T   # (n_pc, N); rows = PCs
    lam = eigenvalues[:n_pc]

    # Sign stabilisation: flip each row of V so max-abs element is positive
    for j in range(n_pc):
        if V[j, np.argmax(np.abs(V[j]))] < 0:
            V[j] *= -1

    return V, lam, Sigma


def build_rolling_cov(
    changes: pd.DataFrame,
    window: int = 252,
    method: str = "ledoit_wolf",
    ewma_halflife: int = 63,
) -> dict[pd.Timestamp, np.ndarray]:
    """
    Build a dict {date -> Sigma_t} using a rolling estimation window.
    Used by run_backtest_v2 to supply a time-varying covariance at each date.

    Parameters
    ----------
    changes   : (T x N) DataFrame of yield changes
    window    : estimation window in days
    method    : covariance estimator
    ewma_halflife : for EWMA

    Returns
    -------
    dict mapping each date (once window is warm) to its (N, N) covariance matrix
    """
    result = {}
    arr    = changes.values.astype(np.float64)
    dates  = changes.index
    N      = arr.shape[1]

    for i in range(window, len(dates)):
        X_win = arr[i - window: i]
        _, Sigma, _ = estimate_cov(X_win, method=method, ewma_halflife=ewma_halflife)
        result[dates[i]] = Sigma

    return result
