"""
costs.py
========
Transaction cost models for the multi-country strategy.
Adapted from Jay's cost framework.

Two models:
  flat     — fixed $/contract one-way charge (simple but over-penalises short end)
  dv01_bp  — cost = half_spread_bp * DV01 * |delta_notional|  (realistic)

WHY dv01_bp IS BETTER THAN FLAT
Jay found that a flat $8.5/contract fee collapsed net IR from 1.3 to -2.11.
The culprit: short-end instruments (3M, 6M bills) have tiny DV01, so to get
the same yield exposure you need many contracts -> flat fee scales with
contracts not with risk. DV01_bp charges proportional to risk moved, which
is how traders actually think about cost (bid-ask spread in yield terms).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import (
    FUTURES_CONTRACT_FACE,
    DEFAULT_CAPITAL_USD,
    get_half_spread_bp,
    PREREGISTERED_HALF_SPREADS_BP,
    tenor_to_years,
)


def compute_cost_flat(
    delta_g: np.ndarray,
    dv01_row: np.ndarray,
    columns: list[str],
    cost_per_contract_usd: float,
    capital_usd: float = DEFAULT_CAPITAL_USD,
) -> float:
    """
    Flat $/contract cost model.

    Steps:
      1. Convert g change ($/bp) to notional change using DV01
      2. Convert notional to number of contracts using face value
      3. Multiply by cost_per_contract_usd

    Parameters
    ----------
    delta_g              : (N,) change in yield-space book ($/bp)
    dv01_row             : (N,) per-instrument DV01 ($/100 face/bp)
    columns              : list of COUNTRY_TENOR labels matching delta_g
    cost_per_contract_usd: one-way $/contract charge
    capital_usd          : portfolio size for normalisation

    Returns
    -------
    total_cost : float, in the same return units as the P&L
    """
    if cost_per_contract_usd == 0.0:
        return 0.0

    total = 0.0
    for j, col in enumerate(columns):
        dg  = abs(float(delta_g[j]))
        dv  = float(dv01_row[j])
        if dv <= 0 or dg == 0:
            continue
        # notional = g / DV01 (in $100 face units)
        dn_100face = dg / dv
        # convert to contract count
        tenor  = col.split("_", 1)[-1]
        face   = FUTURES_CONTRACT_FACE.get(tenor, 100_000.0)
        n_cont = (dn_100face * 100.0) / (face / 100.0)
        total += n_cont * cost_per_contract_usd

    # Normalise to return units (divide by capital)
    return total / capital_usd


def compute_cost_dv01bp(
    delta_g: np.ndarray,
    dv01_row: np.ndarray,
    columns: list[str],
    capital_usd: float = DEFAULT_CAPITAL_USD,
    half_spreads: dict | None = None,
    spread_mult: float = 1.0,
) -> float:
    """
    DV01-bp half-spread cost model (realistic).

    cost_i = half_spread_i(bp) * DV01_i($/100face/bp) * |delta_notional_i|

    This charges proportional to the risk being moved, not the number of
    contracts.  Short-end instruments carry the same cost per unit of risk
    as long-end instruments, which is the correct economic framing.

    Parameters
    ----------
    delta_g      : (N,) change in yield-space book
    dv01_row     : (N,) per-instrument DV01 values
    columns      : list of COUNTRY_TENOR labels
    capital_usd  : portfolio size
    half_spreads : override for PREREGISTERED_HALF_SPREADS_BP
    spread_mult  : scale spreads (e.g. 1.5 for +50% stress test)

    Returns
    -------
    total_cost : float, in return units
    """
    total = 0.0
    for j, col in enumerate(columns):
        dg  = abs(float(delta_g[j]))
        dv  = float(dv01_row[j])
        if dv <= 0 or dg == 0:
            continue
        hs   = get_half_spread_bp(col, half_spreads) * spread_mult
        # delta_notional: change in $100 face notional
        dn   = dg / dv
        # cost = half_spread(bp) * DV01($/100face/bp) * notional($100face)
        # = hs * dv * dn  [in $ per $100 face units]
        # Then / capital_usd to get return units
        total += hs * dv * dn

    return total / capital_usd


def compute_daily_costs(
    positions_g: pd.DataFrame,
    dv01_matrix: pd.DataFrame,
    cost_model: str = "dv01_bp",
    cost_per_contract_usd: float = 0.0,
    capital_usd: float = DEFAULT_CAPITAL_USD,
    spread_mult: float = 1.0,
) -> pd.Series:
    """
    Compute daily transaction costs for a portfolio over time.

    Parameters
    ----------
    positions_g  : (T x N) DataFrame of yield-space book g_t
    dv01_matrix  : (T x N) DataFrame of DV01 values
    cost_model   : "flat" or "dv01_bp"
    cost_per_contract_usd : for flat model
    capital_usd  : portfolio size
    spread_mult  : scale spreads for stress testing

    Returns
    -------
    costs : Series (date) of daily cost in return units
    """
    columns = positions_g.columns.tolist()
    dates   = positions_g.index
    result  = pd.Series(0.0, index=dates)

    prev_g = np.zeros(len(columns))

    for date in dates:
        g_t   = positions_g.loc[date].values.astype(float)
        dv_t  = dv01_matrix.reindex([date]).ffill().iloc[0].values.astype(float)                 if date in dv01_matrix.index else                 dv01_matrix.iloc[-1].values.astype(float)

        delta_g = g_t - prev_g

        if cost_model == "flat":
            c = compute_cost_flat(
                delta_g, dv_t, columns,
                cost_per_contract_usd, capital_usd,
            )
        else:  # dv01_bp
            c = compute_cost_dv01bp(
                delta_g, dv_t, columns, capital_usd, spread_mult=spread_mult,
            )

        result[date] = c
        prev_g = g_t.copy()

    return result
