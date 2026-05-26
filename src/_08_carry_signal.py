"""
08_carry_signal.py
==================
Carry + Roll-Down strategy for sovereign bonds.

Theory
------
For a bond with tenor T:
  Carry     = yield_T  -  short_rate
  Roll-Down = yield_T  -  yield_{T - horizon}    (where horizon = 1 year)

  carry_roll_i = (carry_i + roll_down_i) / DV01_i

This is duration-normalised to make signals comparable across tenors.

Signal rules
------------
  Long  when  carry_roll_score  >  +entry_thr
  Short when  carry_roll_score  <  -entry_thr
  Close long  when  score  <  +exit_thr
  Close short when  score  >  -exit_thr

Regime filter: GOOD -> full size | NEUTRAL -> 50% | BAD -> flat
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

TENOR_DV01 = {
    "0.5Y": 0.50, "1Y":  0.98, "2Y":  1.92, "3Y":  2.83,
    "5Y":   4.55, "7Y":  6.20, "10Y": 8.60, "15Y": 12.0,
    "20Y": 15.5,  "25Y": 18.0, "30Y": 19.5, "40Y": 22.0,
}
TENOR_YEARS = {t: float(t.replace("Y","")) for t in TENOR_DV01}

SHORT_RATE_TENOR = {"US": "2Y", "DE": "2Y", "UK": "2Y", "JP": "2Y"}


def _tenor_years(tenor: str) -> float:
    return TENOR_YEARS.get(tenor, float(tenor.replace("Y","")))


def _get_dv01(tenor: str) -> float:
    return TENOR_DV01.get(tenor.upper(), 1.0)


def _parse_col(col: str):
    parts = col.split("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (col, "")


# ======================================================================
#  Core: vectorised carry + roll-down
# ======================================================================

def compute_carry_roll(
    yields_clean,
    roll_horizon_years: float = 1.0,
    smooth_window: int = 21,
) -> pd.DataFrame:
    """
    Vectorised computation of carry + roll-down for all instruments.

    Parameters
    ----------
    yields_clean       : DataFrame, MultiIndex columns (country, tenor), levels in %.
    roll_horizon_years : holding horizon for roll-down (default 1Y).
    smooth_window      : rolling mean window to smooth the raw signal.

    Returns
    -------
    carry_roll : DataFrame (date x 'COUNTRY_TENOR') in %/year per unit DV01.
    """
    # Flatten MultiIndex -> COUNTRY_TENOR
    if isinstance(yields_clean.columns, pd.MultiIndex):
        flat = yields_clean.copy()
        flat.columns = [f"{c}_{t}" for c, t in yields_clean.columns]
    else:
        flat = yields_clean.copy()

    # Group by country
    countries = {}
    for col in flat.columns:
        c, t = _parse_col(col)
        countries.setdefault(c, []).append((t, col))

    result = {}

    for country, tenor_col_list in countries.items():
        # Sort by maturity
        tenor_col_list = sorted(tenor_col_list, key=lambda x: _tenor_years(x[0]))
        tenor_yrs = np.array([_tenor_years(t) for t, _ in tenor_col_list])
        cols_arr  = [col for _, col in tenor_col_list]

        # Short rate column
        short_t   = SHORT_RATE_TENOR.get(country, "2Y")
        short_col = f"{country}_{short_t}"
        if short_col not in flat.columns:
            short_col = cols_arr[0]   # fallback to shortest available

        short_rate = flat[short_col]

        # Country yield matrix: (dates x tenors)
        ym = flat[cols_arr].values          # shape (T, K)
        dates = flat.index

        for t_label, col in tenor_col_list:
            t_years = _tenor_years(t_label)
            dv01    = _get_dv01(t_label)

            # Carry = instrument yield - short rate
            carry = flat[col] - short_rate

            # Roll-down (vectorised over dates)
            rolled_t = t_years - roll_horizon_years
            if rolled_t <= 0:
                roll_down = pd.Series(0.0, index=dates)
            else:
                # Interpolate along tenor axis for each date row
                y_now = flat[col].values          # (T,)
                # For each date, interpolate ym[date, :] at rolled_t
                roll_vals = np.empty(len(dates))
                roll_vals[:] = np.nan
                for i in range(len(dates)):
                    row = ym[i]
                    mask = ~np.isnan(row)
                    if mask.sum() >= 2:
                        try:
                            roll_vals[i] = np.interp(rolled_t, tenor_yrs[mask], row[mask])
                        except Exception:
                            pass
                y_rolled  = pd.Series(roll_vals, index=dates)
                roll_down = flat[col] - y_rolled

            raw = (carry + roll_down) / dv01
            result[col] = raw

    carry_roll = pd.DataFrame(result, index=flat.index)
    if smooth_window > 1:
        carry_roll = carry_roll.rolling(smooth_window, min_periods=1).mean()
    return carry_roll


# ======================================================================
#  Signal / position generation
# ======================================================================

def generate_carry_signals(
    carry_roll,
    regime,
    entry_thr: float = 0.10,
    exit_thr:  float = 0.02,
    size_neutral: float = 0.50,
):
    common  = carry_roll.index.intersection(regime.index)
    scores  = carry_roll.loc[common]
    reg     = regime.loc[common]
    instrs  = scores.columns.tolist()
    n       = len(common)

    pos_matrix = np.zeros((n, len(instrs)))
    sig_matrix = np.zeros((n, len(instrs)))

    for j, instr in enumerate(instrs):
        s   = scores[instr].values
        pos = 0.0
        for i in range(n):
            r = reg.iloc[i]
            if r == 2:
                pos = 0.0; sig_matrix[i,j] = 0; pos_matrix[i,j] = 0; continue
            si = s[i]
            if np.isnan(si):
                sig_matrix[i,j] = np.nan; pos_matrix[i,j] = pos; continue
            size = 1.0 if r == 0 else size_neutral
            if pos > 0 and si < exit_thr:
                pos = 0.0; sig_matrix[i,j] = 0
            elif pos < 0 and si > -exit_thr:
                pos = 0.0; sig_matrix[i,j] = 0
            if pos == 0:
                if si > entry_thr:
                    pos = size; sig_matrix[i,j] = 1
                elif si < -entry_thr:
                    pos = -size; sig_matrix[i,j] = -1
            pos_matrix[i,j] = pos

    positions = pd.DataFrame(pos_matrix, index=common, columns=instrs)
    signals   = pd.DataFrame(sig_matrix, index=common, columns=instrs)
    return positions, signals


# ======================================================================
#  Main pipeline
# ======================================================================

def run_carry(
    yields_clean,
    regime_dict,
    roll_horizon:  float = 1.0,
    smooth_window: int   = 21,
    entry_thr:     float = 0.10,
    exit_thr:      float = 0.02,
    save:          bool  = True,
) -> dict:
    print("=" * 50)
    print("  Carry + Roll-Down Pipeline")
    print("=" * 50)

    print("\n[1] Computing carry + roll-down scores ...")
    carry_roll = compute_carry_roll(
        yields_clean,
        roll_horizon_years=roll_horizon,
        smooth_window=smooth_window,
    )
    print(f"    Shape: {carry_roll.shape}")
    stacked = carry_roll.stack().dropna()
    print(f"    Score range: [{stacked.min():.3f}, {stacked.max():.3f}]  "
          f"mean={stacked.mean():.3f}")

    print("[2] Generating carry positions ...")
    positions, signals = generate_carry_signals(
        carry_roll, regime_dict["regime"],
        entry_thr=entry_thr, exit_thr=exit_thr,
    )

    active_pct = (positions != 0).mean().mean() * 100
    long_pct   = (positions > 0).mean().mean() * 100
    short_pct  = (positions < 0).mean().mean() * 100
    print(f"    Active: {active_pct:.1f}%  Long: {long_pct:.1f}%  Short: {short_pct:.1f}%")

    if save:
        out = os.path.join(OUTPUT_DIR, "carry")
        os.makedirs(out, exist_ok=True)
        carry_roll.to_csv(os.path.join(out, "carry_roll.csv"))
        positions.to_csv(os.path.join(out, "carry_positions.csv"))
        signals.to_csv(os.path.join(out, "carry_signals.csv"))
        print(f"    Saved to {out}/")

    return {"carry_roll": carry_roll, "positions": positions, "signals": signals}
