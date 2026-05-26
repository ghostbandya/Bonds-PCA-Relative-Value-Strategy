"""
08_carry_signal.py
==================
Carry + Roll-Down strategy for sovereign bonds.

═══════════════════════════════════════════════════════════════
STRATEGY THEORY
═══════════════════════════════════════════════════════════════
This strategy harvests TWO structural bond return premia:

1. CARRY — the excess yield over the short rate
   ───────────────────────────────────────────
   Carry_i = yield_T  −  short_rate
   (short_rate = 2Y yield as proxy for financing cost)

   A bond with positive carry earns more than its financing cost
   if yields don't change.  Carry is the "income" component of
   a funded bond position.  On average, the yield curve slopes
   upward (long rates > short rates) so most bonds carry positively.

2. ROLL-DOWN — the yield pickup from curve roll
   ─────────────────────────────────────────────
   As time passes, a bond's remaining maturity shrinks.
   A 10Y bond held for 1 year becomes a 9Y bond.
   If the yield curve is upward-sloping, 9Y yields < 10Y yields,
   so the bond's yield *falls* → its price *rises*.
   This capital gain from "rolling down the curve" is roll-down.

   Roll-Down_i = yield_T  −  yield_{T − 1Y}
   (= the yield drop expected over the next year from rolling)

   We interpolate yield_{T-1Y} from the available curve data.

3. COMBINED SCORE (duration-normalised)
   ──────────────────────────────────────
   score_i = (Carry_i + Roll_Down_i) / DV01_i

   Dividing by DV01 normalises for duration so a 1Y bond and a 10Y
   bond with the same raw carry+rolldown get the same signal magnitude.
   Without this, long-duration bonds would always dominate the signal
   simply because their carry is measured in more "price-sensitive" units.

4. WHY RAW YIELD CHANGES FOR P&L (not factor-neutral residuals)?
   ──────────────────────────────────────────────────────────────
   The PCA strategy uses factor-neutral residual changes as its P&L
   driver to isolate idiosyncratic spread reversion.  Carry is different:
   carry IS the systematic level signal.  If we subtracted out the PC1
   (level) move, we would remove exactly the yield move that carry is
   trying to capture.  Carry P&L must be computed on raw yield changes.

═══════════════════════════════════════════════════════════════
SIGNAL RULES
═══════════════════════════════════════════════════════════════
  Long  when score >  +entry_thr  (default 0.10)
  Short when score <  −entry_thr
  Close long  when score <  +exit_thr   (default 0.02)
  Close short when score >  −exit_thr

  The thresholds are in carry+roll units (% per year per DV01 unit).
  entry_thr = 0.10 means: only go long if the expected annual return
  from carry+roll exceeds 0.10% per unit of DV01 — a modest filter
  to avoid trading noise.

REGIME FILTER: GOOD → full size | NEUTRAL → 50% | BAD → flat
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

            # ── Carry = instrument yield − short rate ────────────────────
            # Short rate proxy: the 2Y yield for each country.
            # This approximates the cost of funding a bond position
            # via repo for 1 year.  In practice repo ≈ OIS ≈ 2Y yield.
            carry = flat[col] - short_rate

            # ── Roll-down: yield drop from aging by 1 year ───────────────
            # A T-year bond held for 1 year becomes a (T-1)-year bond.
            # If the yield curve slopes up, its yield falls → price rises.
            # roll_down = yield(T) − yield(T − 1Y)
            # We linearly interpolate yield(T−1Y) from the available tenors.
            rolled_t = t_years - roll_horizon_years     # target maturity after 1Y roll
            if rolled_t <= 0:
                # Short-end bonds (1Y) roll off entirely → no roll benefit
                roll_down = pd.Series(0.0, index=dates)
            else:
                # For each date, interpolate the country yield curve at rolled_t.
                # ym[i] is the vector of all country yields on date i.
                # tenor_yrs is the sorted array of available maturities.
                y_now = flat[col].values
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
                roll_down = flat[col] - y_rolled     # positive when curve is upward-sloping

            # ── Combined score, duration-normalised ───────────────────────
            # Dividing by DV01 converts from "yield units" to "return units":
            # a 0.10% carry on a 10Y bond (DV01=8.60) scores 0.012,
            # same as a 0.012% carry on a 1Y bond (DV01=0.98) scores 0.012.
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
