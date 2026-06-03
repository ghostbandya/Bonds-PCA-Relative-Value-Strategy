"""
02_data_prep.py
===============
Data cleaning, alignment, and preparation layer.

What this module does
---------------------
1. Loads the multi-country sovereign yields (from 01_data_fetch)
2. Loads the existing bond futures price data (px_last, metadata)
3. Aligns both datasets to a common business-day calendar
4. Interpolates small gaps (≤ 5 days); drops rows that remain too sparse
5. Computes daily yield *changes* (first differences) — the input to PCA
6. Optionally computes per-country term structures for multi-market PCA
7. Saves cleaned artefacts to data/yields/ for downstream use

Outputs
-------
  data/yields/yields_clean.csv        — cleaned levels, MultiIndex cols
  data/yields/yield_changes.csv       — daily first-differences, MultiIndex cols
  data/futures_clean.csv              — aligned futures prices (front-month)
"""

import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
YIELD_DIR   = os.path.join(DATA_DIR, "yields")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  1. Load raw yield data
# ══════════════════════════════════════════════════════════════════════════════

def load_yields_raw(path: str = None) -> pd.DataFrame:
    """
    Load combined_yields.csv produced by 01_data_fetch.py.
    Returns MultiIndex-column DataFrame (country, tenor).
    """
    if path is None:
        path = os.path.join(YIELD_DIR, "combined_yields.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Combined yields not found at {path}.\n"
            "Run 01_data_fetch.py first."
        )
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    df.index.name = "date"
    df.columns.names = ["country", "tenor"]
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  2. Load existing bond futures data
# ══════════════════════════════════════════════════════════════════════════════

def load_futures(px_path: str = None, meta_path: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load futures price (px_last) and metadata CSVs.

    Returns
    -------
    prices   : DataFrame, index=date, columns=Bloomberg tickers
    metadata : DataFrame with columns [ticker, country, tenor, root, exchange, generic_n]
    """
    if px_path is None:
        px_path = os.path.join(DATA_DIR, "bond_futures_px_last.csv")
    if meta_path is None:
        meta_path = os.path.join(DATA_DIR, "bond_futures_metadata.csv")

    prices   = pd.read_csv(px_path,   index_col=0, parse_dates=True)
    metadata = pd.read_csv(meta_path)

    prices.index.name = "date"
    prices = prices.apply(pd.to_numeric, errors="coerce")
    return prices, metadata


def extract_front_month_futures(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    countries: list = None,
) -> pd.DataFrame:
    """
    Extract front-month (generic_n == 1) contract prices for each instrument.

    Returns DataFrame with MultiIndex columns (country, instrument_label).
    e.g. ('US', '10y'), ('DE', 'Bund 10y')
    """
    front = metadata[metadata["generic_n"] == 1].copy()
    if countries:
        front = front[front["country"].isin(countries)]

    result = {}
    for _, row in front.iterrows():
        ticker = row["ticker"]
        if ticker not in prices.columns:
            continue
        label    = (row["country"], row["tenor"])
        col_data = prices[ticker].dropna()
        result[label] = col_data

    df = pd.DataFrame(result)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["country", "tenor"])
    df.index.name = "date"
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  3. Clean and align yields
# ══════════════════════════════════════════════════════════════════════════════

def clean_yields(
    df: pd.DataFrame,
    start: str = "1990-01-01",
    end:   str = None,
    max_gap_fill: int = 5,
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """
    Clean raw yield levels.

    Steps
    -----
    1. Restrict to date range
    2. Reindex to business-day calendar
    3. Forward-fill gaps ≤ max_gap_fill days (e.g. bank holidays)
    4. Drop columns with coverage below min_coverage
    5. Drop rows where ALL values are NaN
    6. Apply basic sanity filters (yields must be in [-2%, 25%])

    Parameters
    ----------
    max_gap_fill  : int  — max consecutive days to ffill (default 5)
    min_coverage  : float — minimum fraction of non-NaN values to keep a column
    """
    if end is None:
        end = df.index.max().strftime("%Y-%m-%d")

    df = df.loc[start:end].copy()

    # Reindex to business days
    bdays = pd.date_range(start=df.index.min(), end=df.index.max(), freq="B")
    df    = df.reindex(bdays).ffill(limit=max_gap_fill)
    df.index.name = "date"

    # Sanity bounds
    df = df.where((df > -2) & (df < 25))

    # Drop low-coverage columns
    coverage = df.notna().mean()
    keep     = coverage[coverage >= min_coverage].index
    dropped  = coverage[coverage < min_coverage].index.tolist()
    if dropped:
        print(f"  [PREP] Dropping low-coverage columns: {dropped}")
    df = df[keep]

    # Drop fully-empty rows
    df = df.dropna(how="all")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  4. Compute yield changes
# ══════════════════════════════════════════════════════════════════════════════

def compute_yield_changes(
    df: pd.DataFrame,
    method: str = "diff",
) -> pd.DataFrame:
    """
    Compute daily yield changes (the direct input to PCA).

    Parameters
    ----------
    method : 'diff'   — simple first difference  Δy = y(t) - y(t-1)  [bps if *100]
             'pct'    — percentage change  (rarely used for yields)

    Returns
    -------
    DataFrame of same shape minus first row, in the same units as input.
    We keep units in percent (1 bps = 0.01).
    """
    if method == "diff":
        changes = df.diff().iloc[1:]
    elif method == "pct":
        changes = df.pct_change().iloc[1:]
    else:
        raise ValueError(f"method must be 'diff' or 'pct', got '{method}'")

    # Drop rows where all values in any country-block are NaN
    changes = changes.dropna(how="all")
    return changes


# ══════════════════════════════════════════════════════════════════════════════
#  5. Per-country term structure helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_country_changes(
    changes: pd.DataFrame,
    country: str,
) -> pd.DataFrame:
    """
    Slice yield changes for a single country.
    Returns a flat-column DataFrame, columns = tenor labels.
    """
    if "country" in changes.columns.names:
        sub = changes.xs(country, axis=1, level="country")
    else:
        sub = changes.filter(like=country)
    return sub.dropna(how="all")


def get_cross_market_changes(
    changes: pd.DataFrame,
    tenors: list = None,
) -> pd.DataFrame:
    """
    Build a cross-market panel for multi-country PCA.
    Columns are named  '<country>_<tenor>'  (e.g. 'US_10Y', 'DE_10Y').

    WHY A CROSS-MARKET PANEL?
    ──────────────────────────
    Running PCA on all countries jointly (rather than separately per country)
    extracts GLOBAL factors that drive yields across US, DE, UK, JP together.
    This is more powerful than per-country PCA because:
      - PC1 becomes a global rates level factor (driven by G4 central bank policy)
      - PC2 captures global curve slope (risk-on/off across all markets)
      - The residuals isolate country-specific AND tenor-specific anomalies

    Japan (JP) is included in the PCA for better factor estimation but is
    EXCLUDED from trading (see main.py) because BoJ YCC 2012–2024 prevented
    JGB yields from freely mean-reverting — the signal would be spurious.

    Rows are restricted to dates where ALL included series are non-NaN.
    This means the panel starts from the latest data source start date
    (ECB data begins ~2004, so the cross-market panel starts ~2004).

    If tenors is specified, only those tenors are included for each country.
    """
    if tenors is None:
        tenors = ["1Y", "2Y", "3Y", "5Y", "10Y", "20Y", "30Y"]

    frames = []
    for country in changes.columns.get_level_values("country").unique():
        sub = changes.xs(country, axis=1, level="country")
        available = [t for t in tenors if t in sub.columns]
        sub       = sub[available].copy()
        sub.columns = [f"{country}_{t}" for t in sub.columns]
        frames.append(sub)

    panel = pd.concat(frames, axis=1).dropna()
    return panel


# ══════════════════════════════════════════════════════════════════════════════
#  6. Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_prep(
    start: str = "2000-01-01",
    end:   str = None,
    countries: list = None,
    save: bool = True,
) -> dict:
    """
    Full preparation pipeline.

    Returns dict with keys:
      'yields_clean'    : cleaned yield levels  (MultiIndex cols)
      'yield_changes'   : daily first differences  (MultiIndex cols)
      'cross_market'    : cross-country panel for global PCA  (flat cols)
      'futures_front'   : front-month futures prices  (MultiIndex cols)
    """
    print("=" * 50)
    print("  Data Preparation Pipeline")
    print("=" * 50)

    # ── Yields ────────────────────────────────────────────────────────────────
    print("\n[1] Loading raw yields …")
    raw_yields = load_yields_raw()

    if countries:
        mask = raw_yields.columns.get_level_values("country").isin(countries)
        raw_yields = raw_yields.loc[:, mask]

    print("[2] Cleaning yields …")
    yields_clean = clean_yields(raw_yields, start=start, end=end)
    print(f"    Shape after cleaning: {yields_clean.shape}")

    print("[3] Computing yield changes …")
    yield_changes = compute_yield_changes(yields_clean)
    print(f"    Shape of changes: {yield_changes.shape}")

    print("[4] Building cross-market panel …")
    cross_market = get_cross_market_changes(yield_changes)
    print(f"    Cross-market panel: {cross_market.shape}  "
          f"({cross_market.index[0].date()} → {cross_market.index[-1].date()})")

    # ── Futures ───────────────────────────────────────────────────────────────
    print("\n[5] Loading futures prices …")
    try:
        prices, metadata = load_futures()
        futures_front = extract_front_month_futures(prices, metadata, countries=countries)
        print(f"    Front-month futures: {futures_front.shape}")
    except FileNotFoundError as e:
        print(f"    [WARN] {e}")
        futures_front = pd.DataFrame()

    # ── Save ──────────────────────────────────────────────────────────────────
    if save:
        yields_clean.to_csv(os.path.join(YIELD_DIR, "yields_clean.csv"))
        yield_changes.to_csv(os.path.join(YIELD_DIR, "yield_changes.csv"))
        cross_market.to_csv(os.path.join(YIELD_DIR, "cross_market_changes.csv"))
        if not futures_front.empty:
            futures_front.to_csv(os.path.join(DATA_DIR, "futures_clean.csv"))
        print("\n  ✓ Cleaned data saved to data/yields/ and data/")

    return {
        "yields_clean":  yields_clean,
        "yield_changes": yield_changes,
        "cross_market":  cross_market,
        "futures_front": futures_front,
    }


# ── Convenience loaders (for downstream modules) ───────────────────────────────

def load_yield_changes() -> pd.DataFrame:
    path = os.path.join(YIELD_DIR, "yield_changes.csv")
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def load_cross_market() -> pd.DataFrame:
    path = os.path.join(YIELD_DIR, "cross_market_changes.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_yields_clean() -> pd.DataFrame:
    path = os.path.join(YIELD_DIR, "yields_clean.csv")
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean and prepare yield data.")
    parser.add_argument("--start",     default="2000-01-01")
    parser.add_argument("--end",       default=None)
    parser.add_argument("--countries", nargs="+", default=None)
    args = parser.parse_args()

    artefacts = run_prep(
        start=args.start,
        end=args.end,
        countries=args.countries,
    )

    print("\nCross-market panel preview:")
    print(artefacts["cross_market"].tail(3).to_string())
    print("\n✓ Done.")
