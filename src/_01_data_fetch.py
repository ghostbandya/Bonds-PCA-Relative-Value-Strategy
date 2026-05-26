"""
01_data_fetch.py
================
Multi-country sovereign yield data fetcher.

Sources
-------
  US  : FRED (Federal Reserve Economic Data)        -- daily, free, no auth
  DE  : ECB Data Portal (SDMX-REST API)             -- daily, free, no auth
  UK  : Bank of England GLC Nominal ZIP download    -- daily, free, no auth
  JP  : Japan Ministry of Finance daily JGB CSV     -- daily, free, no auth

All yields in percent (e.g. 4.25 = 4.25%).
Outputs: data/yields/<country>_yields.csv
         data/yields/combined_yields.csv  (MultiIndex columns)

Usage
-----
    python src/_01_data_fetch.py
    python src/_01_data_fetch.py --start 2005-01-01
    python src/_01_data_fetch.py --countries UK JP
"""

import os, re, io, zipfile, argparse, warnings
from io import StringIO
from datetime import datetime

import pandas as pd
import requests

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YIELD_DIR = os.path.join(BASE_DIR, "data", "yields")
os.makedirs(YIELD_DIR, exist_ok=True)

DEFAULT_START = "2000-01-01"
DEFAULT_END   = datetime.today().strftime("%Y-%m-%d")
COMMON_TENORS = ["1Y", "2Y", "3Y", "5Y", "10Y", "20Y", "30Y"]
TENORS = COMMON_TENORS   # backwards-compat alias

_BROWSER_HDR = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
}


# ─────────────────────────── US (FRED) ────────────────────────────────────────

US_FRED_SERIES = {
    "1Y": "DGS1", "2Y": "DGS2", "3Y": "DGS3", "5Y": "DGS5",
    "10Y": "DGS10", "20Y": "DGS20", "30Y": "DGS30",
}

def fetch_us_yields(start=DEFAULT_START, end=DEFAULT_END):
    if end is None: end = DEFAULT_END
    try:
        import pandas_datareader.data as web
    except ImportError:
        print("  [ERROR] pandas_datareader not installed"); return pd.DataFrame()
    print("  [US] Fetching from FRED ...")
    frames = {}
    for tenor, sid in US_FRED_SERIES.items():
        try:
            frames[tenor] = web.DataReader(sid, "fred", start, end)[sid]
        except Exception as e:
            print(f"    [WARN] {sid}: {e}")
    if not frames: return pd.DataFrame()
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index); df.index.name = "date"
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    df = df.reindex(columns=COMMON_TENORS)
    _summary("US", df); return df


# ─────────────────────────── Germany (ECB) ────────────────────────────────────

ECB_TENORS = {"1Y":1,"2Y":2,"3Y":3,"5Y":5,"10Y":10,"20Y":20,"30Y":30}
ECB_BASE = ("https://data-api.ecb.europa.eu/service/data/"
            "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{n}Y"
            "?startPeriod={start}&endPeriod={end}&format=csvdata")

def _fetch_ecb_tenor(args):
    """Fetch a single ECB tenor series; used with ThreadPoolExecutor."""
    lbl, n, start, end = args
    try:
        url = ECB_BASE.format(n=n, start=start, end=end)
        r = requests.get(url, timeout=30); r.raise_for_status()
        raw = pd.read_csv(StringIO(r.text))
        raw["TIME_PERIOD"] = pd.to_datetime(raw["TIME_PERIOD"])
        return lbl, raw.set_index("TIME_PERIOD")["OBS_VALUE"].rename(lbl)
    except Exception as e:
        print(f"    [WARN] ECB {lbl}: {e}")
        return lbl, None


def fetch_de_yields(start=DEFAULT_START, end=DEFAULT_END):
    if end is None: end = DEFAULT_END
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print("  [DE] Fetching from ECB SDMX-REST (parallel) ...")
    frames = {}
    tasks = [(lbl, n, start, end) for lbl, n in ECB_TENORS.items()]
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        for lbl, s in pool.map(_fetch_ecb_tenor, tasks):
            if s is not None:
                frames[lbl] = s
    if not frames: return pd.DataFrame()
    df = pd.DataFrame(frames).sort_index()
    df.index.name = "date"
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    df = df.reindex(columns=COMMON_TENORS)
    _summary("DE", df); return df


# ─────────────────────────── UK (BoE ZIP) ─────────────────────────────────────
#
# BoE publishes GLC Nominal daily data as a ZIP of Excel files, each covering
# a date range (e.g. "GLC Nominal daily data_2005 to 2015.xlsx").
# Sheet "4. spot curve": row 3 = maturities in years (0.5, 1.0, 1.5, ..., 40.0)
#                        col 0 = date, data rows start at row 5.

BOE_ZIP_URL = ("https://www.bankofengland.co.uk/-/media/boe/files/statistics/"
               "yield-curves/glcnominalddata.zip")
BOE_SPOT_SHEET = "4. spot curve"

# Tenor targets: maturity (years) -> our label
BOE_TENOR_MAP = {
    0.5: "0.5Y", 1.0: "1Y", 1.5: "1.5Y",
    2.0: "2Y",  2.5: "2.5Y", 3.0: "3Y",
    4.0: "4Y",  5.0: "5Y",   6.0: "6Y",
    7.0: "7Y",  8.0: "8Y",   9.0: "9Y",
    10.0: "10Y", 15.0: "15Y", 20.0: "20Y",
    25.0: "25Y", 30.0: "30Y", 40.0: "40Y",
}
BOE_TARGET = ["1Y", "2Y", "3Y", "5Y", "10Y", "20Y", "30Y"]


def fetch_uk_yields(start=DEFAULT_START, end=DEFAULT_END):
    """
    Fetch UK gilt nominal spot curve from BoE ZIP download.
    Falls back to FRED OECD 10Y if download fails.
    """
    if end is None: end = DEFAULT_END
    print("  [UK] Fetching from Bank of England (GLC ZIP) ...")
    df = _uk_boe_zip(start, end)
    if not df.empty: return df
    print("    [UK] ZIP failed - falling back to FRED OECD 10Y ...")
    return _uk_fred(start, end)


def _uk_boe_zip(start, end):
    """Download BoE GLC Nominal ZIP, parse all relevant Excel files."""
    try:
        hdr = dict(_BROWSER_HDR)
        hdr["Referer"] = "https://www.bankofengland.co.uk/statistics/yield-curves"
        r = requests.get(BOE_ZIP_URL, headers=hdr, timeout=60)
        r.raise_for_status()

        zf = zipfile.ZipFile(io.BytesIO(r.content))
        all_frames = []

        start_yr = pd.to_datetime(start).year
        end_yr   = pd.to_datetime(end).year

        for name in sorted(zf.namelist()):
            if not name.endswith(".xlsx"): continue
            # Parse year range from filename, e.g. "...2005 to 2015.xlsx"
            m = re.search(r"(\d{4})\s+to\s+(\d{4}|present)", name, re.I)
            if m:
                file_start = int(m.group(1))
                file_end   = 9999 if "present" in m.group(2).lower() else int(m.group(2))
                # Skip if no overlap with requested range
                if file_end < start_yr or file_start > end_yr:
                    continue

            try:
                xl = pd.ExcelFile(io.BytesIO(zf.read(name)), engine="openpyxl")
                # Try both sheet name variants
                sheet = None
                for candidate in ["4. spot curve", "4. nominal spot curve"]:
                    if candidate in xl.sheet_names:
                        sheet = candidate; break
                if sheet is None: continue

                raw = xl.parse(sheet, header=None)

                # Row 3 = maturity headers (in years), col 0 = "years:"
                # Data from row 5 onwards, col 0 = date
                mat_row = raw.iloc[3, 1:]         # maturities as floats
                dates   = pd.to_datetime(raw.iloc[5:, 0], errors="coerce")
                data    = raw.iloc[5:, 1:].copy()
                data.index = dates
                data.columns = range(len(data.columns))

                # Map maturity floats -> tenor labels
                rename = {}
                for i, val in enumerate(mat_row):
                    try:
                        v = float(val)
                        if v in BOE_TENOR_MAP:
                            rename[i] = BOE_TENOR_MAP[v]
                    except Exception: pass
                data = data.rename(columns=rename)
                keep = [c for c in BOE_TARGET if c in data.columns]
                if not keep: continue

                chunk = data[keep].apply(pd.to_numeric, errors="coerce").dropna(how="all")
                chunk.index.name = "date"
                all_frames.append(chunk)

            except Exception as e:
                print(f"    [WARN] parsing {name}: {e}")
                continue

        if not all_frames: return pd.DataFrame()

        df = pd.concat(all_frames).sort_index()
        df = df[~df.index.isna()]
        df = df.loc[start:end]
        if df.empty: return pd.DataFrame()
        df = df.reindex(columns=COMMON_TENORS)
        _summary("UK", df); return df

    except Exception as e:
        print(f"    [WARN] BoE ZIP download failed: {e}"); return pd.DataFrame()


def _uk_fred(start, end):
    """Last resort: FRED OECD monthly 10Y UK gilt, forward-filled to daily."""
    try:
        import pandas_datareader.data as web
        s = web.DataReader("IRLTLT01GBM156N", "fred", start, end)["IRLTLT01GBM156N"]
        df = s.to_frame("10Y")
        df.index = pd.to_datetime(df.index); df.index.name = "date"
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        idx = pd.date_range(df.index.min(), df.index.max(), freq="B")
        df = df.reindex(idx).ffill(); df.index.name = "date"
        _summary("UK (FRED fallback - 10Y monthly->daily)", df); return df
    except Exception as e:
        print(f"    [ERROR] FRED UK: {e}"); return pd.DataFrame()


# ─────────────────────────── Japan (MoF) ──────────────────────────────────────
#
# Japan MoF CSV format:
#   Row 0: "Interest Rate,,,,,,,,,,,,,,,(Unit : %)"   <- title, skip
#   Row 1: "Date,1Y,2Y,3Y,4Y,5Y,6Y,7Y,8Y,9Y,10Y,15Y,20Y,25Y,30Y,40Y"  <- header
#   Row 2+: YYYY/M/D, values (- for missing)

JP_MOF_URL = ("https://www.mof.go.jp/english/policy/jgbs/reference/"
              "interest_rate/historical/jgbcme_all.csv")
JP_TARGET  = ["1Y","2Y","3Y","5Y","10Y","20Y","30Y"]
JP_FRED    = {"10Y": "IRLTLT01JPM156N"}


def fetch_jp_yields(start=DEFAULT_START, end=DEFAULT_END):
    """
    Fetch Japan JGB benchmark yields.
    Primary:  Japan MoF daily CSV (multi-tenor, from 1974)
    Fallback: FRED OECD monthly 10Y
    """
    if end is None: end = DEFAULT_END
    print("  [JP] Fetching from Japan Ministry of Finance ...")
    df = _jp_mof(start, end)
    if not df.empty: return df
    print("    [JP] MoF failed - falling back to FRED OECD monthly 10Y ...")
    return _jp_fred(start, end)


def _jp_mof(start, end):
    try:
        hdr = dict(_BROWSER_HDR)
        r = requests.get(JP_MOF_URL, headers=hdr, timeout=45)
        r.raise_for_status()

        # Skip the title row (row 0), use row 1 as header
        df = pd.read_csv(
            StringIO(r.text),
            skiprows=1,           # skip "Interest Rate,...,(Unit : %)"
            na_values=["-","N/A",""],
        )
        # Column 0 is "Date", rest are "1Y","2Y",...
        df.columns = [str(c).strip() for c in df.columns]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        df.index.name = "date"

        keep = [c for c in JP_TARGET if c in df.columns]
        if not keep:
            print(f"    [WARN] MoF: cols available: {list(df.columns[:10])}")
            return pd.DataFrame()

        df = df[keep].apply(pd.to_numeric, errors="coerce").dropna(how="all")
        df = df.loc[start:end]
        if df.empty: return pd.DataFrame()
        df = df.reindex(columns=COMMON_TENORS)
        _summary("JP", df); return df

    except Exception as e:
        print(f"    [WARN] MoF fetch failed: {e}"); return pd.DataFrame()


def _jp_fred(start, end):
    try:
        import pandas_datareader.data as web
        frames = {}
        for tenor, sid in JP_FRED.items():
            try: frames[tenor] = web.DataReader(sid,"fred",start,end)[sid]
            except Exception as e: print(f"    [WARN] FRED {sid}: {e}")
        if not frames: return pd.DataFrame()
        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index); df.index.name = "date"
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        idx = pd.date_range(df.index.min(), df.index.max(), freq="B")
        df = df.reindex(idx).ffill(); df.index.name = "date"
        _summary("JP (FRED fallback - 10Y monthly->daily)", df); return df
    except Exception as e:
        print(f"    [ERROR] FRED JP: {e}"); return pd.DataFrame()


# ─────────────────────────── Combine & Save ───────────────────────────────────

FETCHERS = {
    "US": fetch_us_yields,
    "DE": fetch_de_yields,
    "UK": fetch_uk_yields,
    "JP": fetch_jp_yields,
}


def fetch_all(countries=None, start=DEFAULT_START, end=DEFAULT_END, save=True):
    if countries is None: countries = list(FETCHERS.keys())
    results = {}
    for country in countries:
        if country not in FETCHERS:
            print(f"[WARN] Unknown: '{country}'"); continue
        print(f"\n{'─'*50}")
        df = FETCHERS[country](start=start, end=end)
        if df.empty:
            print(f"  [SKIP] {country} -- no data."); continue
        results[country] = df
        if save:
            path = os.path.join(YIELD_DIR, f"{country.lower()}_yields.csv")
            df.to_csv(path); print(f"  Saved -> {path}")
    return results


def build_combined(results, save=True):
    if not results:
        print("[ERROR] No data to combine."); return pd.DataFrame()
    dfs = []
    for country, df in results.items():
        df.columns = pd.MultiIndex.from_tuples(
            [(country, c) for c in df.columns], names=["country","tenor"])
        dfs.append(df)
    combined = pd.concat(dfs, axis=1).sort_index().ffill(limit=5)
    if save:
        path = os.path.join(YIELD_DIR, "combined_yields.csv")
        combined.to_csv(path)
        print(f"\n{'='*50}")
        print(f"Combined -> {path}")
        print(f"Shape: {combined.shape[0]} rows x {combined.shape[1]} cols")
        print(f"Dates: {combined.index[0].date()} -> {combined.index[-1].date()}")
    return combined


def _summary(country, df):
    t0 = df.index[0].date() if len(df) else "--"
    t1 = df.index[-1].date() if len(df) else "--"
    print(f"  [{country}] OK  {len(df):,} rows | {list(df.columns)} | {t0} -> {t1}")


def load_combined(path=None):
    if path is None: path = os.path.join(YIELD_DIR, "combined_yields.csv")
    df = pd.read_csv(path, header=[0,1], index_col=0, parse_dates=True)
    df.index.name = "date"; return df


# ─────────────────────────── CLI ──────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fetch multi-country sovereign yields.")
    p.add_argument("--start",     default=DEFAULT_START)
    p.add_argument("--end",       default=DEFAULT_END)
    p.add_argument("--countries", nargs="+", default=None)
    p.add_argument("--no-save",   action="store_true")
    a = p.parse_args()
    print("="*50)
    print("  Multi-Country Yield Data Fetcher")
    print(f"  Period: {a.start}  ->  {a.end}")
    print("="*50)
    results  = fetch_all(a.countries, a.start, a.end, save=not a.no_save)
    combined = build_combined(results, save=not a.no_save)
    print("\nDone.")
