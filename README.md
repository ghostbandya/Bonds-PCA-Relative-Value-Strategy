# Applied Quantitative Macro Strategies
## PCA-Based Sovereign Bond Mean-Reversion + Carry/Roll-Down


---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Strategy Logic](#2-strategy-logic)
3. [Repository Structure](#3-repository-structure)
4. [Data Sources](#4-data-sources)
5. [Installation & Setup](#5-installation--setup)
6. [How to Run](#6-how-to-run)
7. [Key Parameters](#7-key-parameters)
8. [Output Files](#8-output-files)
9. [Mathematical Background](#9-mathematical-background)
10. [Key Design Decisions](#10-key-design-decisions)
11. [Results Summary](#11-results-summary)

---

## 1. Project Overview

This project implements two complementary systematic fixed-income strategies on
G4 sovereign yield curves (US Treasuries, German Bunds, UK Gilts, Japanese JGBs):

| Strategy | Signal Source | P&L Driver |
|---|---|---|
| **PCA Mean-Reversion** | Rolling PCA residuals + OU S-scores | Yield spread reversion to factor-implied fair value |
| **Carry + Roll-Down** | Carry (yield − short rate) + roll-down (curve roll) | Systematic yield income + curve roll capital gain |

Both strategies are filtered by a **3-regime Hidden Markov Model** that classifies
the market environment as GOOD / NEUTRAL / BAD based on PCA structural stability.
Positions are sized at full / half / zero accordingly.

---

## 2. Strategy Logic

### 2.1 PCA Mean-Reversion

The core insight from **Credit Suisse "PCA Unleashed" (2012):** the entire yield curve
across multiple countries can be described by **3 uncorrelated factors** — Level,
Slope, and Curvature. After projecting out these factors, the remaining
**idiosyncratic residual** of each bond's yield mean-reverts strongly.


**Pipeline:**
```
Daily Yield Changes → Rolling PCA (252-day window) → Factor Scores + Residuals
→ Cumulate Residuals → Fit OU Process (60-day window) → S-Score
→ Regime Filter → Position Sizing → Factor-Neutral P&L
```

**Signal:**
```
S-score = X_i(t) / σ_eq,i

  S > +1.25  →  LONG  (yield above model = bond cheap)
  S < −1.25  →  SHORT (yield below model = bond rich)
  |S| < 0.75 →  close long
  |S| < 0.50 →  close short
```

> **Note on yield vs equity convention**: In equity stat-arb (Avellaneda & Lee),
> a high residual = overpriced stock = go SHORT. In bonds, a high residual means
> the *yield* is elevated = the *price* is depressed = the bond is CHEAP = go LONG.
> Signs are inverted relative to equities.

### 2.2 Carry + Roll-Down

Harvests two structural premia in government bonds:

- **Carry** = yield_T − short_rate (excess yield over financing cost)
- **Roll-Down** = yield_T − yield_{T−1Y} (capital gain from curve ageing)

```
Combined Score_i = (Carry_i + Roll-Down_i) / DV01_i
```

Dividing by DV01 makes signals comparable across tenors (duration-normalised).
The carry strategy uses **raw yield changes** (not factor-neutral residuals) as its
P&L driver — because carry *is* the systematic level signal that PCA would subtract out.

### 2.3 Regime Filter

A 3-state Gaussian HMM classifies each date based on PCA stability metrics:

| Regime | Meaning | Signal Action |
|---|---|---|
| **GOOD** (0) | PCA structure stable, high R², low eigenvector rotation | Full position (±1.0) |
| **NEUTRAL** (1) | Moderate structural shift | Half position (±0.5) |
| **BAD** (2) | PCA broken, structural break | All positions closed |

Features fed to HMM (mapped to expanding percentile ranks before fitting):
- `cum_var` — cumulative variance explained by PC1+PC2+PC3
- `mean_cos_sim` — average cosine similarity of eigenvectors vs. previous day
- `r_squared` — mean R² of factor model across all instruments
- `var_pc1` — fraction of variance in PC1 alone

---

## 3. Repository Structure

```
Applied Quant Macro Strategies/
│
├── main.py                     # Orchestrator — runs the full pipeline end-to-end
│
├── src/
│   ├── _01_data_fetch.py       # Fetch sovereign yields from FRED, ECB, BoE, MoF
│   ├── _02_data_prep.py        # Clean, align, and build cross-market panel
│   ├── _03_rolling_pca.py      # Rolling PCA engine (correlation matrix, residuals)
│   ├── _04_regime_detection.py # 3-regime HMM classifier on PCA stability features
│   ├── _05_signal_generation.py# OU fitting, S-score computation, position sizing
│   ├── _06_backtest.py         # P&L engine, transaction costs, performance metrics
│   ├── _07_visualisation.py    # All plots (yield curves, PCA, regimes, P&L, comparison)
│   └── _08_carry_signal.py     # Carry + Roll-Down strategy
│
├── data/
│   ├── yields/
│   │   ├── combined_yields.csv         # MultiIndex (country, tenor) — raw levels
│   │   ├── yields_clean.csv            # Cleaned & aligned levels
│   │   ├── yield_changes.csv           # Daily first-differences (PCA input)
│   │   ├── cross_market_changes.csv    # Flat panel: US_1Y, US_2Y, ..., JP_30Y
│   │   ├── us_yields.csv               # US-only (FRED)
│   │   ├── de_yields.csv               # Germany (ECB)
│   │   ├── uk_yields.csv               # UK (BoE)
│   │   └── jp_yields.csv               # Japan (MoF)
│   ├── bond_futures_px_last.csv        # Optional: Bloomberg futures prices
│   └── bond_futures_metadata.csv       # Optional: futures contract metadata
│
└── outputs/
    ├── pca/                    # residuals.csv, factor_scores.csv, var_explained.csv, ...
    ├── regimes/                # regime.csv, regime_label.csv, features.csv, ...
    ├── signals/                # s_scores.csv, positions.csv, signals.csv, ou_params.csv
    ├── backtest/               # pnl_daily.csv, pnl_total.csv, metrics.csv
    ├── carry/                  # carry_roll.csv, carry_positions.csv, carry_signals.csv
    ├── carry_backtest/         # carry_pnl_daily.csv, carry_pnl_total.csv, carry_metrics.csv
    └── plots/                  # All PNG figures
        ├── yield_curves.png
        ├── pca_variance_explained.png
        ├── pca_loadings.png
        ├── eigenvector_stability.png
        ├── regime_timeline.png
        ├── s_scores.png
        ├── pnl_curve.png
        ├── regime_pnl_breakdown.png
        └── strategy_comparison.png
```

---

## 4. Data Sources

| Country | Source | URL | Tenors |
|---|---|---|---|
| US | FRED (Federal Reserve) | `fred.stlouisfed.org` | DGS1, DGS2, DGS3, DGS5, DGS10, DGS20, DGS30 |
| DE | ECB SDMX-REST API | `data-api.ecb.europa.eu` | Parametric spot curve at 1Y, 2Y, 3Y, 5Y, 10Y, 20Y, 30Y |
| UK | Bank of England GLC ZIP | `bankofengland.co.uk` | Nominal spot curve from ZIP of Excel files |
| JP | Japan Ministry of Finance | `mof.go.jp` | JGB benchmark yields, all tenors |

All data is **free and requires no API key**. The ECB fetcher makes requests in
parallel (8 simultaneous) to avoid serial latency. The BoE fetcher downloads a
single ZIP containing multiple Excel files spanning different year ranges.

**Common tenor set**: `1Y, 2Y, 3Y, 5Y, 10Y, 20Y, 30Y`
- 7Y excluded: inconsistent availability across all sources
- 15Y excluded: FRED has no DGS15 series; to keep country data symmetric it is
  dropped from all countries rather than interpolated for US

---

## 5. Installation & Setup

### Requirements
```
Python 3.10+
pandas, numpy, scipy, scikit-learn
requests, openpyxl
pandas_datareader       # FRED data fetch
hmmlearn                # Gaussian HMM for regime detection
matplotlib, seaborn     # Visualisation
```

### Install
```bash
pip install pandas numpy scipy scikit-learn requests openpyxl \
            pandas_datareader hmmlearn matplotlib seaborn
```

---

## 6. How to Run

### Full pipeline (fetch + all steps)
```bash
python main.py
```

### Skip data fetch (use cached CSVs)
```bash
python main.py --skip-fetch
```

### Skip plots (faster, useful for iterating on parameters)
```bash
python main.py --skip-fetch --no-plots
```

### Rule-based regimes instead of HMM
```bash
python main.py --skip-fetch --regime-method rules
```

### Run individual modules
```bash
python src/_01_data_fetch.py --start 2005-01-01
python src/_02_data_prep.py
python src/_03_rolling_pca.py --mode cross --k 3 --corr-window 252
python src/_04_regime_detection.py --method hmm
```

### Pipeline flow
```
Step 1  _01_data_fetch.py    → data/yields/*.csv
Step 2  _02_data_prep.py     → data/yields/yields_clean.csv, yield_changes.csv
Step 3  _03_rolling_pca.py   → outputs/pca/*.csv
Step 4  _04_regime_detection → outputs/regimes/*.csv
Step 5  _05_signal_generation→ outputs/signals/*.csv
Step 6  _06_backtest.py      → outputs/backtest/*.csv   (PCA strategy)
Step 7  _08_carry_signal.py  → outputs/carry/*.csv
Step 7b _06_backtest.py      → outputs/carry_backtest/  (Carry strategy)
Step 8  _07_visualisation.py → outputs/plots/*.png
```

---

## 7. Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--start` | `2000-01-01` | History start date |
| `--countries` | `US DE UK JP` | Countries included in PCA |
| `--trade-countries` | `US DE UK` | Countries actually traded (JP excluded — BoJ YCC) |
| `--corr-window` | `252` | Rolling window for PCA correlation matrix (1 year) |
| `--resid-window` | `60` | Inner window for OU parameter estimation (~3 months) |
| `--k` | `3` | Number of principal components retained |
| `--regime-method` | `hmm` | Regime classifier: `hmm` or `rules` |
| `--smooth` | `21` | Rolling median window for regime features (1 month) |
| `--s-bo` | `1.25` | S-score threshold to open long (bond cheap) |
| `--s-bc` | `0.75` | S-score threshold to close long |
| `--s-so` | `1.25` | S-score threshold to open short (bond rich) |
| `--s-sc` | `0.50` | S-score threshold to close short |
| `--tc` | `0.0002` | Transaction cost per trade (2 bps, one-way) |

### Why Japan is excluded from trading
The Bank of Japan's Yield Curve Control (YCC) policy from 2012 to 2024 pegged
the 10Y JGB yield at a target level (initially 0%, later widened). This policy
mechanically prevented JGB yields from freely mean-reverting — any residual
deviation was absorbed by BoJ intervention, not by market mean-reversion.
Including JP in *PCA* improves factor estimation (more instruments = better
spanning of the global factor space). Including it in *trading* would generate
spurious signals on a yield series that was administratively pegged.

---

## 8. Output Files

### PCA outputs (`outputs/pca/`)
| File | Description |
|---|---|
| `residuals.csv` | Cumulated OU residuals X_i(t) — one column per instrument |
| `factor_scores.csv` | Daily PC1, PC2, PC3 time series |
| `var_explained.csv` | Fraction of variance per PC, plus cumulative |
| `eigenvalues.csv` | Raw eigenvalues λ1, λ2, λ3 per date |
| `ev_stability.csv` | Cosine similarity of each PC to previous day's eigenvector |
| `r_squared.csv` | Mean R² of factor model across all instruments |

### Signal outputs (`outputs/signals/`)
| File | Description |
|---|---|
| `s_scores.csv` | S-score per instrument per date (NaN if κ too small) |
| `positions.csv` | Target position per instrument: +1, +0.5, 0, -0.5, -1 |
| `signals.csv` | Raw open/close signals: +1 (open long), -1 (open short), 0 |
| `ou_params.csv` | OU parameters (κ, μ, σ_eq, half-life, R²) per (date, instrument) |

### Performance metrics (both strategies)
| Metric | Description |
|---|---|
| `ann_return_%` | Annualised return (daily mean × 252) |
| `ann_vol_%` | Annualised volatility (daily std × √252) |
| `sharpe` | Annualised Sharpe ratio |
| `max_drawdown_%` | Maximum peak-to-trough drawdown |
| `win_rate_%` | Fraction of days with positive P&L |
| `n_days` | Number of trading days in regime/overall |

Metrics are reported for Overall, GOOD, NEUTRAL, and BAD regimes separately.

---

## 9. Mathematical Background

### 9.1 Rolling PCA

Given daily yield changes $\Delta Y \in \mathbb{R}^{T \times N}$ over a 252-day window:

**Step 1 — Standardise** (zero mean, unit variance per instrument):
$$Z = \text{StandardScaler}(\Delta Y) \in \mathbb{R}^{T \times N}$$

**Step 2 — Correlation matrix:**
$$C = \frac{Z^\top Z}{T - 1} \in \mathbb{R}^{N \times N}$$

**Step 3 — Eigendecomposition:**
$$C = V \Lambda V^\top, \quad \Lambda = \text{diag}(\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_N)$$

**Step 4 — Keep top k=3 eigenvectors** $V_k \in \mathbb{R}^{N \times k}$:
$$\text{VarExplained}_j = \frac{\lambda_j}{\sum_i \lambda_i}$$

**Step 5 — Factor scores and residuals** (inner 60-day window):
$$F = Z \cdot V_k \in \mathbb{R}^{T \times k}$$
$$\hat{\Delta y}_i = \beta_{i1} F_1 + \beta_{i2} F_2 + \beta_{i3} F_3 \qquad (\text{OLS, no intercept})$$
$$\varepsilon_i(t) = \Delta y_i(t) - \hat{\Delta y}_i(t) \qquad X_i(t) = \sum_{s \leq t} \varepsilon_i(s)$$

### 9.2 Ornstein-Uhlenbeck Model

The cumulated residual $X_i(t)$ is modelled as:
$$dX = \kappa(\mu - X)\, dt + \sigma\, dW$$

Estimated via OLS on the discrete AR(1):
$$\Delta X(t) = a + b \cdot X(t-1) + \varepsilon$$

Parameter mapping:
$$\kappa = -\frac{\log(1 + b)}{\Delta t}, \quad \mu = -\frac{a}{b}, \quad \sigma_{eq} = \frac{\sigma_{OU}}{\sqrt{2\kappa}}$$

S-score:
$$S_i(t) = \frac{X_i(t) - \mu_i}{\sigma_{eq,i}}$$

$\sigma_{eq}$ is floored at $0.60 \times \text{std}(X_i)$ to prevent blow-up when $\kappa \to \infty$.
S-scores are hard-capped at $\pm 5$.

Filter: only trade if $\kappa > 8.4$ (half-life $< 30$ days).

### 9.3 Duration-Neutral P&L

Bond price sensitivity to yield: $\Delta P_i \approx -\text{DV01}_i \times \Delta y_i$

We use a **constant** $\text{DV01}_{base} = 8.60$ (10Y reference) for all instruments:
$$\text{pnl}_i(t) = -\text{DV01}_{base} \times \text{pos}_i(t-1) \times \Delta\varepsilon_i(t) / 100$$

This is duration-neutral sizing: a 1 bps residual move contributes equal P&L
regardless of whether the instrument is a 1Y or a 30Y bond.

### 9.4 Carry + Roll-Down

$$\text{Carry}_i = y_T - y_{2Y} \qquad \text{(yield minus short rate)}$$

$$\text{RollDown}_i = y_T - y_{T-1Y} \qquad \text{(interpolated from curve)}$$

$$\text{Score}_i = \frac{\text{Carry}_i + \text{RollDown}_i}{\text{DV01}_i}$$

---

## 10. Key Design Decisions

### Why correlation matrix (not covariance)?
Covariance PCA lets high-volatility tenors dominate PC1 just because they move
more in absolute bps. We want PC1 to represent a structural parallel shift, not
volatility-dominated variance. Standardising first gives each tenor equal weight.

### Why rolling 252-day window?
The covariance structure of yields changes over time (2008, 2020, 2022 were very
different). A static full-sample PCA uses stale loadings. Rolling 252 days (~1 year)
keeps the model current while using enough data for stable covariance estimation.

### Why 60-day inner window for OU?
60 days is short enough to reflect current mean-reversion speed (not 1-year-old
dynamics) but long enough for OLS to estimate the AR(1) reliably (minimum ~30 obs).

### Why exclude Japan from trading?
BoJ Yield Curve Control (2012–2024) suppressed JGB yield mean-reversion. JP stays
in the PCA to improve global factor estimation but is excluded from signal generation.

### Why factor-neutral P&L for PCA (not raw yield changes)?
Using daily diffs of cumulated residuals as the P&L driver removes all systematic
factor moves (PC1/PC2/PC3) from P&L. Only the idiosyncratic spread reversion counts.
This gives a true measure of the strategy's alpha.

### Why raw yield changes for carry (not residuals)?
Carry is the systematic level signal — it IS PC1 in a sense. Subtracting out the
PCA factor would remove exactly what carry tries to capture.

### Why percentile normalisation before HMM?
Cross-market PCA features live in very narrow absolute ranges (e.g. cum_var 0.69–0.84).
A Gaussian HMM trained on absolute values cannot separate regimes. Converting to
expanding percentile ranks maps each day to its position in historical distribution,
giving the HMM well-separated [0,1] inputs.

---

## 11. Results Summary

Based on the latest pipeline run (US, DE, UK, JP; 2000–2026; 28 instruments;
7 tenors × 4 countries):

### PCA Mean-Reversion Strategy (23 tradeable instruments: US + DE + UK)

| Regime | Sharpe | Ann Return | Ann Vol | Max Drawdown |
|---|---|---|---|---|
| **Overall** | 0.16 | 0.21% | 2.3% | -10.0% |
| **GOOD** | **0.50** | 0.92% | 3.3% | -3.0% |
| NEUTRAL | 0.38 | 0.05% | 0.1% | 0.0% |
| BAD | -0.37 | -0.59% | 1.6% | -6.0% |

### Carry + Roll-Down Strategy (same universe)

| Regime | Sharpe | Ann Return | Ann Vol | Max Drawdown |
|---|---|---|---|---|
| **Overall** | 0.10 | 0.30% | 2.9% | -15.4% |
| **GOOD** | **0.40** | 1.48% | 3.7% | -11.4% |
| NEUTRAL | 0.38 | 0.04% | 0.1% | 0.0% |
| BAD | -0.50 | -1.40% | 2.8% | -10.7% |

**Key takeaways:**
- PCA mean-reversion has lower drawdown (-10% vs -15%) and better regime conditioning
- Carry is hurt by the 2022 rate hike cycle (long bias collides with rapid yield rises)
- Both strategies generate Sharpe ~0.4–0.5 in GOOD regimes — the regime filter works
- BAD regime is effectively flat for both (regime filter does its job)
- The GOOD Sharpe is the most meaningful metric: it measures alpha in the environment
  the strategy is designed for

---
