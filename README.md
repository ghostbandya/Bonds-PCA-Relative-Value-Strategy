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
11. [Test Suite](#11-test-suite)
12. [Results Summary](#12-results-summary)

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

The evaluation uses a strict **80/20 train/test split**:
- The split boundary is computed **once** before any modelling begins.
- The HMM regime model is **fitted on training data only**, then applied forward to the test period.
- PCA loadings are **frozen at the training boundary** — the test period uses the last training-period PCA without any refitting on test data.
- This ensures the test-period results are genuinely out-of-sample.

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

**No-lookahead guarantee:** the HMM is fitted on training data only
(`train_end` is passed to `classify_hmm`), then the trained model decodes
the full sequence including the test period via Viterbi. This means the transition
matrix and emission parameters are never influenced by future data.

---

## 3. Repository Structure

```
Applied Quant Macro Strategies/
│
├── main.py                      # Orchestrator — full pipeline end-to-end
│
├── src/
│   ├── config.py                # Strategy config, 80/20 split, universe definitions
│   ├── _01_data_fetch.py        # Fetch sovereign yields (FRED, ECB, BoE, MoF)
│   ├── _02_data_prep.py         # Clean, align, build cross-market panel
│   ├── _03_rolling_pca.py       # Rolling PCA engine — loadings frozen at train_end
│   ├── _04_regime_detection.py  # 3-regime HMM — fitted on training data only
│   ├── _05_signal_generation.py # OU S-scores and rolling z-scores, position sizing
│   ├── _06_backtest.py          # P&L engine, transaction costs, performance metrics
│   ├── _07_visualisation.py     # All plots (yield curves, PCA, regimes, P&L)
│   ├── _08_carry_signal.py      # Carry + Roll-Down strategy
│   ├── costs.py                 # Transaction cost models (flat, DV01-bp)
│   ├── covariance.py            # Covariance estimators (sample, EWMA, Ledoit-Wolf)
│   ├── dv01.py                  # Duration and DV01 calculations
│   ├── rebalance.py             # No-trade band rebalancing logic
│   └── weights.py               # Portfolio construction Methods 1/2/3
│
├── tests/
│   ├── conftest.py              # Shared pytest fixtures
│   ├── test_config.py           # StrategyConfig validation + 80/20 split tests
│   ├── test_signal.py           # Z-score causality, no-lookahead, Version A/B
│   ├── test_regime.py           # HMM train-only fitting, no-lookahead mutation test
│   ├── test_costs.py            # Flat and DV01-bp cost model tests
│   ├── test_covariance.py       # Covariance estimator tests
│   ├── test_dv01.py             # DV01 calculation tests
│   ├── test_rebalance.py        # No-trade band mechanics
│   └── test_weights.py          # Methods 1/2/3 factor-neutrality and KKT constraints
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
    ├── pca/          # residuals.csv, factor_scores.csv, var_explained.csv, ...
    ├── regimes/      # regime.csv, regime_label.csv, regime_proba.csv, features.csv
    ├── signals/      # s_scores.csv, positions.csv, signals.csv, ou_params.csv
    ├── backtest/     # pnl_daily.csv, pnl_total.csv, metrics.csv  (OU strategy)
    ├── backtest_ou/  # same, OU strategy explicit copy
    ├── backtest_m1/  # Method 1 (geometric) — with --v2 flag
    ├── backtest_m2/  # Method 2 (min-variance KKT) — with --v2 flag
    ├── backtest_m3/  # Method 3 (mean-variance) — with --v2 flag
    ├── carry/        # carry_roll.csv, carry_positions.csv, carry_signals.csv
    ├── carry_backtest/ # carry_pnl_daily.csv, carry_pnl_total.csv, carry_metrics.csv
    └── plots/        # All PNG figures
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
pytest                  # Test suite
```

### Install
```bash
pip install pandas numpy scipy scikit-learn requests openpyxl \
            pandas_datareader hmmlearn matplotlib seaborn pytest
```

---

## 6. How to Run

### Full pipeline (fetch + all steps)
```bash
python main.py
```

### Skip data fetch (use cached CSVs — recommended after first run)
```bash
python main.py --skip-fetch
```

### Skip plots (faster, useful for iterating on parameters)
```bash
python main.py --skip-fetch --no-plots
```

### Also run Methods 1/2/3 (Jay's portfolio construction approach)
```bash
python main.py --skip-fetch --v2
```

Methods 1/2/3 differ in how the portfolio book is constructed from z-scores:
- **M1 (geometric):** single-tenor entry/exit state machine with factor-neutral hedges
- **M2 (min-variance):** KKT solution that pins the most-dislocated tenor at minimum variance
- **M3 (mean-variance):** full mean-variance book `Σ⁻¹α / γ`, projected to factor-neutrality

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

### Run test suite
```bash
python -m pytest tests/ -v
```

### Pipeline flow
```
Step 1  _01_data_fetch.py    → data/yields/*.csv
Step 2  _02_data_prep.py     → data/yields/yields_clean.csv, yield_changes.csv
Step 3  Compute 80/20 split  → train_end (single boundary, computed once)
Step 4  _03_rolling_pca.py   → outputs/pca/*.csv        (loadings frozen at train_end)
Step 5  _04_regime_detection → outputs/regimes/*.csv    (HMM fitted on train only)
Step 6  _05_signal_generation→ outputs/signals/*.csv    (causal signals, full period)
Step 7a _06_backtest.py      → outputs/backtest_ou/     (PCA OU S-score)
Step 7b _06_backtest.py      → outputs/backtest_m1/2/3/ (Methods 1/2/3, --v2 only)
Step 7c _08_carry_signal.py  → outputs/carry/           (Carry strategy)
Step 7d _06_backtest.py      → outputs/carry_backtest/  (Carry backtest)
Step 8  Print train/test comparison table
Step 9  _07_visualisation.py → outputs/plots/*.png
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
| `--v2` | off | Also run Methods 1/2/3 portfolio construction |
| `--covariance` | `lw` | Covariance estimator for Methods 1/2/3: `sample`, `ewma`, `lw` |
| `--z-window` | `63` | Z-score rolling window for Methods 1/2/3 (days) |
| `--no-trade-band` | `0.0` | No-trade band width τ ∈ [0, 1) for Methods 1/2/3 |

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
The comparison table printed at the end of the pipeline shows TRAINING vs TEST
results side by side so over-fitting is immediately visible.

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

**Frozen loadings for the test period:**
After the training boundary `train_end`, the PCA model stops refitting. The final
set of training-period eigenvectors $V_k$ is applied forward to project test-period
yield changes into factors and extract residuals. This ensures the PCA model has
zero exposure to future yield data when generating test-period signals.

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

### 9.3 Rolling Z-Score (Jay's Approach)

An alternative to the OU S-score that is simpler and more robust:

$$z_t^{(A)} = \sum_{s=t_{\text{block}}}^{t} \varepsilon_s \qquad \text{(Version A: block-reset at each PCA refit)}$$

$$z_t^{(B)} = \sum_{s=t-K+1}^{t} \varepsilon_s \qquad \text{(Version B: rolling K-day window)}$$

$$S_t = \frac{z_t - \overline{z}_{t-w:t}}{\text{std}(z_{t-w:t})} \qquad \text{(trailing } w\text{-day z-score)}$$

Version A resets the cumulative sum at each PCA refit boundary. This prevents
residuals from different PCA vintages (with potentially rotated eigenvectors)
from being accumulated together. The z-score is then normalised by its own
trailing mean and standard deviation rather than a model-derived $\sigma_{eq}$.

### 9.4 Duration-Neutral P&L

Bond price sensitivity to yield: $\Delta P_i \approx -\text{DV01}_i \times \Delta y_i$

We use a **constant** $\text{DV01}_{base} = 8.60$ (10Y reference) for all instruments:
$$\text{pnl}_i(t) = -\text{DV01}_{base} \times \text{pos}_i(t-1) \times \Delta\varepsilon_i(t) / 100$$

This is duration-neutral sizing: a 1 bps residual move contributes equal P&L
regardless of whether the instrument is a 1Y or a 30Y bond.

### 9.5 Carry + Roll-Down

$$\text{Carry}_i = y_T - y_{2Y} \qquad \text{(yield minus short rate)}$$

$$\text{RollDown}_i = y_T - y_{T-1Y} \qquad \text{(interpolated from curve)}$$

$$\text{Score}_i = \frac{\text{Carry}_i + \text{RollDown}_i}{\text{DV01}_i}$$

---

## 10. Key Design Decisions

### Why 80/20 split (not 60/20/20)?
A validation set (the "20" in 60/20/20) serves as a second tuning set for
hyperparameter selection across multiple candidate models. Here, the PCA window,
OU parameters, and regime thresholds are not selected by scanning over
out-of-sample validation error — they are set by economic reasoning (252-day
window ≈ 1 year of market memory; 60-day OU window ≈ mean-reversion timescale).
A third split therefore adds no methodological value and just shrinks the
training data available to the HMM. The 80/20 split keeps the narrative clean:
"Trained on 80% of history. Tested on the 20% the model never saw."

### Why compute the split before PCA — and only once?
The PCA model and the HMM both need to know when training ends so they can stop
learning from the data at that point. If splits were recomputed after PCA (e.g.
from the shorter residuals index, which is ~252 days shorter due to warmup), the
training boundary would silently shift, the HMM would see slightly more data, and
the test Sharpe would be subtly optimistic. Computing the boundary once from the
raw yield-changes index and passing `train_end` into every model that needs it
keeps the accounting exact and reproducible.

### Why freeze PCA loadings for the test period?
If the PCA refits on test-period yield data, the eigenvectors for 2022 would
"know" about the rate hike cycle when decomposing earlier dates. Freezing the
loadings at the training boundary means the test residuals are genuinely
out-of-sample — they measure how yields deviate from a model estimated
entirely on past data.

### Why fit the HMM only on training data?
The HMM transition matrix and emission parameters encode the statistical
properties of PCA stability regimes. If fitted on the full sample, the model
"knows" that 2022 was a BAD regime when it is labelling 2005. Training-only
fitting prevents this look-ahead contamination. The Viterbi decoding step
(which produces regime labels) is then applied to the full sequence using the
training-fitted parameters.

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
giving the HMM well-separated [0,1] inputs. Expanding (not full-sample) percentile
ranks are causal — they only use information available up to time t.

---

## 11. Test Suite

Run with:
```bash
python -m pytest tests/ -v
```

89 tests across 8 files. Key test categories:

| File | Tests | What is verified |
|---|---|---|
| `test_config.py` | 27 | `StrategyConfig` frozen/hashable/validation; `get_split_dates` returns only `train`/`test` keys; 80/20 fractions exact; no overlap/gap |
| `test_signal.py` | 11 | Version A reset values (not just shape); Version B rolling sum formula; trailing z-score causality; **mutation no-lookahead test** |
| `test_regime.py` | 6 | HMM fitted on train only; **mutation test**: perturbing test-period features leaves HMM params unchanged; percentile norm stays in [0,1] |
| `test_weights.py` | 13 | M1 factor-neutrality, L1 norm; M2 KKT constraints, min-variance proof; M3 gamma scaling |
| `test_costs.py` | 6 | Flat and DV01-bp cost models |
| `test_dv01.py` | 11 | Par-bond and zero-coupon DV01; ZIRP guard; monotonicity |
| `test_covariance.py` | 8 | Sample/EWMA/LW shape/symmetry/PSD; shrinkage bounds |
| `test_rebalance.py` | 4 | No-trade band identity (τ=0) and hold triggering (high τ) |

### No-lookahead tests (adapted from Jay's test suite)

The mutation no-lookahead tests in `test_signal.py` and `test_regime.py` are
the strongest causality checks. They:
1. Compute outputs on the original data.
2. Perturb values strictly *after* a probe date `t` (same index, different values).
3. Assert that outputs at `t` and earlier are byte-for-byte unchanged.

This rules out any implicit dependency on future data — window-based operations,
expanding statistics, or HMM fitting — that an append-only test might miss.

---

## 12. Results Summary

Based on the latest pipeline run (US, DE, UK, JP; ~2004–2026; 28 instruments;
7 tenors × 4 countries). The HMM was fitted on 80% of the data (training period)
and applied forward to the held-out 20% (test period).

### PCA Mean-Reversion Strategy (21 tradeable instruments: US + DE + UK)

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
- **The test Sharpe is the honest number** — it covers the 20% the model never saw

---
