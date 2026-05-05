# SDS Hungarian Election Forecasting 2026 — Thesis

A Bayesian hierarchical polling aggregation and seat forecasting model for the 2026 Hungarian parliamentary election. This project combines historical polling data, district-level electoral geography, and Monte Carlo simulation to generate probabilistic forecasts of seat distributions.

## Project Overview

This model forecasts the 2026 Hungarian parliamentary election by:

1. **Aggregating polls** with bias correction and quality weighting
2. **Building forecast distributions** using the Dirichlet distribution to enforce vote share constraints
3. **Translating national forecasts to districts** using historical swing patterns
4. **Simulating seat allocation** under the Hungarian mixed-member electoral system
5. **Backtesting on 2022** to validate calibration and uncertainty estimates

### Key Features

- **Pollster bias & quality metrics**: Estimates systematic bias and RMSE for each polling organization
- **Correlated vote shares**: Uses Dirichlet distribution to avoid independent party forecasts
- **District-level swing coefficients**: Captures stable geographic patterns from 2014–2024 elections
- **Mixed-member seat allocation**: Correctly implements 106 SMD seats + 93 list seats via d'Hondt
- **5% threshold enforcement**: Applies 5% legal threshold to raw national list votes before seat allocation
- **Uncertainty quantification**: Monte Carlo simulation (1000 iterations) produces full posterior distributions
- **Sensitivity analysis**: Tests three participation scenarios (75%, base, 85%)
- **Pollster filtering**: Optional exclusion of biased pollsters for robustness checks

## Repository Structure

```
.
├── README.md                          # This file
├── main.ipynb                         # Main analysis notebook
├── data.py                            # Data loading & processing
├── model.py                           # Polling aggregation & forecasting
├── seat_allocation.py                 # Seat allocation logic
├── data/
│   └── 2026-OEVK.csv                 # 2026 OEVK district data
├── output/
│   ├── base/                         # Base model results
│   ├── 75/                           # 75% participation scenario
│   ├── 85/                           # 85% participation scenario
│   └── filtered/                     # Results excluding biased pollsters
└── visualizations/
    ├── Actual/
    ├── Base/
    ├── Filtered/
    └── Polls/
```

## Data Sources

The model uses:

- **Election results**: 2014, 2018, 2022 Hungarian parliamentary elections (by OEVK district)
- **EU Parliament elections**: 2019, 2024 (for swing coefficient calibration)
- **Polling data**: 63 national polls from 2014–2026 (9 pollsters)
- **Macroeconomic data**: Net real salary index (for alternative model validation)

All election and polling data are merged and categorized into 6 party groups:
- **Fidesz** (governing coalition)
- **Tisza** (main opposition)
- **MiHazánk** 
- **DK** 
- **MKKP** 
- **Other**

## Methodology

### 1. Pollster Bias & Quality

For each pollster `j`, we compute:

$$\text{Bias}(j,k) = \frac{1}{N_j} \sum_{e=1}^{N_j} [\text{poll}(j,k,e) - \text{result}(k,e)]$$

$$\text{RMSE}(j) = \sqrt{\frac{1}{N_j K} \sum_e \sum_k (\text{poll}(j,k,e) - \text{result}(k,e))^2}$$

Industry-wide polling error is pooled across all pollsters:

$$\sigma_{\text{poll}} = \sqrt{\frac{1}{\sum_j N_j K} \sum_j \sum_e \sum_k (\text{poll}(j,k,e) - \text{result}(k,e))^2}$$

### 2. Bias-Corrected Polling Average

Each poll is weighted by recency and pollster accuracy:

$$w(i) = \exp(-\lambda \cdot \text{days}_i) \cdot \frac{1}{\text{RMSE}(j)^2}$$

$$\mu(k) = \frac{\sum_i w(i) \cdot [\text{poll}(i,k) - \text{bias}(j,k)]}{\sum_i w(i)}$$

Default parameters: $\lambda = 0.03$ (time decay), $\alpha = 0.1$ (staleness adjustment).

### 3. Forecast Distribution (Dirichlet)

National vote shares are drawn from a **Dirichlet distribution** parameterized to match polling means $\mu(k)$ and variance $\sigma(k)^2$:

$$\sigma(k) = \sigma_{\text{poll}} \cdot (1 + \alpha \cdot \text{months\_since\_last\_poll})$$

This ensures:
- Vote shares sum to 100%
- Negative correlations between parties (if one rises, another falls)
- Calibrated uncertainty reflecting historical polling error

### 4. District-Level Swing Coefficients

For each district `d` and party `k`, compute average deviation from national result:

$$\bar{\Delta}(d,k) = \frac{1}{T} \sum_{t=1}^{T} [\text{Election}_t^{\text{district}}(d,k) - \text{Election}_t^{\text{national}}(k)]$$

Baseline: 2019 European Parliament election; applied to 2022 results to calibrate district noise $\sigma_d$.

### 5. District-Level Projection

Given a drawn national vote share $\pi(k)$, project to district $d$:

$$\text{vote}(d,k) = \pi(k) + \Delta(d,k) + \varepsilon(d), \quad \varepsilon(d) \sim \mathcal{N}(0, \sigma_d^2)$$

Then clip negatives to zero and renormalize so $\sum_k \text{vote}(d,k) = 1$.

### 6. Seat Allocation (Mixed-Member System)

For each simulation iteration:

1. **SMD winners (106 seats)**: Determine winner in each district (plurality rule)
2. **National list votes**: Sum across parties from polling-derived percentages
3. **5% threshold**: Filter parties with < 5% of **raw national list votes** (before SMD fragments)
4. **Effective list votes**: Add losing votes + surplus votes from SMD to eligible parties' list totals
5. **d'Hondt allocation**: Allocate 93 remaining seats proportionally to effective list votes
6. **Total seats**: Sum SMD + list seats for each party (max 199 total)

**Critical**: The 5% threshold is applied to raw list votes, not effective votes. Only parties surpassing this threshold compete for list seats.

### 7. Backtesting on 2022

Validation procedure:
- Use only **pre-2022 polls** (filtered to before 2022-04-03)
- Estimate pollster bias from **2014 & 2018 elections only**
- Compute polling error from pre-2022 data
- Use **2019 EP election** as swing baseline
- Run 1000 simulations targeting 2022-04-03
- Compare 95% confidence intervals to actual 2022 seat outcomes

This tests whether:
- Model uncertainty is well-calibrated
- Assumptions about polling error are reasonable
- District swing patterns are stable

## Usage

### Prerequisites

```bash
pip install pandas numpy scipy scikit-learn matplotlib openpyxl
```

### Running the Full Pipeline

Open `main.ipynb` and execute cells in order:

1. **Data loading & processing** (Section 2): Load all election, polling, and candidate data
2. **Pollster metrics** (Section 3.1): Compute bias, quality, industry-wide error
3. **Polling average** (Section 4.1): Aggregate polls with bias correction & quality weighting
4. **Forecast distribution** (Section 4.2): Build Dirichlet distributions for national vote shares
5. **Swing coefficients** (Section 5.1): Compute district deviations from 2019 EP election
6. **Backtesting** (Section 7): Validate on 2022 election
7. **Base model** (Section 8): Run 1000 simulations for 2026
8. **Scenarios** (Section 9): Run 75%, 85% participation scenarios
9. **Sensitivity** (Section 10): Filter biased pollsters and rerun

### Output Files

For each scenario (base, 75%, 85%, filtered):

- **`oevk_perc.xlsx`**: OEVK-level mean vote percentages
- **`oevk_counts.xlsx`**: OEVK-level mean vote counts
- **`megye_counts.xlsx`**: County-aggregated vote counts
- **`sim_result.xlsx`**: Party-level seat summary with 95% CI and majority probabilities

### Example: Base Model Summary

```python
# After running simulation() for 2026
display(sim_results)

# Output:
#        Party  Mean seats      95% CI  P(majority)  P(supermajority)
# 0     Fidesz          97    82–112         23%            3%
# 1      Tisza         110    59–149         66%           19%
# 2  MiHazánk           8     5–11          0%            0%
# 3         DK           0     0–0           0%            0%
# 4       MKKP           0     0–0           0%            0%
# 5      Other           0     0–0           0%            0%
```

## Key Findings (Base Model)

- **Fidesz**: 97 seats (82–112) — 23% chance of majority (100+)
- **Tisza**: 110 seats (59–149) — 66% chance of majority
- **MiHazánk**: 8 seats (5–11)
- **DK, MKKP, Other**: Below 5% threshold, excluded from list seats

**Majority threshold**: 100 seats (out of 199)
**Supermajority threshold**: 133 seats

## Model Limitations & Caveats

1. **2022 polls were inaccurate**: Historical polling error ($\sigma_{\text{poll}} \approx 4–5\%$) reflects this
2. **Swing patterns may shift**: District coefficients assume 2014–2024 patterns persist
3. **Last-minute campaigns**: Model does not account for late shifts or scandals
4. **Turnout uncertainty**: Scenarios (75%, 85%) provide sensitivity, but actual turnout is unpredictable
5. **Third-order effects**: Does not model strategic voting, coalition effects, or tactical voting
6. **Data as of**: Polls included through early 2026; newer data would require re-calibration

## Validation Results

**Backtesting on 2022**:
- Actual: Fidesz 135, DK 57, MiHazánk 6, Other 1 (199 total)
- Simulated mean: [See backtesting output in Section 7]
- **Key question**: Did actual 2022 result fall within 95% CI?

Review the backtesting output (`comparison_df`) to assess calibration.

## Extensions & Sensitivity

### 1. Participation Scenarios

Run simulations with adjusted turnout:
- **75% participation**: Reduces all vote counts proportionally
- **85% participation**: Increases all vote counts proportionally
- Results stored in `output/75/` and `output/85/`

### 2. Pollster Filtering

Exclude known biased pollsters (Alapjogokért Központ, Századvég, Nézőpont):
- Re-compute bias, quality, polling average
- Run simulation with filtered polls
- Results stored in `output/filtered/`

### 3. Macroeconomic Model (Zsiday Viktor)

Alternative forecast using net real salary change:
```python
model = LinearRegression().fit(salary_index, governing_party_vote_share)
prediction_2026 = model.predict(salary_2026)
```

## File Reference

### `data.py`

**Key functions**:
- `load_data()`: Loads all CSV/Excel files into DataFrames
- `aggregate_to_oevk()`: Aggregates results to OEVK district level
- `categorize_party_result()`, `categorize_party_polls()`: Bins parties into 6 categories
- `transform_wide_to_long()`: Converts wide election data to long format
- `clean_candidates()`, `create_incumbent_dummy()`: Candidate-level features

### `model.py`

**Key functions**:
- `pollster_bias()`, `pollster_quality()`, `pollster_sigma()`: Compute pollster metrics
- `polling_avg()`: Bias-corrected, quality-weighted polling average
- `forecast_distr()`: Build Dirichlet forecast distributions
- `correl_parties()`: Sample correlated vote shares from Dirichlet
- `swing_coef()`: Compute district swing coefficients
- `OEVK_projection()`: Project national draw to district level
- `calibrate_sigma_d()`: Estimate district noise via backtesting
- `simulation()`: Run full 1000-iteration Monte Carlo simulation
- `backtesting()`: Validate on 2022 election

### `seat_allocation.py`

**Key function**:
- `seat_simulated()`: Allocate 106 SMD + 93 list seats under Hungarian system
  - Enforces 5% threshold on raw national list votes
  - Computes effective list votes (list + SMD fragments)
  - Uses d'Hondt method for proportional allocation

## Contributing & Citation

If using this model for academic work, please cite:

> [Lilla Szilvia Polgár] (2026). "Forecasting the 2026 Hungarian Parliamentary Elections Using Machine Learning Models." Master's Thesis, [Corvinus University of Budapest].


---

**Last updated**: [2022.04.12.]  
**Model version**: 1.0  
**Python version**: 3.9+