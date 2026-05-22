# Fuel Price EDA Results

> Last updated: 2026-05-22

Exploratory data analysis of Malaysian retail fuel prices from `fuelprice.csv`, covering RON95, RON97, diesel, East Malaysia diesel, and subsidy-programme variants (BUDI Madani RON95, SKPS RON95) from 2019 onwards.

---

## Autocorrelation Function (ACF) Analysis

### acf_diesel.png
**What:** ACF of daily diesel prices (after forward-fill upsampling from weekly announcements).
**Analysis:** Diesel prices in Malaysia are announced weekly by the government and held constant between announcements, producing a step-function series. The ACF decays very slowly (near unit-root behaviour) because prices persist for exactly 7 days before each update. The `fuelprice.py` pipeline captures this by forward-filling weekly prices onto a daily index — the ACF confirms no information is lost between announcement days.

### acf_diesel_eastmsia.png
**What:** ACF of daily diesel prices for East Malaysia (Sabah/Sarawak/Labuan).
**Analysis:** East Malaysia diesel has historically been priced differently to Peninsular Malaysia due to logistics costs and regional subsidies. The ACF structure mirrors the national diesel series (weekly step function) but the levels are consistently different, motivating the inclusion of `fp_lv_diesel_eastmsia` as a separate feature rather than using the national price alone.

### acf_ron95.png
**What:** ACF of daily RON95 prices.
**Analysis:** RON95 is the most widely used fuel grade and is price-controlled below market rates as a government subsidy. Its ACF shows the same 7-day persistence structure. RON95 subsidies were subject to the BUDI Madani and SKPS targeted subsidy programmes during the study period, creating occasional structural level shifts that appear as slow decay tails beyond the weekly lag.

### acf_ron97.png
**What:** ACF of daily RON97 prices.
**Analysis:** RON97 is market-priced (no subsidy cap) and therefore shows more frequent and smaller price changes than RON95/diesel. The ACF decays faster, reflecting a less persistent price process. The contrast between RON95/diesel ACFs (slow decay, heavily subsidised) and RON97 ACF (faster decay, market-priced) validates the decision to treat them as separate features in the pipeline.

---

## Price Level Visualisations

### price_levels_comparison.png (if present)
**What:** Time series overlay of all fuel price levels (RON95, RON97, diesel, East Malaysia diesel, BUDI95, SKPS).
**Analysis:** Reveals the divergent pricing trajectories across fuel types. RON95 remains the most stable due to price controls. RON97 tracks global crude oil more closely. The introduction of the BUDI Madani (targeted RON95 subsidy for eligible vehicles) creates a permanent level bifurcation from mid-2023 onwards, which is why `fp_lv_ron95_budi95` and `fp_lv_ron95_skps` are included as separate features alongside the base `fp_lv_ron95`.

### pct_change_distribution.png (if present)
**What:** Distribution of weekly percent-changes for RON95, RON97, and diesel.
**Analysis:** RON97 has the widest percent-change distribution (market pricing), RON95 has the narrowest (subsidised with infrequent large step changes). The `ron95_pct_chg`, `ron97_pct_chg`, `diesel_pct_chg` features added in `fuelprice.py` convert the non-stationary level series into more stationary features for gradient-based optimisation in LSTM/GCN models.

---

## Weekly Change (`change_weekly`) Series

### change_weekly_distribution.png (if present)
**What:** Distribution and time series of `fp_chg_*` columns (signed weekly price changes).
**Analysis:** The change series are approximately zero-mean with occasional large positive or negative spikes corresponding to government price revisions. These series are more informative as predictors of short-term ridership responses (transit demand may shift within days of a fuel price change) than the level series alone, motivating their inclusion as separate `fp_chg_*` columns in `features_aligned.csv`.
