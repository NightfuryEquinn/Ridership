# Bivariate Analysis Report

Pairwise relationship analyses between the feature sources and the ridership targets in this Malaysian transit forecasting project. All analyses use the post-MCO data window (2022-01-01 – 2025-12-31) unless otherwise noted.

---

## Fuel Price × Ridership

### fuel_ridership_correlation.png
**What:** Scatter plot of weekly-averaged RON95 fuel price vs. weekly total ridership, with a linear or polynomial regression overlay.
**Analysis:** The relationship between fuel prices and transit ridership is expected to be mildly positive in Malaysia: higher RON95 prices increase the relative attractiveness of public transit over private vehicles. However, because RON95 is government price-controlled (held flat for months then updated in a single step), variation is limited and the correlation is modest. RON97 (market-priced) shows stronger correlation. The regression slope quantifies the demand elasticity — relevant context for interpreting why fuel price features are included in the pipeline alongside ridership lags.

### fuel_ridership_lagged_effects.png
**What:** Cross-correlation function (CCF) between weekly fuel price changes and total ridership at lags 0–8 weeks.
**Analysis:** Immediate ridership response to fuel price changes (lag 0) is expected to be weak — commuters with fixed transit habits do not respond overnight to a weekly fuel price announcement. A modest lagged effect at 2–4 weeks may be present, reflecting gradual mode-shift decisions. This analysis motivates the `fp_chg_ron95` / `fp_chg_ron97` / `fp_chg_diesel` weekly-change features (not just levels) as predictors of future ridership shifts.

### fuel_ridership_mode_substitution.png
**What:** Separate fuel-ridership scatter plots for urban rail, bus, and inter-city rail, showing mode-level heterogeneity in fuel price sensitivity.
**Analysis:** Inter-city rail (KTM ETS, Intercity) is expected to show a stronger positive fuel-ridership correlation than urban rail, because inter-city travellers have clearer car substitution options. Urban rail ridership in the Klang Valley is less price-sensitive as it is a habitual commuting mode. Bus ridership may show the weakest correlation, reflecting the lower-income passenger base with fewer modal alternatives.

### fuel_ridership_scatter_regression.png
**What:** Scatter plot of monthly average RON95/diesel price vs. monthly total ridership with OLS regression line and R² statistic.
**Analysis:** The R² from a simple bivariate OLS is expected to be low (<0.3), confirming that fuel price alone is an insufficient predictor of ridership. The residuals from this regression exhibit strong seasonal and weekly patterns, validating the need for temporal features (holiday, DOW, month) alongside fuel price in the full 79-feature pipeline.

### fuel_ridership_threshold.png
**What:** Non-linear response plot (e.g., piecewise regression or LOESS) examining whether ridership sensitivity to fuel price changes at specific RM/litre thresholds.
**Analysis:** Malaysia's fuel subsidy system introduces discrete price levels rather than continuous variation (prices jump in fixed steps, then freeze). The threshold analysis tests whether ridership jumps more at price announcement dates than at other dates — relevant for calibrating the `fp_chg_*` features as change-signal predictors rather than relying solely on level features.

---

## GTFS × Ridership

### gtfs_ridership_frequency.png
**What:** Scatter of average service headway (minutes, from `gtfs_avg_segment_s`) vs. ridership per route or per stop.
**Analysis:** Shorter headways (higher frequency service) generally correlate with higher ridership. The relationship is expected to be log-linear rather than linear — diminishing returns beyond 5-minute headways. This validates the inclusion of `gtfs_avg_segment_s` as a transit network capacity proxy in the static feature set.

### gtfs_ridership_stops.png
**What:** Scatter of number of GTFS stops per operator vs. reported ridership by mode.
**Analysis:** Rail operators with fewer but larger-capacity stops (Rapid Rail KL) outperform bus operators (more stops, lower capacity). This cross-modal comparison demonstrates that `gtfs_n_stops` and `gtfs_n_routes` encode network scale, not network quality — consistent with their use as static context scalars rather than primary predictors.

---

## Holiday × Ridership

### holiday_ridership_overall_impact.png
**What:** Box plot or violin of total ridership on public holiday days vs. non-holiday weekdays vs. weekends.
**Analysis:** Public holidays produce ridership distributions shifted ~40–60% below non-holiday weekdays. The spread on public holiday days is also narrower (less day-to-day variability) because the suppression effect dominates. This confirms that `is_public_holiday` and `is_holiday_any` carry significant signal and that the lead-lag features (`days_to_next_public_hol`) are necessary to capture the pre-holiday run-up suppression visible in the 1–2 days before major holidays.

### holiday_ridership_pre_post_profile.png
**What:** Average ridership profile for the 7-day window before and after public holidays, aggregated across all public holiday events in the dataset.
**Analysis:** A gradual decline in ridership begins 1–2 days before public holidays, bottoms on the holiday itself, and recovers over 1–2 days after. Hari Raya Aidilfitri shows the deepest pre-holiday decline (multi-day travel window as workers return to hometowns). This asymmetric lead-lag pattern is captured by `days_to_next_public_hol` (pre-holiday) and `days_since_last_public_hol` (post-holiday) features.

### holiday_ridership_type_analysis.png
**What:** Per-holiday-type ridership impact (Hari Raya, Chinese New Year, Deepavali, National Day, etc.) as percentage deviation from non-holiday baseline.
**Analysis:** Hari Raya Aidilfitri causes the deepest suppression in KL urban rail (workers leave the city). Chinese New Year causes moderate suppression in urban rail but a spike in KTM inter-city (long-distance travel). National Day / Malaysia Day show mild suppression. These type-specific effects are implicitly learned by the models from the `is_public_holiday` flag together with the `month_sin`/`month_cos` features that correlate with holiday timing.

---

## OSM × Population × Ridership

### population_ridership_correlation.png
**What:** Scatter of per-state median population density vs. total ridership for states with transit coverage.
**Analysis:** Higher-density states (Selangor, Kuala Lumpur) account for the dominant share of ridership, confirming that the `pop_density_median` static feature encodes a meaningful demand proxy. The correlation is not 1:1 because transit availability (GTFS coverage) is also a strong moderating variable — Sabah and Sarawak have moderate populations but limited ridership due to lower transit network density.

---

## Rainfall × Ridership

### rainfall_ridership_correlation.png
**What:** Scatter of daily total rainfall (sum across all 15 state columns) vs. total ridership.
**Analysis:** The expected relationship is weakly negative for bus ridership (rain discourages walking to stops) and near-zero for rail ridership (covered stations, habitual commuters unaffected). The aggregate correlation is weak because rainfall varies across Malaysia's 15 states simultaneously and the total ridership aggregates services in different regions. Per-state rainfall–ridership analysis (not shown) likely reveals stronger state-specific effects, particularly for RapidBus services.

### rainfall_ridership_weekday_weekend.png
**What:** Separate rainfall–ridership scatter plots for weekdays and weekends.
**Analysis:** Weekend ridership is more sensitive to rainfall than weekday ridership because weekend trips are more discretionary. This interaction (rainfall × day_type) is implicitly learned by the models from the combination of per-state `rainfall_mm__MY*` features and `is_weekend`/`dow_sin`/`dow_cos` features.
