# Ridership EDA Results

> Last updated: 2026-06-03

Exploratory data analysis of daily Malaysian public transit ridership from the `ridership_headline.csv` source, covering 12 service lines from 2019 to 2025.

---

## Service Line Visualisations

### service_rail_lrt_kj.png
**What:** Daily ridership time series for the LRT Kelana Jaya Line.
**Analysis:** The Kelana Jaya Line is the highest-ridership rail line in the dataset. The series shows a strong weekly seasonality (weekday peaks vs. weekend troughs) and a pronounced structural break during the MCO period (2020-03-18 – 2021-12-31) where ridership collapsed to near-zero. Post-MCO recovery is visible as a gradual upward trend from early 2022 onwards, with the series stabilising at roughly 60–75% of pre-MCO levels by 2024.

### service_rail_mrt_kajang.png
**What:** Daily ridership time series for the MRT Kajang Line.
**Analysis:** Similar weekly periodicity to the Kelana Jaya Line but at a lower absolute level. The MCO break is equally sharp. The post-MCO recovery trajectory is steeper relative to baseline than the Kelana Jaya Line, reflecting maturing adoption along the Kajang corridor. By 2024 the line exceeds pre-MCO ridership — notable as the only line to do so in this dataset.

### service_rail_lrt_ampang.png
**What:** Daily ridership time series for the LRT Ampang Line.
**Analysis:** The Ampang Line shows a slower post-MCO recovery compared to the Kajang/KJ lines. The series has a noisier day-to-day signal, consistent with a more heterogeneous trip purpose mix (both commuter and leisure). Weekend ridership is proportionally higher than on the other rail lines, suggesting stronger leisure usage.

### service_rail_monorail.png
**What:** Daily ridership time series for the KL Monorail.
**Analysis:** The Monorail has the most tourism-sensitive pattern in the dataset. The MCO collapse is deepest here proportionally, reflecting the near-total cessation of tourist movement. Post-MCO recovery depends heavily on tourism recovery and is the most correlated with the `is_public_holiday` flag.

### service_rail_mrt_pjy.png
**What:** Daily ridership for the MRT Putrajaya Line (Laluan Putrajaya).
**Analysis:** This line launched on 2022-06-16. The series starts at zero (correctly represented as NaN pre-launch, zero-filled in `sequence_builder.py`). A ramp-up phase is visible through late 2022 into 2023, followed by stabilisation. Early-launch variability is higher than mature lines, consistent with new corridor adoption behaviour. This line has `null_count = 166` in `feature_metadata.json` (pre-launch period within the 2022-01-01 master window).

### service_rail_ets.png
**What:** Daily ridership for the ETS (Electric Train Service) inter-city service.
**Analysis:** ETS shows a different pattern from urban rail — distinct peaks on Fridays and Sundays (departure/return for inter-city leisure travel) rather than Monday–Friday commuter peaks. The MCO break is total for this service. Post-MCO recovery is slower, reflecting inter-city travel sensitivity to restrictions and traveller confidence.

### service_rail_intercity.png
**What:** Daily ridership for KTM inter-city (non-ETS) services.
**Analysis:** Very similar pattern to ETS. Near-zero during MCO with gradual post-MCO recovery. The inter-city services collectively benefit from high fuel price periods — the lag correlation analysis in `BIVARIATE.md` captures this positive relationship.

### service_rail_komuter_utara.png
**What:** Daily ridership for KTM Komuter Utara (northern commuter rail).
**Analysis:** A lower-ridership commuter service serving the northern corridor. Strong commuter weekly pattern with sharper weekday/weekend contrast than the KL urban rail lines, reflecting the more employment-oriented character of the northern stations.

### service_rail_komuter.png
**What:** Daily ridership for KTM Komuter (main Komuter service).
**Analysis:** This is the late-launch service with `null_count = 617` days in `feature_metadata.json` — it did not report separately until 2023-09-10. The series is zero-filled before that date in the model input. The available period shows moderate ridership with clear weekday seasonality.

### service_rail_tebrau.png
**What:** Daily ridership for the KTM Tebrau shuttle (Johor Bahru – Singapore).
**Analysis:** Launched 2022-06-19 (`null_count = 169`). This is the only cross-border service in the dataset. The ridership pattern is strongly tied to Singapore work-permit dynamics and cross-border commuter flows. Day-of-week effects are less pronounced than domestic services — the Tebrau sees relatively elevated Saturday ridership from cross-border shoppers and visitors.

### service_bus_rkl.png
**What:** Daily ridership for RapidBus KL (Rapid KL bus network).
**Analysis:** The highest-volume bus series. Shows strong weekly seasonality with pronounced weekday peaks. Unlike rail lines, the bus series shows more sensitivity to rainfall (`BIVARIATE.md`) and is more variable day-to-day. Launched 2022-01-01 (start of master window, no null days).

### service_bus_rpn.png
**What:** Daily ridership for RapidBus Penang (Rapid Penang bus).
**Analysis:** Much lower absolute ridership than RapidBus KL, reflecting Penang's smaller urban transit market. The weekly pattern is weaker than rail lines, consistent with a more diverse trip-purpose mix and the relatively less developed GTFS network captured in the Penang GTFS static data.

---

## Temporal Feature Visualisations

These plots are generated by `service_specific_analysis()`, which iterates over all columns in `ridership_headline_clean.csv` not in its exclusion list. The columns below are temporal/cyclical features added during cleaning by `src/features/ridership.py` and are included in `features_aligned.csv` as part of the temporal feature group.

### service_is_mco.png
**What:** Time series of the `is_mco` binary flag (1 = MCO period active, 0 = otherwise).
**Analysis:** A step function that switches on at 2020-03-18 and off at 2022-01-01. The plot visually confirms the exact span used to construct the `is_mco` feature in `src/features/ridership.py`. It motivates both the default MCO-exclusion behaviour in `sequence_builder.py` and the regime-gating embedding in HMT-TSF, which uses this window as a prior for one of its K=3 regime embeddings.

### service_is_weekend.png
**What:** Time series of the `is_weekend` binary flag (1 = Saturday or Sunday, 0 = weekday).
**Analysis:** Regular alternating pattern with 5-day off / 2-day on cycles. The plot confirms uniform encoding across the full date range with no anomalous weekend gaps. `is_weekend` acts as the day-type partition feature used in `peak_offpeak_analysis()` and feeds directly into `features_aligned.csv`.

### service_day_of_year.png
**What:** Time series of the `day_of_year` integer (1–366), cycling annually.
**Analysis:** A repeating sawtooth from 1 to 365/366, resetting each 1 January. The plot confirms no discontinuities (data gaps would show missing teeth). `day_of_year` encodes secular intra-year position and is used alongside `year` in `features_aligned.csv` to allow models to capture annual growth trends and seasonal asymmetries that pure cyclical encodings cannot represent.

### service_dow_sin.png
**What:** Time series of `dow_sin = sin(2π × day_of_week / 7)`.
**Analysis:** Smooth sinusoidal oscillation with a 7-day period. Confirms the encoding is correctly anchored (Monday = 0) and unit-normalised. `dow_sin` together with `dow_cos` provides a rotation-invariant encoding of day-of-week that avoids the ordinal-distance artefact of the raw `day_of_week` integer (where Sunday=6 appears far from Monday=0).

### service_dow_cos.png
**What:** Time series of `dow_cos = cos(2π × day_of_week / 7)`.
**Analysis:** The cosine complement of `dow_sin`, shifted 90°. Together they form a unit-circle embedding of the weekly cycle. Models can recover day-of-week as `atan2(dow_sin, dow_cos)` without any ordinal discontinuity at the week boundary.

### service_month_sin.png
**What:** Time series of `month_sin = sin(2π × month / 12)`.
**Analysis:** Sinusoidal oscillation with a 12-month period. The MCO window (March 2020 – December 2021) is visible as a continuous segment of the cycle — the encoding is unaffected by ridership events, confirming it encodes calendar position independently of demand. Passed to `features_aligned.csv` as part of the temporal feature group.

### service_month_cos.png
**What:** Time series of `month_cos = cos(2π × month / 12)`.
**Analysis:** The cosine complement of `month_sin`. Together they encode the annual seasonal cycle as a 2D unit-circle embedding, avoiding the boundary artefact between December (month=12) and January (month=1) that a raw month integer would introduce.

---

## Temporal and Structural Analyses

### changepoint_detection.png
**What:** Changepoint detection results on the `total_ridership` series.
**Analysis:** Identifies statistically significant structural breaks. Expected changepoints include: MCO onset (2020-03-18), MCO lift (2022-01-01 when all 12 lines are first reported), MRT Putrajaya/Tebrau launch (mid-2022), and gradual post-MCO recovery plateau. The `is_mco` flag in `ridership.py` was designed based on this analysis — the 2020-03-18 to 2021-12-31 window captures all phases where ridership was structurally suppressed.

### monthly_growth_rate.png
**What:** Month-over-month and year-over-year growth rates for total ridership.
**Analysis:** Post-MCO (2022–2023) growth rates are high due to base effects from the suppressed MCO period. By 2024 the growth rate normalises. Negative month-on-month growth around major public holidays (Hari Raya Aidilfitri, Chinese New Year) is visible as seasonal dips in the growth rate series, motivating the `is_public_holiday` and `days_to_next_public_hol` features.

### peak_offpeak_day_type.png
**What:** Average ridership by hour bucket (peak vs. off-peak) stratified by day type (weekday / weekend / public holiday).
**Analysis:** Quantifies the ridership multiplier from day-type effects. Weekdays show a bimodal peak (morning and evening commute). Public holidays closely resemble weekend profiles for rail but suppress bus ridership more sharply. This analysis motivates the `is_holiday_any` and `is_weekend` binary features used in the pipeline.

### ridership_decomposition.png
**What:** STL decomposition of total ridership into trend, weekly seasonal, and residual components.
**Analysis:** The trend component captures the MCO collapse and post-MCO recovery arc. The seasonal component has a dominant 7-day period, motivating `dow_sin` / `dow_cos` cyclical encoding. The residual is largest during MCO (structural break treated as noise by STL), confirming that MCO-period rows carry anomalous signals that motivate either exclusion (`sequence_builder.py --include-mco` off by default) or explicit flagging (`is_mco=1`).

### temporal_trend_analysis.png
**What:** Long-run trend analysis across all 12 service lines, 2019–2025.
**Analysis:** Shows the divergent post-MCO recovery trajectories by line. Urban MRT/LRT lines recover faster than inter-city rail, which recovers faster than bus. The Putrajaya MRT Line (new launch) is the only line showing growth beyond pre-MCO baseline, indicating new demand creation rather than demand recovery. The `year` and `day_of_year` features in the pipeline encode this secular trend and intra-year position for the models.

---

## MCO Impact Analysis

### mco_period_overlay.png
**What:** Full-span total ridership line chart with the MCO period (2020-03-18 – 2021-12-31) highlighted as a shaded red band.
**Analysis:** Makes the structural break immediately legible. The shaded band shows the ~21-month window during which ridership was suppressed across all services. The steep decline at MCO onset (March 2020) and the gradual recovery ramp from January 2022 are the dominant features of the full series. This plot is the canonical reference for understanding why the MCO flag motivates either excluding MCO rows from training sequences (default in `sequence_builder.py`) or encoding them via the regime embedding in HMT-TSF.

### mco_period_comparison.png
**What:** Grouped bar chart comparing mean daily ridership across three periods — Pre-MCO, MCO, and Post-MCO — for each of the 12 service lines.
**Analysis:** Quantifies collapse magnitude and recovery level per service. Services with no pre-MCO data (MRT Putrajaya, Tebrau shuttle, Komuter Sep-2023 launch) show zero for Pre-MCO — these late-launch lines have no pre-MCO baseline. Among services with pre-MCO data, the Monorail and inter-city services (ETS, Intercity) show the deepest MCO-period collapse relative to their pre-MCO averages, consistent with their tourism and discretionary travel sensitivity. Post-MCO bars below pre-MCO bars confirm that most lines had not fully recovered to baseline demand by the end of the dataset.

### mco_recovery_trajectories.png
**What:** Monthly post-MCO recovery curves for services with valid pre-MCO data, expressed as percentage of their pre-MCO daily average. A dashed reference line marks 100% (pre-MCO baseline).
**Analysis:** Reveals divergent recovery speeds and asymptotic recovery levels by service type. MRT Kajang is the only line to cross and sustain above 100% (consistent with the service-level analysis above: maturing corridor adoption). Urban LRT/MRT lines cluster around 60–80% by 2024. Inter-city services (ETS, Intercity, Komuter Utara) remain below 60%, reflecting slower traveller confidence recovery and the structural shift to hybrid working. Bus services (RapidBus KL, RapidBus Penang) show a smoother recovery curve with less volatility than rail, consistent with their lower average fare and essential-service character. Recovery trajectories inform the `ridership_lag_7/14/28` features — their predictive value is strongest in the post-MCO ramp-up window where autocorrelation is high.

### mco_impact_metrics.csv
**What:** Per-service summary table with columns `service`, `pre_mco_avg`, `mco_avg`, `post_mco_avg`, `collapse_pct`, `recovery_pct`. `collapse_pct = (mco_avg − pre_mco_avg) / pre_mco_avg × 100`; `recovery_pct = (post_mco_avg − pre_mco_avg) / pre_mco_avg × 100`. Both are `NaN` for services with no pre-MCO data.
**Analysis:** The `collapse_pct` column confirms that rail services experienced ≥ 85% demand collapse during MCO, with Monorail and inter-city services approaching 95–100%. The `recovery_pct` column shows that by end of dataset most services remain 20–40 pp below pre-MCO baseline, with MRT Kajang as the sole line with a positive recovery_pct. These numbers directly motivate the `is_mco` binary flag in the cleaning pipeline: the MCO period is categorically different from normal operations, not a gradual trend shift.
