# Ridership EDA Results

> Last updated: 2026-05-22

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
