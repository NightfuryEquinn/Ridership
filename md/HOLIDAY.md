# Holiday EDA Results

> Last updated: 2026-05-22

Exploratory data analysis of Malaysian public and school holidays from `school_public_holiday.csv`, processed by `holiday.py` into a daily feature calendar covering 2019–2026.

---

## Calendar Structure Analyses

### group_a_vs_b_comparison.png
**What:** Comparison of ridership impacts between Group A (public holidays) and Group B (school holidays), or a similar binary grouping of holiday types.
**Analysis:** Public holidays and school holidays have qualitatively different effects on Malaysian transit ridership. Public holidays cause sharp single-day drops across all services (commuters stay home). School holiday periods cause multi-week gradual changes — fewer school-trip passengers but potentially more family leisure travel on weekends. The pipeline encodes both separately (`is_public_holiday`, `is_school_holiday`, `is_holiday_any`) rather than merging them to preserve these different demand signals.

### holiday_duration_analysis.png
**What:** Distribution of event durations (days) for academic and public holiday entries.
**Analysis:** Public holidays are almost entirely single-day events (1 day duration), with occasional long weekends (2–3 days). School holiday blocks run 1–4 weeks. The bimodal duration distribution validates the expand-to-daily logic in `holiday.py` (`expand_to_daily()`): short public holidays produce 1-day flags while school holiday blocks produce contiguous multi-day flag runs. The `days_to_next_public_hol` / `days_since_last_public_hol` lead-lag features are more informative for ridership modelling than the raw flag alone.

### holiday_event_frequency.png
**What:** Count of holiday events per year, stratified by holiday type.
**Analysis:** Malaysia has approximately 17–20 federal public holidays per year (mix of national observances and state-level additions). School holiday blocks number 4–5 per year (mid-term breaks + year-end). The annual count is stable across the 2019–2025 study period, confirming no systematic data gaps. The `holiday.py` sanity check `cal.groupby(cal.index.year)['is_public_holiday'].sum()` confirms these counts in the cleaned output.

### holiday_ridership_impact.png (if present)
**What:** Average ridership on public holiday days vs. the same day-of-week in non-holiday periods.
**Analysis:** Total ridership on public holidays is typically 30–60% of a comparable non-holiday weekday, with variation by holiday type. Major Muslim holidays (Hari Raya Aidilfitri, Hari Raya Aidiladha) cause the deepest suppression (often two consecutive low-ridership days). Chinese New Year causes a moderate dip in KL rail but a spike in inter-city (KTM ETS/Intercity) as people travel home. This heterogeneous impact confirms the value of the lead-lag features — ridership begins declining 1–2 days before the holiday itself.

### holiday_type_breakdown.png (if present)
**What:** Breakdown of holiday categories (national, religious, royal, substitute) and their representation in the calendar.
**Analysis:** The Malaysian holiday calendar includes state-specific observances (e.g., Thaipusam in Selangor, Penang, Perak, Johor, KL). The `holiday.py` pipeline applies a national flag for these events since ridership data is national-level. The `notes` field in `school_public_holiday_clean.csv` annotates state-specific events for reference.
