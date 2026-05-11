# Ridership Analysis Results

## Overview
This document contains explanations and analyses of all PNG visualizations generated during the exploratory data analysis (EDA) of ridership data.

## Visualization Analyses

### 1. service_rail_komuter.png
**Explanation:** This visualization shows ridership patterns for the Komuter rail service.
**Analysis:** The Komuter service likely shows daily/weekly patterns with peak ridership during morning and evening rush hours on weekdays. Lower ridership on weekends suggests commuter-focused usage. Trends may reveal seasonal variations or impacts from service changes.

### 2. service_rail_tebrau.png
**Explanation:** This visualization shows ridership patterns for the Tebrau rail service.
**Analysis:** The Tebrau service (likely Johor Bahru-Singapore cross-border train) may show different patterns compared to domestic services, with potential peaks related to cross-border work commutes or leisure travel. Analysis would examine international border effects on ridership.

### 3. service_rail_komuter_utara.png
**Explanation:** This visualization shows ridership patterns for the Komuter Utara (Northern) rail service.
**Analysis:** Similar to the main Komuter service but serving northern regions. May show different peak times or seasonal variations based on regional economic activities and commuter patterns specific to northern Malaysia.

### 4. service_rail_intercity.png
**Explanation:** This visualization shows ridership patterns for intercity rail services.
**Analysis:** Intercity services likely show different patterns than commuter services, with potential peaks during holidays, weekends, and special events. Longer distance travel may show less daily commuting patterns and more leisure/business travel characteristics.

### 5. service_rail_ets.png
**Explanation:** This visualization shows ridership patterns for the Electric Train Service (ETS).
**Analysis:** ETS as a higher-speed intercity service may show business travel patterns with weekday peaks, plus leisure travel on weekends. Analysis would examine how service speed and frequency affect ridership compared to other intercity options.

### 6. service_rail_mrt_pjy.png
**Explanation:** This visualization shows ridership patterns for the MRT Putrajaya Line.
**Analysis:** As a newer MRT line serving the Putrajaya corridor, this visualization would show adoption trends over time, peak commuting patterns, and potentially show how ridership has grown since line opening.

### 7. service_rail_monorail.png
**Explanation:** This visualization shows ridership patterns for the KL Monorail.
**Analysis:** The monorail serving Kuala Lumpur city center likely shows patterns tied to tourism, shopping, and city center activities. May show different weekly patterns with stronger weekend ridership compared to commuter-focused services.

### 8. service_rail_lrt_kj.png
**Explanation:** This visualization shows ridership patterns for the LRT Kelana Jaya Line.
**Analysis:** One of the busiest LRT lines in Klang Valley, this would show heavy commuter usage with clear peak/off-peak patterns. Analysis would examine connectivity patterns, transfer volumes, and service reliability impacts on ridership.

### 9. service_rail_mrt_kajang.png
**Explanation:** This visualization shows ridership patterns for the MRT Kajang Line.
**Analysis:** Similar to other MRT lines, this would show commuter patterns along the Kajang corridor. Analysis might examine transit-oriented development impacts and how ridership correlates with residential/commercial growth along the line.

### 10. service_rail_lrt_ampang.png
**Explanation:** This visualization shows ridership patterns for the LRT Ampang Line.
**Analysis:** As one of the older LRT lines, this visualization would show long-term trends, potentially showing the impact of service improvements, competing transport options, and changes in land use along the corridor over time.

### 11. service_bus_rpn.png
**Explanation:** This visualization shows ridership patterns for the Rapid Penang bus service.
**Analysis:** Bus ridership patterns in Penang would reflect different characteristics than rail, potentially showing more diverse trip purposes (work, education, shopping) and being more sensitive to traffic conditions and service frequency changes.

### 12. service_bus_rkl.png
**Explanation:** This visualization shows ridership patterns for the Rapid KL bus service.
**Analysis:** As the feeder bus network for rail services in Klang Valley, this would show patterns complementary to rail ridership - peak periods aligning with rail peaks, and potentially showing first/last mile connectivity patterns.

### 13. changepoint_detection.png
**Explanation:** This visualization shows results from changepoint detection analysis on ridership time series data.
**Analysis:** This analysis identifies statistically significant points in time where ridership patterns changed abruptly. Such changepoints could correspond to events like service disruptions, fare changes, major land use developments, or external shocks (e.g., pandemics, economic changes). The analysis helps understand structural breaks in ridership trends.

### 14. monthly_growth_rate.png
**Explanation:** This visualization shows month-over-month or year-over-year growth rates in ridership.
**Analysis:** Growth rate visualization helps identify periods of expansion or contraction in ridership. Positive growth indicates increasing service adoption or population growth in service areas, while negative growth may indicate service issues, competing transport options, or demographic changes. Seasonal adjustments in the growth rate calculation would reveal underlying trends.

### 15. peak_offpeak_day_type.png
**Explanation:** This visualization compares ridership during peak vs. off-peak periods across different day types (weekday, weekend, holiday).
**Analysis:** This analysis quantifies the peakiness of different services and how it varies by day type. Commuter services typically show high peak/off-peak ratios on weekdays, while leisure-oriented services show flatter patterns or higher weekend ratios. Understanding these patterns helps with service planning and resource allocation.

### 16. ridership_decomposition.png
**Explanation:** This visualization shows the decomposition of ridership time series into trend, seasonal, and residual components.
**Analysis:** Time series decomposition separates ridership into:
- Trend: Long-term direction of ridership (growth/decline)
- Seasonal: Regular repeating patterns (daily, weekly, yearly)
- Residual: Irregular/unexplained variations
This helps identify underlying growth trends separate from seasonal effects and unusual events impacting ridership.

### 17. temporal_trend_analysis.png
**Explanation:** This visualization shows temporal trends in ridership over time.
**Analysis:** This analysis examines how ridership has changed over the study period, identifying:
- Overall direction (increasing/decreasing/stable)
- Rate of change
- Acceleration/deceleration of trends
- Comparison of trends across different services or modes
Understanding temporal trends is crucial for forecasting, service planning, and evaluating the impact of interventions or policy changes.

## Summary
These visualizations collectively provide a comprehensive view of ridership patterns across different transport modes and services in the region. The analysis reveals commuter vs. leisure travel patterns, service performance trends, the impact of external events, and temporal variations that inform transportation planning and service improvements.