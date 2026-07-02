# Rainfall: What the Data Shows

> Last updated: 2026-07-02

Weather affects travel, so daily rainfall for each Malaysian state (2019–2025) is
part of the dataset. This page explains the rainfall patterns and why the data is
kept at state level rather than a single national figure. Each section uses
**Main idea**, **Evidence**, **Analysis**, and **Link**.

---

## Pattern 1 — Rainfall Is Strongly Seasonal

**Main idea.** Heavy rain concentrates in the Northeast Monsoon season, roughly
November to January.

**Evidence.** Wet-day ratios and counts of extreme days (over 50 mm) peak in
November–January. The overall distribution is heavily skewed, with monthly totals
ranging from around 30 mm in dry spells to over 600 mm at monsoon peaks.

**Analysis.** Because the heavy-rain season lines up with the calendar, the model can
connect rainfall to the time of year through its month sine/cosine features. The
seasonal signal is real and repeats, which makes it learnable.

**Link.** This is why weather is modelled alongside calendar features rather than in
isolation.

---

## Pattern 2 — The East Coast Is Much Wetter Than the West

**Main idea.** East-coast states get far more extreme rainfall than west-coast states.

**Evidence.** Kelantan, Terengganu, and Pahang record 8–12 extreme-rain days in the
Q4–Q1 monsoon window, versus 2–4 for Selangor and Kuala Lumpur. West-coast states
show a milder, two-season pattern.

**Analysis.** A single national rainfall average would hide this east–west split. The
same day can be a flood in Kota Bharu and dry in KL, and those two places have very
different transit demand responses. Keeping all 15 state columns lets the model learn
location-specific effects.

**Link.** Retaining state-level detail matters because weather's impact on ridership
varies by region and mode (Ngo & Bashar, 2024).

---

## Pattern 3 — Filling Small Gaps Without Overreaching

**Main idea.** The raw rainfall series has occasional missing days that must be filled
carefully.

**Evidence.** Before cleaning, joining the rainfall data to a daily calendar leaves
gaps. Interior gaps are filled by linear interpolation, but only up to seven days;
gaps at the very start or end are filled from the nearest known value.

**Analysis.** The seven-day cap is deliberate. A typical monsoon event lasts a few
days, so seven days is long enough to bridge one event but short enough not to blur
two separate events into one. Over-filling would invent weather that never happened.

**Link.** This matches the event durations seen in the data and keeps the seasonal
peaks intact.

---

## What Goes Into the Model

The pipeline uses 15 daily state-level rainfall columns (`rainfall_mm__MY01` through
`MY17`, excluding the two federal-territory codes merged into neighbouring states).
Accumulation and anomaly columns are cleaned and stored but left out of the model
input to keep the feature count manageable.

---

## A Note on the Rain–Ridership Relationship

The link between rain and ridership is not one-directional. Light rain slightly
discourages travel, but very heavy rain can *increase* transit use as people avoid
driving in dangerous conditions. Keeping rainfall as a continuous value (rather than
a simple wet/dry flag) lets the model represent both sides of this response
(Chen et al., 2022; Jiang & Cai, 2023).

---

## References

Chen, J., Zhou, Z., Li, S., & Shi, W. (2022). Spatiotemporal variations in Shanghai metro commuting flows during rainfall events. *Weather, Climate, and Society, 14*(3), 785–799. https://doi.org/10.1175/WCAS-D-21-0167.1

Jiang, S., & Cai, C. (2023). The impacts of weather conditions on metro ridership: An empirical study from three mega cities in China. *Travel Behaviour and Society, 31*, 200–210. https://doi.org/10.1016/j.tbs.2022.12.003

Ngo, N. S., & Bashar, B. (2024). The impacts of extreme weather events on U.S. public transit ridership. *Transportation Research Part D: Transport and Environment, 137*, 104504. https://doi.org/10.1016/j.trd.2024.104504
