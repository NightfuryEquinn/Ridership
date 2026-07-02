# The Big Picture: Network-Wide Themes

> Last updated: 2026-07-02

This page pulls together the whole picture — how ridership, geography, and context fit
together across the network. It sits above the single-source pages and explains the
system-level story. Each section uses **Main idea**, **Evidence**, **Analysis**, and
**Link**.

---

## Theme 1 — Rail Leads the Network

**Main idea.** The Malaysian transit network is rail-dominated, and its recovery is
rail-led.

**Evidence.** Urban rail (LRT/MRT/Monorail) accounts for roughly 55–65% of total
ridership, bus for 25–30%, and inter-city rail for the rest. The two bus services are
each smaller than any single rail line, and the busiest days are rail days.

**Analysis.** Because one aggregate number would be dominated by rail and hide bus
dynamics, the project keeps all 12 services as separate values while forecasting the
combined total as the main target. This preserves the ability to see each mode's
behaviour.

**Link.** Modelling services separately avoids masking the smaller but distinct bus
patterns (Yang et al., 2023).

---

## Theme 2 — People, Places, and Stops Line Up

**Main idea.** Population, nearby destinations, and transit stops all concentrate in
the same places.

**Evidence.** Overlaying population density, points of interest, and transit stops for
the Klang Valley shows strong co-location: dense neighbourhoods (Ampang, Chow Kit,
Bangsar) have both many destinations and several stops within walking distance.

**Analysis.** This confirms the population and POI features encode real demand signals,
not noise — the places with the most people and activity are the places with the most
transit. It validates building these features around stop catchments.

**Link.** The spatial features carry genuine demand information that the models can use
(Li et al., 2024).

---

## Theme 3 — The COVID Break Dominates the Long History

**Main idea.** The lockdown period is the defining event in the full 2019–2025 record.

**Evidence.** Ridership before, during, and after the lockdown shows a deep drop and a
partial, uneven recovery — most services had not returned to pre-COVID levels by 2025.

**Analysis.** This one event changes the statistical behaviour of the data so much that
the project treats the pre-COVID, lockdown, and post-COVID periods as distinct
regimes. The proposed model learns to recognise which regime it is in without being
told the date, which is what makes it robust when the break is included.

**Link.** Handling this structural break is the central design challenge the modelling
work addresses (Lee et al., 2024).

---

## Theme 4 — Calendar and Weather Are the Everyday Drivers

**Main idea.** Day-to-day, ridership is driven mostly by the calendar, with weather and
fuel as secondary influences.

**Evidence.** The weekly cycle, holidays, and seasonal position explain most of the
routine variation; rainfall and fuel prices add smaller, context-specific effects.

**Analysis.** This ordering guides the feature design: calendar features are central,
while weather and fuel are supporting signals that matter mainly at specific times
(monsoon season, price revisions). The model's own importance analysis later confirms
this ranking.

**Link.** A multi-source view of demand drivers is consistent with recent urban-rail
forecasting work (Cui et al., 2025).

---

## References

Cui, H., Si, B., Chi, D., Li, Y., Li, G., & Chen, Y. (2025). Short-term passenger flow prediction for urban rail systems: A deep learning approach utilizing multi-source big data. *PLOS ONE, 20*(1), e0333094. https://doi.org/10.1371/journal.pone.0333094

Lee, S., Kim, J., & Cho, K. (2024). Temporal dynamics of public transportation ridership in Seoul before, during, and after COVID-19 from urban resilience perspective. *Scientific Reports, 14*, 9078. https://doi.org/10.1038/s41598-024-59323-w

Li, Y., Zhang, Q., & Wang, H. (2024). An efficient approach for identifying potential bus passenger demand based on multisource data. *Journal of Advanced Transportation, 2024*, 5368577. https://doi.org/10.1155/2024/5368577

Yang, C., Yu, C., Dong, W., & Yuan, Q. (2023). Substitutes or complements? Examining effects of urban rail transit on bus ridership using longitudinal city-level data. *Transportation Research Part A: Policy and Practice, 174*, 103489. https://doi.org/10.1016/j.tra.2023.103489
