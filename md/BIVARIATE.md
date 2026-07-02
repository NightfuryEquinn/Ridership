# How Each Factor Relates to Ridership

> Last updated: 2026-07-02

This page looks at ridership against one factor at a time — fuel prices, service
frequency, holidays, population, and rainfall — to see which relationships are strong
enough to matter. All analyses use the post-COVID window (2022–2025). Each section
uses **Main idea**, **Evidence**, **Analysis**, and **Link**.

---

## Fuel Prices vs Ridership

**Main idea.** Higher fuel prices are linked to modestly higher transit use, but the
effect is small and delayed.

**Evidence.** The correlation between fuel prices and ridership is mildly positive but
weak — a simple fuel-only model explains less than 30% of the variation. RON97
(market-priced) correlates more strongly than RON95 (subsidised and mostly frozen).
The strongest response appears 1–3 weeks after a price change, not immediately, and
inter-city rail is more sensitive than habitual urban commuting.

**Analysis.** People do not change their commute overnight when fuel prices move, but
some gradually shift modes. That delayed, partial response is why the pipeline keeps
both price levels (for baseline context) and weekly price changes (for the short-term
signal), and why fuel is one input among many rather than a standalone predictor.

**Link.** This matches evidence that fuel-price shocks shift travel behaviour with a
lag and vary by trip type (Rahimi et al., 2024; Belloc et al., 2024).

---

## Service Frequency and Size vs Ridership

**Main idea.** More frequent, larger-capacity service is associated with higher
ridership, with diminishing returns.

**Evidence.** Shorter waits between services generally mean more riders, but the
relationship flattens once service is already frequent. Rail operators with fewer but
larger stations outperform bus operators with many small stops.

**Analysis.** This tells us the network-size features (stop counts, route counts,
average segment time) describe scale, not quality — so they are used as background
context rather than primary predictors.

**Link.** Network scale is a useful but secondary signal, consistent with how the
model weights it (Li et al., 2024).

---

## Holidays vs Ridership

**Main idea.** Public holidays sharply cut commuter ridership, and the drop starts
before the day.

**Evidence.** Public-holiday ridership is 40–60% below a normal weekday, with a
narrower spread (the suppression dominates). The dip begins 1–2 days early and
recovers over the following days. Hari Raya shows the deepest run-up; Chinese New Year
lifts inter-city rail even as it dips urban rail.

**Analysis.** This confirms the value of the "days to/from the next holiday" counters
rather than a single holiday flag — they capture the run-up and recovery that a flag
alone would miss.

**Link.** Encoding holiday timing this way is a well-established forecasting
improvement (Wu et al., 2023).

---

## Population vs Ridership

**Main idea.** Denser states generate more ridership, but only where transit exists.

**Evidence.** High-density states (Selangor, Kuala Lumpur) account for most ridership.
The link is not perfect: Sabah and Sarawak have moderate populations but limited
ridership because their transit networks are sparse.

**Analysis.** Population density is a genuine demand signal, but it only converts to
ridership where service is available. That is why population is used alongside the
network features, not on its own.

**Link.** Density and coverage jointly shape transit use (Al-Ansari & Al-Mamoori,
2022).

---

## Rainfall vs Ridership

**Main idea.** Rain's effect is weak overall and differs by mode and by day type.

**Evidence.** Rain slightly discourages bus ridership (more walking to stops) but has
little effect on rail (covered stations, habitual commuters). Weekend ridership is
more rain-sensitive than weekday ridership because weekend trips are more optional.
The national aggregate looks weak because rain varies across 15 states at once.

**Analysis.** These interactions — rain × mode, rain × weekday/weekend — are why the
pipeline keeps 15 separate state rainfall columns and combines them with weekend and
day-of-week features, letting the model learn the differences instead of assuming one
national rule.

**Link.** Weather's ridership impact is genuinely mode- and context-dependent (Jiang &
Cai, 2023).

---

## References

Al-Ansari, N., & Al-Mamoori, S. K. (2022). Do the population density and coverage rate of transit affect the public transport contribution? *Cogent Engineering, 9*(1), 2143059. https://doi.org/10.1080/23311916.2022.2143059

Belloc, I., Giménez-Nadal, J. I., & Molina, J. A. (2024). The gasoline price and the commuting behavior of US commuters: Exploring changes to green travel mode choices. *Journal of Transport Geography, 116*, 104006. https://doi.org/10.1016/j.jtrangeo.2024.104006

Jiang, S., & Cai, C. (2023). The impacts of weather conditions on metro ridership: An empirical study from three mega cities in China. *Travel Behaviour and Society, 31*, 200–210. https://doi.org/10.1016/j.tbs.2022.12.003

Li, Y., Zhang, Q., & Wang, H. (2024). An efficient approach for identifying potential bus passenger demand based on multisource data. *Journal of Advanced Transportation, 2024*, 5368577. https://doi.org/10.1155/2024/5368577

Rahimi, E., Shamshiripour, A., Shabanpour, R., Mohammadian, A., & Auld, J. (2024). Analyzing the influence of fuel price shock on urban public transit and interurban automobile travel demand: Evidence from a country with fixed fuel price regulation. *Transportation Research Interdisciplinary Perspectives, 36*.

Wu, J.-L., Lu, M., & Wang, C.-Y. (2023). Forecasting metro rail transit passenger flow with multiple-attention deep neural networks and surrounding vehicle detection devices. *Applied Intelligence, 53*, 11789–11808. https://doi.org/10.1007/s10489-023-04483-x
