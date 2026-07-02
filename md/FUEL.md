# Fuel Prices: What the Data Shows

> Last updated: 2026-07-02

Fuel prices are included because they influence whether people drive or take public
transit. This page covers Malaysian retail fuel data (RON95, RON97, diesel, an East
Malaysia diesel variant, and two RON95 subsidy schemes) from 2019 onward, and why it
is shaped the way it is for modelling. Each section uses **Main idea**, **Evidence**,
**Analysis**, and **Link**.

---

## Pattern 1 — Prices Move in Steps, Not Smoothly

**Main idea.** Malaysian fuel prices are announced weekly and then held flat until
the next announcement, so the daily series looks like a staircase.

**Evidence.** The autocorrelation of daily diesel and RON95 prices decays very
slowly — a near "unit-root" pattern — because each price stays fixed for exactly
seven days before it can change.

**Analysis.** The raw weekly data has a price for one day and gaps for the other six.
The pipeline fills those gaps by carrying the last announced price forward, which is
exactly how prices behave in real life: the price you paid on Wednesday still applies
on Saturday. No information is lost by doing this.

**Link.** This step behaviour is why absolute price *levels* alone are a weak
predictor and why weekly *change* signals are added (Pattern 3).

---

## Pattern 2 — Subsidised vs Market-Priced Fuels Behave Differently

**Main idea.** RON95 and diesel are subsidised and very stable; RON97 tracks the
market and moves more often.

**Evidence.** RON97's price changes are more frequent and its autocorrelation decays
faster, showing a livelier price process. RON95, capped below market rates, changes
rarely and in larger single steps. From mid-2023, targeted subsidy schemes (BUDI
Madani, SKPS) created a permanent split in the effective RON95 price for eligible
drivers.

**Analysis.** Because these fuels behave differently, they are kept as separate
features rather than merged: RON97 carries real week-to-week variation, while the
subsidy variants capture policy interventions that changed what households actually
paid. Later analysis found the frozen subsidy series carry little predictive signal
(see `FEATURE.md`), but keeping them separate first was the right call.

**Link.** Modal substitution — switching from car to transit when petrol gets
expensive — depends most on the fuels people actually buy, so the distinction
matters (Belloc et al., 2024).

---

## Pattern 3 — Weekly Changes Carry the Short-Term Signal

**Main idea.** The signed weekly price change is more useful for forecasting than the
raw price level.

**Evidence.** The change series (`fp_chg_*`) sit near zero most of the time with
occasional spikes on government revision dates. RON97 has the widest spread of
changes; RON95 the narrowest.

**Analysis.** Raw price levels drift upward over years, which makes them awkward for
the kind of models used here. Converting them to percent and absolute weekly changes
produces steadier, better-behaved features and highlights the moment a price actually
moves — which is when riders are most likely to react.

**Link.** International evidence supports this: fuel price shocks shift travel choices
with a short delay, so change signals are a natural predictor of near-term ridership
(Mily et al., 2024; Rahimi et al., 2024).

---

## What Goes Into the Model

The cleaned fuel data contributes 15 columns: six price levels, three percent-change
series for the main grades, and six weekly absolute changes. Levels give baseline
context; changes give the short-term demand signal.

---

## References

Belloc, I., Giménez-Nadal, J. I., & Molina, J. A. (2024). The gasoline price and the commuting behavior of US commuters: Exploring changes to green travel mode choices. *Journal of Transport Geography, 116*, 104006. https://doi.org/10.1016/j.jtrangeo.2024.104006

Mily, I., Haque, M., & Islam, M. T. (2024). Unveiling the consequence of unprecedented fuel price hike in Bangladesh on consumer travel behavior. *Transportation Safety and Environment*. https://doi.org/10.1080/29941849.2024.2409081

Rahimi, E., Shamshiripour, A., Shabanpour, R., Mohammadian, A., & Auld, J. (2024). Analyzing the influence of fuel price shock on urban public transit and interurban automobile travel demand: Evidence from a country with fixed fuel price regulation. *Transportation Research Interdisciplinary Perspectives, 36*.
