# Population Density: What the Data Shows

> Last updated: 2026-07-02

Where people live shapes where transit demand comes from. This project uses a 2020
gridded population-density map of Malaysia (roughly 100 m cells) to measure how
densely populated each area is. Each section uses **Main idea**, **Evidence**,
**Analysis**, and **Link**.

---

## Pattern 1 — Density Is Extremely Concentrated

**Main idea.** A small part of Malaysia is very densely populated; most of the country
is not.

**Evidence.** Central Kuala Lumpur grid cells exceed 10,000 people per km², while the
vast forested interior is near zero. Secondary clusters appear along the west coast
(Penang, Johor Bahru, Ipoh) and around Kota Kinabalu.

**Analysis.** This extreme skew means a plain average would be dominated by a few KL
cells and misrepresent the country. The pipeline therefore also computes a
log-transformed density, which compresses the extreme tail and gives a more
representative central value. The median (rather than the mean) is used for the same
reason — it resists being pulled by outliers.

**Link.** The model uses the national median and log-median density as stable summary
features (Al-Ansari & Al-Mamoori, 2022).

---

## Pattern 2 — Higher Density Lines Up With Better Transit

**Main idea.** The busiest transit areas are also the densest.

**Evidence.** Federal territories (KL, Putrajaya, Labuan) have the highest median
densities as compact urban enclaves; Selangor leads among the states. Placing the
grid onto state boundaries confirms Klang Valley cells correctly fall in Selangor and
KL, not neighbouring states.

**Analysis.** High-density states also tend to have denser transit networks, which
means the population, GTFS, and POI features all point in a consistent direction. This
cross-check gives confidence the spatial data is aligned correctly.

**Link.** Transit stops sit systematically in higher-density areas than the national
average — a basic network-design principle that the stop-level sampling confirms
(Yadav et al., 2024).

---

## How Stop-Level Density Is Measured

For each transit stop, the pipeline finds the nearest population grid cell (typically
within 0.5 km for urban stops) and records its density. These per-stop values are kept
for possible future stop-level modelling; the main model uses the national median
summary.

---

## References

Al-Ansari, N., & Al-Mamoori, S. K. (2022). Do the population density and coverage rate of transit affect the public transport contribution? *Cogent Engineering, 9*(1), 2143059. https://doi.org/10.1080/23311916.2022.2143059

Yadav, M., Mepparambath, R. M., & Patil, G. R. (2024). An enhanced transit accessibility evaluation framework by integrating Public Transport Accessibility Levels (PTAL) and transit gap. *Journal of Transport Geography, 120*, 103965. https://doi.org/10.1016/j.jtrangeo.2024.103965
