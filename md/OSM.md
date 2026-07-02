# Points of Interest (OSM): What the Data Shows

> Last updated: 2026-07-02

Places people travel to — shops, restaurants, schools, hospitals — help explain why a
transit stop is busy. This project uses OpenStreetMap (OSM) points of interest (POIs)
to measure what surrounds each stop. Each section uses **Main idea**, **Evidence**,
**Analysis**, and **Link**.

---

## Pattern 1 — POIs Are Sorted Into Seven Simple Categories

**Main idea.** Raw OSM tags are messy, so they are grouped into seven clear
categories.

**Evidence.** OSM labels places with a mix of tags. These are mapped into
`transport`, `food`, `retail`, `education`, `healthcare`, `leisure`, and `other`. The
most common individual place types are restaurants, cafés, bus stops, schools, and
pharmacies — typical of Malaysian urban areas.

**Analysis.** Grouping keeps the data usable while preserving the meaningful
distinctions (a hospital and a café attract different trips). The category mapping was
checked to confirm it covers the dominant tags in the Malaysian data.

**Link.** These categories become the per-stop counts described below.

---

## Pattern 2 — Counting What Is Within Walking Distance

**Main idea.** Each stop is described by how many POIs sit within a short walk.

**Evidence.** For every transit stop, the pipeline counts POIs within 500 metres — a
standard walking catchment — for each category plus a total.

**Analysis.** 500 metres reflects how far most people will walk to a stop, so the
count is a reasonable proxy for how much activity a stop serves. City-centre stops
(near KL Sentral, Bukit Bintang) can have 200–500+ POIs nearby; suburban stops may
have fewer than 10.

**Link.** This large gap between busy and quiet stops is why the counts are also stored
in a log-scaled form, which keeps a handful of extreme city-centre stops from
dominating (Yadav et al., 2024).

---

## What Goes Into the Model

The per-stop counts are averaged across all stops to produce eight national summary
features (one total plus seven categories). These describe the typical mix of
destinations around a Malaysian transit stop.

Because these are national averages, they capture the overall setting rather than
stop-by-stop variation. Richer per-stop or per-line POI features could improve
line-level forecasts in future work.

---

## Reference

Yadav, M., Mepparambath, R. M., & Patil, G. R. (2024). An enhanced transit accessibility evaluation framework by integrating Public Transport Accessibility Levels (PTAL) and transit gap. *Journal of Transport Geography, 120*, 103965. https://doi.org/10.1016/j.jtrangeo.2024.103965
