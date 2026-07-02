# Ridership: What the Data Shows

> Last updated: 2026-07-02

This is the core dataset: daily passenger counts for 12 Malaysian public transit
services from 2019 to 2025. Everything else in the project exists to help forecast
these numbers. This page explains the patterns that shaped how the data was cleaned
and modelled, written for a general reader. Each section follows a simple structure —
**Main idea**, **Evidence**, **Analysis**, and **Link** (how it feeds the model).

---

## The 12 Services at a Glance

| Service | Type | Character |
|---|---|---|
| LRT Kelana Jaya | Urban rail | Highest-ridership line; strong weekday commuter pattern |
| MRT Kajang | Urban rail | Only line to exceed its pre-COVID level by 2024 |
| LRT Ampang | Urban rail | Slower recovery; noisier, more leisure travel |
| KL Monorail | Urban rail | Most tourism-sensitive; deepest COVID collapse |
| MRT Putrajaya | Urban rail | Launched 2022-06-16; ramp-up then stabilises |
| KTM ETS | Inter-city rail | Friday/Sunday peaks (leisure travel), not weekday |
| KTM Intercity | Inter-city rail | Similar to ETS; slow post-COVID recovery |
| KTM Komuter Utara | Commuter rail | Northern corridor; sharp weekday/weekend contrast |
| KTM Komuter | Commuter rail | Reported separately only from 2023-09-10 |
| KTM Tebrau | Cross-border rail | Johor Bahru–Singapore; tied to commuter flows |
| RapidBus KL | Bus | Highest-volume bus; rain-sensitive, day-to-day variable |
| RapidBus Penang | Bus | Smaller market; weaker weekly pattern |

`total_ridership` (the main forecast target) is the daily sum of all 12 services.

---

## Pattern 1 — The COVID Lockdown Is the Biggest Event in the Data

**Main idea.** The Movement Control Order (MCO, 2020-03-18 to 2021-12-31) cut
ridership by 65–90% across every service and remains the single largest signal in
the full history.

**Evidence.** Change-point detection on `total_ridership` finds four clear phases:
a pre-COVID baseline, the lockdown trough, a phased reopening, and a stable
post-COVID trend. The `is_mco` flag marks roughly 650 days (about a quarter of the
2019–2025 series). Recovery is uneven: most rail lines sit at 60–80% of their
pre-COVID level by 2024, inter-city rail stays below 60%, and MRT Kajang is the
only line to climb back above 100%.

**Analysis.** A break this large breaks the assumption that the data behaves the
same way over time, which most forecasting models rely on. That is why the project
keeps two versions of the dataset — one with the lockdown period and one without —
and why the proposed HMT-TSF model includes a "regime" component that learns to
tell the pre-COVID, lockdown, and post-COVID periods apart. Keeping the lockdown
rows but tagging them (rather than deleting them) preserves the day-to-day
continuity that sequence models need.

**Link.** This behaviour matches international findings that transit demand does not
simply snap back after a shock and recovers at different speeds by mode (Airak et
al., 2023; Lee et al., 2024).

---

## Pattern 2 — A Strong Weekly Rhythm

**Main idea.** Ridership rises on weekdays and falls on weekends, and this weekly
cycle is the most reliable short-term pattern in the data.

**Evidence.** A seasonal decomposition of `total_ridership` splits the series into a
long trend, a repeating 7-day cycle, and leftover noise. The 7-day cycle is
dominant. Urban rail (LRT/MRT/Monorail) shows a bimodal weekday peak (morning and
evening commute); inter-city services such as ETS instead peak on Fridays and
Sundays as people travel for the weekend.

**Analysis.** Because the pattern repeats every seven days, the model needs a way to
know the day of week without treating "Sunday" as numerically far from "Monday". The
pipeline encodes each day and month as a pair of sine/cosine values, which places
them on a smooth circle so the model reads the calendar without artificial jumps at
week or year boundaries.

**Link.** This weekly structure is why the default 14-day look-back window works well —
it covers exactly two full commuting cycles.

---

## Pattern 3 — Holidays Reshape Demand

**Main idea.** Public holidays suppress commuter ridership, and the effect starts a
day or two before the holiday itself.

**Evidence.** Total ridership on public holidays typically falls to 30–60% of a
comparable weekday. Major festivals differ: Hari Raya Aidilfitri causes the deepest
drop in KL as workers leave the city, while Chinese New Year dips urban rail but
lifts inter-city KTM travel. Month-on-month growth shows visible dips around these
festivals.

**Analysis.** A simple "is it a holiday?" flag is not enough because demand shifts
before and after the day. The pipeline adds "days until the next holiday" and "days
since the last holiday" counters so the model can learn the run-up and the recovery,
not just the day itself.

**Link.** Holiday timing is one of the strongest non-seasonal drivers the model
learns to use (Wu et al., 2023; Li et al., 2022).

---

## Pattern 4 — New Lines and Missing Days

**Main idea.** Some services simply did not exist for part of the study window, so
their "missing" values are structural, not data errors.

**Evidence.** MRT Putrajaya launched 2022-06-16, Tebrau 2022-06-19, and KTM Komuter
began reporting separately on 2023-09-10. Within the 2022–2025 window, this leaves
166, 169, and 617 missing days respectively.

**Analysis.** Before a service launches, its ridership is genuinely zero, so those
gaps are filled with zero rather than an average — a service that does not run cannot
have "typical" demand. New lines also show more early volatility as riders discover
them.

**Link.** Correct handling here keeps the sliding-window sequences clean and stops
false signals from leaking into training.

---

## Pattern 5 — A Gradual, Trend-Driven Recovery

**Main idea.** Beyond the weekly cycle, ridership carries a slow upward trend as the
network recovers and matures.

**Evidence.** Post-COVID growth rates were high in 2022–2023 (partly a rebound from a
very low base) and normalised by 2024. MRT Kajang shows genuine new demand rather
than recovery, while inter-city and bus services recover more slowly and smoothly.

**Analysis.** Cyclical day/month encodings repeat every year and cannot represent a
one-directional trend, so the pipeline also adds plain `year` and `day_of_year`
features. These let the model track where the network sits on its longer recovery arc.

**Link.** Recent ridership history is also captured through 7-, 14-, and 28-day lag
features, which are most useful during the steep recovery ramp when today's demand
closely follows recent days.

---

## References

Airak, S., Abd Sukor, N. S., & Abd Rahman, N. (2023). Travel behaviour changes and risk perception during COVID-19: A case study of Malaysia. *Transportation Research Interdisciplinary Perspectives, 18*, 100784. https://doi.org/10.1016/j.trip.2023.100784

Lee, S., Kim, J., & Cho, K. (2024). Temporal dynamics of public transportation ridership in Seoul before, during, and after COVID-19 from urban resilience perspective. *Scientific Reports, 14*, 9078. https://doi.org/10.1038/s41598-024-59323-w

Li, W., Guan, H., Han, Y., Zhu, H., & Wang, A. (2022). Short-term holiday travel demand prediction for urban tour transportation: A combined model based on STC-LSTM deep learning approach. *KSCE Journal of Civil Engineering, 26*(9), 4086–4102. https://doi.org/10.1007/s12205-022-1698-3

Wu, J.-L., Lu, M., & Wang, C.-Y. (2023). Forecasting metro rail transit passenger flow with multiple-attention deep neural networks and surrounding vehicle detection devices. *Applied Intelligence, 53*, 11789–11808. https://doi.org/10.1007/s10489-023-04483-x
