# Holidays: What the Data Shows

> Last updated: 2026-07-02

Malaysia's public and school holidays change travel demand in clear ways, so the
holiday calendar (2019–2026) is built into the dataset. This page explains the
holiday patterns and how they become model features. Each section uses **Main idea**,
**Evidence**, **Analysis**, and **Link**.

---

## Pattern 1 — Public and School Holidays Act Differently

**Main idea.** Public holidays cause sharp one-day drops; school holidays cause slow,
multi-week shifts.

**Evidence.** Public holidays are almost all single-day events (sometimes a 2–3 day
long weekend). School breaks run one to four weeks. On a public holiday, total
ridership typically falls to 30–60% of a normal weekday.

**Analysis.** Because the two holiday types behave so differently, they are kept as
separate flags (`is_public_holiday`, `is_school_holiday`) rather than merged. Merging
them would blur a sharp single-day commuter drop into a gentle multi-week trend and
weaken both signals.

**Link.** This separation lets the model treat a festival day and a school break as
the distinct events they are.

---

## Pattern 2 — Demand Shifts Before and After the Day Itself

**Main idea.** Ridership starts changing a day or two before a holiday and takes a
day or two to recover afterward.

**Evidence.** A gradual decline begins 1–2 days before public holidays, bottoms out
on the day, and recovers over the following days. Hari Raya Aidilfitri shows the
deepest run-up as workers travel back to their hometowns.

**Analysis.** A plain "is it a holiday today?" flag cannot capture this spread, so the
pipeline adds counters for "days until the next holiday" and "days since the last
holiday" for both holiday types. These let the model learn the anticipatory dip and
the recovery, not just the day itself.

**Link.** Encoding the run-up and recovery is a proven way to improve holiday-period
forecasts (Wu et al., 2023; Li et al., 2022).

---

## Pattern 3 — Different Festivals, Different Effects

**Main idea.** The size and direction of a holiday's effect depend on the festival.

**Evidence.** Hari Raya Aidilfitri causes the deepest KL urban-rail suppression;
Chinese New Year dips urban rail but lifts long-distance KTM travel; National Day and
Malaysia Day cause only mild dips.

**Analysis.** The model does not need a separate feature for each festival. Because
festivals fall at consistent times of year, the holiday flags combined with the
month sine/cosine features let the model implicitly learn these festival-specific
effects.

**Link.** This keeps the feature set compact while still capturing festival variety.

---

## A Note on State-Specific Holidays

Some holidays (such as Thaipusam) are observed only in certain states. Because the
ridership data is reported nationally, these days are flagged as public holidays for
the whole country — a deliberate simplification that avoids needing state-by-state
ridership targets. The original state details are kept in a notes field for reference.

---

## Data Quality Check

Malaysia has roughly 17–20 federal public holidays and 80–100 school-holiday days per
year, and these counts are stable across the study period. That stability confirms
there are no systematic gaps in the calendar.

---

## References

Li, W., Guan, H., Han, Y., Zhu, H., & Wang, A. (2022). Short-term holiday travel demand prediction for urban tour transportation: A combined model based on STC-LSTM deep learning approach. *KSCE Journal of Civil Engineering, 26*(9), 4086–4102. https://doi.org/10.1007/s12205-022-1698-3

Wu, J.-L., Lu, M., & Wang, C.-Y. (2023). Forecasting metro rail transit passenger flow with multiple-attention deep neural networks and surrounding vehicle detection devices. *Applied Intelligence, 53*, 11789–11808. https://doi.org/10.1007/s10489-023-04483-x
