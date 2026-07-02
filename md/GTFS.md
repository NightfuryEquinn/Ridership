# Transit Network (GTFS): What the Data Shows

> Last updated: 2026-07-02

GTFS is the standard format transit operators use to publish their routes, stops, and
schedules. This project uses feeds from four operators — Rapid Rail KL, RapidBus KL,
RapidBus Penang, and KTMB — to describe the physical transit network. Each section
uses **Main idea**, **Evidence**, **Analysis**, and **Link**.

---

## Pattern 1 — Rail Routes Are Long; Bus Routes Are Short

**Main idea.** Route length varies enormously by mode.

**Evidence.** KTM inter-city routes reach 300–400+ km, Rapid Rail KL lines run 10–60
km, and most RapidBus routes are under 30 km. The short bus segments reflect dense,
closely spaced stops typical of an urban feeder network.

**Analysis.** Average segment travel time is a useful summary: shorter segments mean
stops are closer together and service is denser. This becomes one of the network
statistics the model sees.

**Link.** These route characteristics feed the network summary features described
below.

---

## Pattern 2 — The Network Is Concentrated in Two Areas

**Main idea.** Transit stops cluster heavily in the Klang Valley and, to a lesser
extent, Penang.

**Evidence.** Clustering the combined stops from all four operators produces two
dominant groups: the Klang Valley metro area (all four operators present) and the
Penang corridor. KTMB stops form a long line along the Peninsular rail spine from
Johor Bahru to Padang Besar, with smaller clusters around Tebrau (JB) and the
northern Komuter corridor.

**Analysis.** This concentration confirms the transit network is effectively
single-centred on Greater Kuala Lumpur. It also means the network features mostly
describe conditions in that corridor — which is fine, because that corridor also
generates most of the ridership.

**Link.** The heavy skew is why the model uses a few national network totals rather
than trying to split the network by region (Li et al., 2024).

---

## Pattern 3 — Operators Differ in Scale and Role

**Main idea.** Each operator plays a different role in the network.

**Evidence.** Rapid Rail KL has the fewest routes but the most riders per route
(high-capacity rail). RapidBus KL has the most routes and stops. KTMB has the most
connections because of its long inter-city lines. RapidBus Penang is the smallest.
Major interchange stations (KL Sentral, Masjid Jamek) show the highest connectivity.

**Analysis.** These per-operator differences are combined into four national totals so
the model gets a single, stable picture of network scale rather than sparse
per-operator detail that would be noisy for smaller networks (Wang et al., 2024).

**Link.** The four network features are: total stops, total routes, total connections
between consecutive stops, and average segment travel time.

---

## Data Quality

The feeds were validated and cleaned per operator. Genuine problems — such as stop
references that pointed nowhere, or trips with too few stops — were fixed. Cosmetic
issues that do not affect the model (capitalisation of stop names, expired historical
service dates, route colours) were left as-is. Full detail is in `DATA.md`.

---

## References

Li, Y., Zhang, Q., & Wang, H. (2024). An efficient approach for identifying potential bus passenger demand based on multisource data. *Journal of Advanced Transportation, 2024*, 5368577. https://doi.org/10.1155/2024/5368577

Wang, Z., Huang, K., Massobrio, R., Bombelli, A., & Cats, O. (2024). Quantification and comparison of hierarchy in public transport networks. *Physica A: Statistical Mechanics and Its Applications, 634*, 129479. https://doi.org/10.1016/j.physa.2023.129479
