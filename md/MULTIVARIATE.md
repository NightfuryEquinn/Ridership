# How the Factors Work Together

> Last updated: 2026-07-02

The earlier pages looked at ridership against one factor at a time. This page looks at
several factors together, to understand the higher-order patterns that shape the model
design. Each section uses **Main idea**, **Evidence**, **Analysis**, and **Link**.

---

## Pattern 1 — The Features Fall Into Clear Blocks

**Main idea.** When all features are compared against each other, they group into a
few natural blocks.

**Evidence.** A correlation view of the full feature set shows five blocks:
1. **Ridership** — all 12 services and the recent-history lag features move strongly
   together.
2. **Calendar** — day and month cycles pair up, and weekends/holidays show clear
   negative links with commuter ridership.
3. **External** — fuel prices move together as one tight group; rainfall states are
   mostly independent of each other and of ridership.
4. **Static** — population, network, and boundary features barely move over time, so
   they show near-zero correlation with everything.

**Analysis.** The strongest signal by far is ridership's own recent history. The
static features, being effectively constant day to day, carry no time-varying signal —
an early hint that they could be dropped later (confirmed in `FEATURE.md`).

**Link.** These blocks also shape the graph-based models, which connect features that
move together and largely isolate the static ones.

---

## Pattern 2 — Recent History Is the Strongest Predictor

**Main idea.** Yesterday's and last week's ridership are the best single predictors of
tomorrow's.

**Evidence.** The 7-, 14-, and 28-day ridership lags correlate very strongly with
total ridership (around 0.87–0.94). A naive "same as last week" forecast already
explains a large share of the variation because of the strong weekly rhythm.

**Analysis.** This sets the bar every model must clear: any useful model has to beat
simple persistence. It also confirms that autoregressive (recent-history) structure
belongs in the feature set regardless of model type.

**Link.** Lag dominance is a consistent finding in transit forecasting and justifies
the lag features (Wu et al., 2023; Guo et al., 2025).

---

## Pattern 3 — Where Simple Models Fail

**Main idea.** A naive recent-history forecast breaks down exactly around the events
that matter most.

**Evidence.** The residuals from a simple 7-day-lag baseline are largest around public
holidays (the lag model cannot see a holiday coming) and at the COVID boundary (a
sudden regime change).

**Analysis.** These failure points are precisely what the deep-learning models are
built to handle — holiday-aware features, run-up/recovery counters, and the
regime-aware design in the proposed HMT-TSF model.

**Link.** Targeting these specific weaknesses is the motivation for the more advanced
architectures.

---

## Pattern 4 — Multicollinearity Is Present but Manageable

**Main idea.** Several features carry overlapping information, especially fuel prices.

**Evidence.** The fuel price levels are highly collinear (they all move under the same
pricing regime), as are the 12 ridership services. Weekly change features are far less
collinear.

**Analysis.** For the deep-learning models used here this does not hurt prediction —
these models do not rely on cleanly separable coefficients. It does, however, make the
feature set redundant, which supports later trimming and points toward learned-graph
models that can find their own structure instead of relying on raw correlations.

**Link.** This redundancy is what the feature-reduction step later exploits (see
`FEATURE.md`; Rahimi et al., 2024).

---

## References

Guo, Z., Wang, L., & Chen, Y. (2025). Data-driven predictive modelling of stop-level public transit patterns. *Transportation*. https://doi.org/10.1007/s11116-025-10689-4

Rahimi, E., Shamshiripour, A., Shabanpour, R., Mohammadian, A., & Auld, J. (2024). Analyzing the influence of fuel price shock on urban public transit and interurban automobile travel demand: Evidence from a country with fixed fuel price regulation. *Transportation Research Interdisciplinary Perspectives, 36*.

Wu, J.-L., Lu, M., & Wang, C.-Y. (2023). Forecasting metro rail transit passenger flow with multiple-attention deep neural networks and surrounding vehicle detection devices. *Applied Intelligence, 53*, 11789–11808. https://doi.org/10.1007/s10489-023-04483-x
