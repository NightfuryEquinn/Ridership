# Speaker Script — EDA, HMT-TSF, Results & Findings (Slides 22–27, 36–38, 44–50, 52, 54)

**Target duration:** ~9:30–10:00 · **Word count:** ~1,864

*Brief slide 21 if needed — it lists all six EDAs; slides 22–27 are the figures.*

---

## Slide 22 — MCO collapse and recovery (~0:35)

This changepoint plot is the anchor finding. Between March 2020 and December 2021, total daily ridership fell **85 to 90 percent** — near-instant collapse, staged recovery. We detect **six structural shifts** across four behavioural phases.

Recovery is **mode-unequal**: rail returns to a lower share of its own baseline than bus, because CBD office demand was removed by remote work rather than displaced. Calendar and weather no longer map to ridership the same way before and after the break.

That motivates three design choices: **dual datasets** — MCO included and excluded; **RevIN** to normalise per-sample distribution shift; and the **regime stream** with three learnable embeddings for pre-MCO, MCO, and post-MCO.

---

## Slide 23 — Fuel price elasticity (~0:35)

Fuel and ridership show a **positive association** — RON95 correlation about 0.29, diesel similar. In developing contexts, rising pump prices push commuters toward transit.

But level prices are weekly staircases; post-MCO, apparent level correlation partly reflects shared upward drift. The bottom-left panel shows **weekly percent changes** capture the lagged elasticity better — typically one to three days.

So we retain both level and change fuel features, and SHAP later drops frozen RON95 variants that carry no day-to-day variance after subsidy fixation.

---

## Slide 24 — Northeast monsoon (~0:30)

These bars count wet, very wet, and extreme rainfall days by state. Intensity peaks **November to January** under the Northeast Monsoon. The east coast — Terengganu, Kelantan — dominates extreme events; west-coast states see fewer.

National rainfall totals would mask that heterogeneity. We keep **state-level continuous rainfall** for all twelve services, and cap interpolation at seven days so one monsoon spell is bridged without inventing a wet season.

---

## Slide 25 — Holiday anticipation (~0:40)

Four panels compare Normal, Pre-Holiday, Holiday, and Post-Holiday regimes. Mean ridership is similar for Normal and Pre-Holiday, then drops sharply on the holiday itself — about **18 percent** — with only partial recovery after.

The day-of-week profile shows a **mid-week surge before holidays** as travellers take leave early; the holiday trough is **concentrated on weekdays** because weekends already sit at leisure baseline. Post-holiday demand re-enters gradually, not in one jump — *balik kampung* outflow, staggered return.

A binary holiday flag would miss the flanking days. That is why we use **directional lead–lag counters** and **X_future** known-calendar conditioning inside the forecast horizon.

---

## Slide 26 — Weather-induced surge (~0:40)

Heavy rain does not uniformly suppress ridership. Two mechanisms compete: moderate rain defers discretionary trips, but above roughly the **75th percentile** — here 100 millimetres — flooded roads push drivers onto grade-separated rail. Substitution.

The pooled correlation heatmap looks near zero — about 0.08 — which is **not** evidence that weather is irrelevant; opposing effects cancel in a linear screen. The bottom-right panel shows **large positive day-to-day surges** on several extreme-rain days.

We forecast all twelve lines separately and keep rainfall continuous at state level so rail–bus sign differences are not collapsed into one national mean.

---

## Slide 27 — Feature correlation (~0:35)

This 79-by-79 matrix summarises linear structure. **Lag features at 7, 14, and 28 days** dominate — PCA attributes about 72 percent of variance to that block. Last week's same weekday is near-sufficient under normal conditions.

Fuel diesel and day-of-week are the strongest exogenous linear predictors. Rainfall and holidays look weak here for the cancellation reasons we just saw. **Static network and demographic columns show near-zero correlation** — not because they are unimportant, but because they do not vary over time.

Linear screening understates non-linear and time-invariant structure. That is why feature reduction uses **SHAP**, not correlation ranking — and why a **14-day look-back** is enough once calendar features supply seasonal context.

---

## Slide 36 — Why hybrid (~0:25)

This is HMT-TSF — the Hybrid Multi-scale Temporal Spatio-Feature Forecaster.

I call it *hybrid* because it fuses **three paradigms** — temporal, relational, and regime — co-trained end-to-end. CNN-LSTM and ASTGCN stack components on the same axis; those are composite, not hybrid. Here each stream acts on a different axis: time, feature relations, and operating state.

Input is a 14-day window over 79 features — or 53 in the reduced variant. RevIN normalises per sample to handle the MCO shift, then fans out three ways. Graph and regime **bypass** fusion and the transformer so feature identity is preserved.

---

## Slide 37 — Three encoders (~0:30)

**Temporal** — green: Feature Group Fusion with learned gates over target, lag, calendar, external, and static groups → transformer → multi-scale causal TCN → **h_t**.

**Graph** — blue bypass: features-as-nodes GCN on Pearson adjacency, |corr| ≥ 0.1 → **h_s**.

**Regime** — blue bypass: three embeddings for pre-MCO, MCO, and post-MCO, inferred from the input without a date stamp → **h_r**.

Each branch is independently ablatable and non-redundant.

---

## Slide 38 — Fusion and output (~0:25)

h_t, h_s, and h_r concatenate into a Squeeze-and-Excitation gate that learns which stream to trust per sample. Dual forecast heads produce the base seven-day output. **X_future** then adds a correction from known future calendar features — holidays and weekends are deterministic at deployment. RevIN denormalises to ridership scale.

---

## Slide 44 — Baseline (~0:20)

Fourteen baselines, MCO excluded, 14-day look-back, ranked by Combined%.

Top eight sit within 1.5 points — a saturated field. **Informer leads at 79.13%**; six of the top eight are LSTM-family. Weekly periodicity in our EDA explains the recurrent strength. Informer wins because ProbSparse attention plus a generative decoder matches that accuracy without autoregressive error build-up.

---

## Slide 45 — Tuned (~0:15)

Tuning is not a free upgrade — median gain ~0.5 points; STFGNN drops 8.47. **Informer stays first at 79.99%**; LSTM gains only 0.30. Extra capacity helps under-fit attention models but hurts flexible graph designs. Good-fit rate falls from 26% to 14%.

---

## Slide 46 — Look-back (~0:20)

**lb14 is best for ten of sixteen models**; none peak at lb56. Fourteen days is two weekly cycles; seasonality is already in calendar features. Informer, BiLSTM, and LSTM all peak at lb14. STFGNN collapses from 73.94 to 43.25 at lb56 — longer windows inject noise.

---

## Slide 47 — MCO robustness (~0:25)

Including the MCO period tests transfer across an 85–90% ridership collapse.

**Informer is the most robust baseline** — only −5.88, mco R² 0.705. BiLSTM holds at −6.29. But **LSTM drops −19.21**, mco R² 0.390 — fifth on test accuracy, yet it memorised pre-break patterns. CNN-LSTM and STFGNN fall below a mean predictor. Informer is the deployable baseline; LSTM-family test scores overstate readiness.

---

## Slide 48 — HMT-TSF with X_future (~0:20)

With known-future calendar conditioning: **nomco_lb14 at 85.78%**, R² 0.897 — **+6.65 over Informer**, MAE 28% lower. Under MCO, 81.99% with degradation only −3.79. RevIN plus regime gating lets the hybrid survive the break.

---

## Slide 49 — HMT-TSF without X_future (~0:20)

Ablation without X_future: **80.89%** — still +1.76 over Informer, but inside the saturated band. MAE cut shrinks to ~10%. MCO degradation worsens to −5.40. X_future is not a cheat — those calendar bits are known at inference — and it aids structural-break transfer, not just headline accuracy.

---

## Slide 50 — Feature-reduced (~0:20)

Same architecture, 53 features: SHAP drops the static group and near-constant fuel columns. **FR with X_future: 86.59%, R² 0.906** — study headline. Zero-attribution inputs add graph noise under normal ops. Under MCO, FR degrades more (−7.85 vs −3.79 full) because near-constant fuel columns stabilise lockdown shift. **Rule: FR for normal ops; full model for shock-prone regimes.**

---

## Slide 52 — Discussion (~0:40)

**One — Hybrid multi-modal architectures outperform single-paradigm models.**
**Because** the top eight baselines sat within 1.5 Combined% — no single paradigm escaped that ceiling.
**Why** HMT-TSF broke it only through parallel fusion and task-appropriate inputs; marginal gains lie in multi-modal design, not incremental paradigm refinement.

**Two — Test-set accuracy conceals systematic overfitting.**
**Because** all six LSTM-family models scored zero good-fit verdicts despite ranking second to fifth, while Informer alone was clean at 6/6.
**Why** deployment decisions must pair Combined% with fit diagnostics — otherwise memorising models look as deployable as generalising ones.

**Three — The 14-day look-back is decisively optimal.**
**Because** ten of sixteen baselines peaked at lb14, none at lb56, and slower seasonality is already encoded in calendar features.
**Why** fourteen days — two weekly cycles — is the operational default for a seven-day horizon; longer windows add redundancy or active harm.

**Four — Structural inductive biases determine robustness under distributional shift.**
**Because** MCO inclusion spread degradation from −5.88 to −32.94, with flexible learners failing hardest while constrained and regime-aware designs transferred.
**Why** robustness is an architectural choice — RevIN and regime gating explain HMT-TSF's −3.79, not peak nomco accuracy alone.

**Five — The learned feature hierarchy is domain-consistent and enables principled reduction.**
**Because** SHAP recovered the same driver hierarchy as our EDA, and removing 26 zero-attribution features raised accuracy to 86.59%.
**Why** feature reduction is validated and actionable — but features useless in normal ops can still stabilise shock regimes, as FR's MCO loss shows.

---

## Slide 54 — Implications (~0:40)

**One — Features-as-nodes transfers to national ridership, but temporal-adjacency graphs break at long windows.**
**Because** MTGNN and STGCN performed competitively on Pearson feature graphs, while STFGNN collapsed at lb56 when temporal adjacency was built from extended profiles.
**Why** graph methods transfer to multivariate ridership panels, but adjacency construction must match the forecasting horizon — not every traffic-graph design ports directly.

**Two — Fit diagnostics are a necessary complement to test accuracy.**
**Because** BiLSTM and TPA-LSTM ranked second and third on Combined% yet scored zero good-fit verdicts across all twelve configurations.
**Why** operators must rank deployable models on generalisation quality, not headline test scores alone — accuracy-only ranking would mislead procurement decisions.

**Three — Regime gating plus RevIN delivers structural-break robustness.**
**Because** HMT-TSF degraded only −3.79 under MCO while retaining 81.99% Combined%, with clean fit throughout the break.
**Why** Malaysian transit must be architecturally prepared for regime discontinuity — a single shock like the MCO is enough to invalidate models trained without explicit shift handling.

**Four — Established networks can adopt phased integration.**
**Because** walk-forward validation showed stable block performance and HMT-TSF-FR cuts MAE by up to 33% over the best baseline under normal conditions.
**Why** Klang Valley and Penang operators can run the forecaster in parallel with existing planning and adopt forecast-informed frequency and fleet decisions incrementally as confidence builds.

**Five — Developing corridors can use demand-responsive network design.**
**Because** the features-as-nodes design scales by adding new service lines as nodes, and SHAP-guided reduction keeps the input lean as the network grows.
**Why** Johor and East Malaysia can plan new services around demand forecasts from the outset rather than retrofitting forecasting onto fixed legacy schedules.

**Close:** 14-day window, feature-reduced model in normal operations, full model when a shock is plausible. Thank you.

---

## Quick reference — EDA → design link

| Slide | EDA | Design consequence |
|-------|-----|-------------------|
| 22 | MCO 85–90% drop, staged recovery | Dual datasets; RevIN; regime gating |
| 23 | Fuel elasticity via weekly change | Level + change features; SHAP drops frozen RON95 |
| 24 | NE monsoon, east-coast extremes | State-level rainfall; 7-day interpolation cap |
| 25 | Holiday pre-dip, post-surge, weekday-only | Lead–lag counters; X_future conditioning |
| 26 | Rain suppression vs substitution | Per-line forecast; continuous state rainfall |
| 27 | Lags dominate; static near-zero | 14-day look-back; SHAP-based FR |

## Quick reference — key numbers

| Item | Value |
|------|-------|
| Best baseline (Informer, nomco lb14) | 79.13% |
| Best tuned baseline (Informer) | 79.99% |
| HMT-TSF + X_future (nomco lb14) | 85.78% (+6.65) |
| HMT-TSF − X_future (nomco lb14) | 80.89% (+1.76) |
| HMT-TSF-FR + X_future (nomco lb14) | 86.59% (+7.46) |
| HMT-TSF MCO Δ (lb14, with X_future) | −3.79 |
| Informer MCO Δ (lb14) | −5.88 |
| LSTM MCO Δ (lb14) | −19.21 |
