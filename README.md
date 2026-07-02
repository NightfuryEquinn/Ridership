# Mitigating Public Transit Unreliability through Deep Learning Analytics and Ridership Forecasting

> Last updated: 2026-07-02

Masters Final Year Project comparing 15 deep-learning models for Malaysian public transit ridership forecasting across three model series using 8 spatio-temporal feature sources.

This README is a navigational entry point. The detailed documentation (data, pipeline, models, metrics, results, and EDA) lives in the dedicated files listed under [Documentation](#documentation) — it is intentionally not reproduced here.

---

## Scope: How Ridership Forecasting Mitigates Transit Unreliability

Public transit unreliability in Malaysia manifests as demand–supply mismatch: overcrowded peak services, ghost buses on low-demand routes, and capacity decisions made reactively rather than ahead of demand. This study addresses the **analytics layer** of that problem — accurate 7-day-ahead ridership forecasts are the prerequisite for proactive mitigation:

- **Capacity planning** — forecasted demand peaks let operators pre-position rolling stock and crews instead of responding after overcrowding occurs;
- **Headway adjustment** — anticipated demand troughs (holidays, monsoon days) justify schedule thinning without stranding passengers, and forecasted surges justify densified headways before queues form;
- **Demand–supply matching** — per-service-line forecasts (12 lines + system total) expose which corridors are chronically under- or over-served, feeding the EDA-identified underserved-zone analysis (`src/eda/multivariate/`).

The modelling scope is therefore **demand-side forecasting**: this study does not model supply-side reliability events (delays, breakdowns, cancellations) directly — no AVL/on-time-performance data is publicly available for Malaysian operators at daily granularity. Integrating such data is the principal future-work direction (see `md/FUTURE.md`).

---

## Project Structure

```
Ridership/
├── data/                   # All data files (raw, cleaned, features, sequences, scalers)
├── src/                    # Source code
│   ├── features/           # Feature engineering pipeline (one cleaner per source)
│   ├── eda/                # Exploratory analysis (univariate, bivariate, multivariate, thematic)
│   ├── models/             # Deep-learning models, organised by series
│   │   ├── spatio-temporal-based/   # Base ST models       + spatio-temporal-tuned/
│   │   ├── graph-based/             # Base graph models     + graph-tuned/
│   │   ├── attention-based/         # Base attention models + attention-tuned/
│   │   └── hybrid/                  # Hybrid SOTA model (HMT-TSF)
│   ├── outputs/            # Timestamped run dirs (results.json, plots, model.pt)
│   └── utils/              # Shared utilities (metrics, comparison table)
├── md/                     # Detailed documentation (see Documentation below)
├── docs/                   # Reports and presentations (PDFs, DOCX)
├── journal_articles/       # Supporting research papers
├── MAIN.md                 # Single-file compendium of every md/ document
├── AGENTS.md               # Guidance for AI coding agents
├── CURSOR.md               # Project guidance (setup, running, architecture, hyperparameters)
├── check_cuda.py           # GPU/CUDA verification script
├── LICENSE
└── README.md               # This file
```

---

## Documentation

All substantive documentation is maintained in the files below. Start with `CURSOR.md` for setup and run commands, then `md/PIPELINE.md` to build the data.

### Setup & running

| Document | Contents |
|----------|----------|
| `CURSOR.md` / `AGENTS.md` | Environment setup, running the feature pipeline and all 15 models, architecture overview, and hyperparameter defaults (base + tuned) |

### Data & pipeline

| Document | Contents |
|----------|----------|
| `md/PIPELINE.md` | End-to-end feature-pipeline commands, from raw sources to model-ready sequences |
| `md/DATA.md` | Layout and contents of every directory under `data/` |
| `md/FEATURE.md` | Feature engineering decisions and the 79-feature aligned matrix |
| `md/RIDERSHIP.md`, `md/FUEL.md`, `md/HOLIDAY.md`, `md/RAINFALL.md`, `md/GADM.md`, `md/GTFS.md`, `md/POPULATION.md`, `md/OSM.md` | Per-source cleaning and EDA for each of the 8 feature sources |

### Exploratory data analysis

| Document | Contents |
|----------|----------|
| `md/BIVARIATE.md` | Pairwise feature–ridership relationship analyses |
| `md/MULTIVARIATE.md` | Multi-source higher-order interaction analyses |
| `md/THEME.md` | Cross-source thematic visualisations |

### Models, metrics & results

| Document | Contents |
|----------|----------|
| `md/MODEL.md` | Model descriptions (Chapter 3 methodology), selection rationale, and journal references |
| `md/HMT-TSF.md` | Hybrid SOTA architecture — full diagram and per-component rationale |
| `md/NOVEL.md` | HMT-TSF novel architectural contributions (for publication) |
| `md/METRICS.md` | Definitions of the evaluation metrics (Combined%, R², MAE, RMSE, …) |
| `md/RESULTS.md` | Baseline and tuned model results across configurations |
| `md/HMT-TSF-RESULTS.md` | HMT-TSF and HMT-TSF-FR cross-configuration results |
| `md/DIAGNOSTICS.md` | Fit (overfit/underfit) diagnostics |
| `md/FINDING.md` | Chapter 4 — findings and results |
| `md/FUTURE.md` | Chapter 5 — discussion, conclusion, and future work |

---

## License

See [`LICENSE`](LICENSE).
