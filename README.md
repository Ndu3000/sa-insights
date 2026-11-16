# SA Insights

A live dashboard for South African macroeconomic statistics, with automated
time-series forecasting, model tracking and a portable semantic model.

**Public dashboard:** <https://ndu3000--sa-insights-dashboard.us-east.modal.run>
**Chat API:** <https://ndu3000--sa-insights-chat-api.us-east.modal.run/docs>
**Static mirror:** <https://ndu3000.github.io/sa-insights/>
**MLflow:** <https://ndu3000--sa-insights-mlflow-server.us-east.modal.run>

| | Static mirror | Public dashboard | Chatbot |
|---|---|---|---|
| Data | Refreshed daily | Live warehouse | Live warehouse |
| Downloads | Pre-built | Generated on request | — |
| Custom dashboards | Session only | Saved to the warehouse | — |
| Ask questions | — | — | Cited answers or an honest refusal |
| Cold start | none (CDN) | ~8 s when idle | ~9 s when idle |

All are fed by one pipeline. A scheduled serverless job on Modal refreshes the
warehouse, re-runs every model, re-renders the artefacts and pushes — so the
static mirror is current to within a day rather than frozen. See
[DEPLOYMENT.md](DEPLOYMENT.md).

---

## What it does

| Tab | What you get |
|-----|--------------|
| **Overview** | Six headline indicators with sparklines, a data-derived read-through, and the IMF's published outlook |
| **Explore** | Any of 34 series, any window, with change, CAGR, volatility, distribution and CSV export |
| **Forecast lab** | Backtest-driven model selection, confidence bands, full leaderboard, comparison against IMF projections |
| **Compare** | Indexed multi-series comparison, correlation matrix, scatter inspection |
| **My dashboard** | Build a layout, save it, reload it later |
| **Downloads** | Power BI bundle, Ossie model, CSVs — all built from live data |

---

## Ask it a question

```bash
curl -X POST https://ndu3000--sa-insights-chat-api.us-east.modal.run/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is CPI inflation now?"}'
```

> CPI inflation (YoY) is **4.5%** (2026-09-01), up 1.10 pp year on year (a
deterioration), with 10 citations from the warehouse.

Three possible responses, and all are correct behaviour:

| | When | Example |
|---|---|---|
| **Answer** | The data covers the question | "What is CPI inflation now?" |
| **Clarify** | Genuinely ambiguous | "How has the economy improved?" → *which indicator, and over what period?* |
| **Refuse** | Off-topic | "Who won the 2010 world cup?" |

**Every number comes from a tool call, never from the model.** Prose is scanned
for figures and rejected if any does not appear in a tool result, so an invented
statistic cannot reach a reader. It also works with no LLM at all — the model
only rewrites prose, and the answer is correct without it.

See [DEPLOYMENT.md](DEPLOYMENT.md#chatbot) for the design and what was learned.

---

## Data sources

| Provider | Used for | Licence |
|----------|----------|---------|
| **SARB** — official Web API | Rates, USD/ZAR, real effective FX, CPI level and inflation, PPI, retail/manufacturing/mining volumes, vehicle sales, leading indicator, **quarterly GDP**, **quarterly labour force survey**, **trade**, **JSE all-share** | Public |
| **Stats SA** (via SARB) | Inflation and the Quarterly Labour Force Survey originate here | Public |
| **IMF DataMapper** | Forward projections to 2031: growth, inflation, unemployment, fiscal balance, debt, current account, GDP per capita, population | IMF terms |
| **World Bank** Open Data | Annual structural aggregates where no monthly/quarterly equivalent exists | CC BY 4.0 |

34 indicators, roughly 7,800 observations.

### On recency

Recency is tracked per series and always displayed, because "latest" means
different things for different indicators:

| Indicator | Latest | Lag |
|-----------|--------|-----|
| Repo, prime, USD/ZAR, bond yield | Nov 2026 | days |
| Vehicle sales | Oct 2026 | ~1 month |
| CPI level & inflation, PPI, trade, JSE all-share | Sep 2026 | ~2 months |
| Retail, manufacturing, mining, leading, real effective FX | Aug 2026 | ~3 months |
| Unemployment, labour force participation | Q3 2026 | ~1 quarter |
| Real GDP (quarterly) | Q2 2026 | ~2 quarters |
| Government debt, internet users | 2024 | annual, ~2 years |
| GDP growth, GDP per capita, population, current account | 2025 | annual, ~1 year |
| IMF projections | to 2031 | forward-looking |

**Every monthly and quarterly series now comes from SARB**, the primary
publisher, and sits within its normal publication lag.

Two rounds of source migration got it there. The DBnomics IMF mirror had frozen
— first rates and inflation, later the CPI level, trade and the equity index —
while still returning valid-looking data over 600 days old. Those were replaced
with SARB codes found in its own catalogue (`EconFinDataForSA`,
`ReleaseOfSelectedData`) rather than guessed. Unemployment and participation
moved from the World Bank's annual ILO modelled estimates to the official
**Quarterly Labour Force Survey**, and quarterly GDP moved from FRED to SARB,
which publishes a quarter earlier.

The remaining annual series are genuinely annual at source: the World Bank's
latest vintage for South African government debt is 2024. They are labelled as
such rather than presented as current.

`scripts/check_freshness.py` fails the build if a critical indicator goes stale
again.

Stats SA publishes the underlying CPI and QLFS but exposes no queryable API, and
its SDMX endpoints return a bot-wall. Rather than scrape, the dashboard reads the
same figures from SARB, which republishes within days of release.

**Units note:** trade is in **rand millions**, not US dollars, because that is
what SARB publishes. A dollar-denominated series moves with the exchange rate
alone, which obscures the underlying trade volumes.

---

## Forecasting

Forecasts are not hard-coded to one algorithm. For each series a
**rolling-origin backtest at the requested horizon** picks the winner:

- Naive, seasonal naive, drift baselines
- Damped-trend ETS, with and without seasonality (plus multiplicative on long monthly series)
- Theta
- SARIMA variants
- **Chronos-Bolt** — Amazon's 48M-parameter time-series foundation model, forecasting zero-shot

Selection uses **MASE** (mean absolute scaled error). Below 1.0 beats a naive
forecast; the dashboard states plainly when a model fails to, rather than
burying it.

A 25-year default window keeps models away from regimes that no longer apply,
such as pre-inflation-targeting South Africa.

### Where the foundation model helps

Chronos-Bolt wins a minority of series, mostly where history is short. On long
monthly series classical ETS wins, because there is enough data to estimate
seasonal structure directly rather than borrow a prior. That is the expected
result and the reason it competes rather than replaces.

Because the model is a 48M-parameter network, it runs **at refresh time**, not
in the browser: shipping ~191 MB of weights so WebAssembly could run hundreds of
forward passes is untenable on mobile. The app ships the resulting numbers.

---

## Model tracking

Each series refresh logs to MLflow: parameters (window, horizon, model family),
metrics (MASE, naive MASE, skill over naive), and the full leaderboard as an
artifact. The run id is stored alongside the forecast, so the database and
MLflow cross-reference.

Tracking is **gated on `MLFLOW_TRACKING_URI`** — with no URI set it is a no-op,
so a laptop run needs no server. On Modal the URL is resolved at startup from the
deployed function itself rather than hardcoded, so it cannot go stale across
deploys.

---

## Apache Ossie semantic model

Metrics and dimensions are published in
[Apache Ossie](https://ossie.apache.org) (incubating) format — formerly Open
Semantic Interchange — so the logic stays portable across BI tools, query
engines and AI agents instead of living inside this app.

Two document shapes are emitted deliberately:

- **0.1.1** — the released revision, wrapping the model in a `semantic_model`
  array. What current consumers such as dbt expect.
- **0.2.0.dev0** — the flat draft that the only published JSON Schema validates.
  Upstream marks it unstable, so it is provided for validation rather than as the
  primary artefact.

The model is a small star schema: three time dimensions, one fact per indicator,
33 relationships, and 150+ metrics including cross-dataset measures such as
`real_policy_rate` (repo less CPI) and `credit_spread`.

Downloads are validated against the official schema before being served — an
invalid model is worse than a failed download.

---

## Power BI

The download tab produces a star schema (`dim_indicator`, `dim_date`,
`fact_observation`, `fact_forecast`) plus a TMSL `.bim` model with relationships,
a calendar hierarchy and 17 DAX measures. One long fact table with `series_key`
as the foreign key, rather than 34 near-identical tables — the mapping is
lossless and the field list stays navigable.

---

## Architecture

```
sources.py ──► refresh.py ──► DuckDB ──► app.py (Streamlit)
   │              │              │              │
   │              ├─► MLflow     └─► downloads ─┘
   │              │                (Power BI, Ossie, CSV)
   └─ SARB, IMF, World Bank, FRED
```

- **Store:** DuckDB — analytical time series, no server, one file to back up
- **Packaging:** uv with `pyproject.toml` + `uv.lock`
- **Scheduling:** a serverless cron on Modal (05:15 UTC), or APScheduler in a
  dedicated `worker` container when running under Docker
- **Downloads:** rendered from the warehouse per request, never committed

Deployed, the same pipeline runs on Modal: `refresh_and_publish` does the
fetch/forecast/render/push cycle, `dashboard` serves Streamlit publicly, and
`mlflow_server` serves the tracking UI. State lives on two Volumes so it
survives redeploys.

See [DEPLOYMENT.md](DEPLOYMENT.md) for setup, environment variables, measured
cold-start and run times, and the reasoning behind each choice.

---

## Project layout

```
sa-insights/
├── app.py                    # Streamlit entry point, six tabs
├── sa_insights/
│   ├── catalog.py            # declarative series definitions
│   ├── sources.py            # provider fetchers with retry/backoff
│   ├── store.py              # DuckDB warehouse + cross-process write lock
│   ├── forecast.py           # model zoo, backtesting, intervals
│   ├── chronos_fm.py         # optional Chronos-Bolt adapter
│   ├── refresh.py            # ingest → forecast → persist pipeline
│   ├── scheduler.py          # APScheduler worker entry point
│   ├── tracking.py           # MLflow logging (gated)
│   ├── downloads.py          # artefacts generated on request
│   ├── semantic.py           # Apache Ossie model generator
│   └── ui.py                 # design system, Plotly builders, CSS
├── scripts/                  # one-off exporters for the static site
├── semantic/                 # committed Ossie artefacts + official schema
├── modal_app.py              # serverless deployment: cron, dashboard, MLflow
├── Dockerfile                # multi-stage, uv, non-root, healthcheck
└── docker-compose.yml        # app + worker + MLflow
```

---

## Caveats

Forecasts are statistical extrapolations of historical patterns. They do not
know about policy decisions, shocks or structural breaks, and they are not
investment advice. Publication lags vary by series — the dashboard always shows
the period each value refers to, because "latest" for annual GDP and for the
rand mean very different dates.

Saved dashboards are namespaced per browser session. This is **not**
authentication; see the known gaps in DEPLOYMENT.md.

---

## Licence

Code is MIT licensed. Data remains under each provider's terms as listed above.
