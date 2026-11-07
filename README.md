# SA Insights

A live dashboard for South African macroeconomic statistics, with a
one-shot time-series forecasting lab and a portable Apache Ossie semantic
model.

**Live data · automated model selection · shareable semantic layer**

**→ [ndu3000.github.io/sa-insights](https://ndu3000.github.io/sa-insights/)**

---

## What it does

| Tab | What you get |
|-----|--------------|
| **Overview** | Six headline indicators with trend sparklines, a data-derived economic read-through, and the IMF's published outlook |
| **Explore** | Any of 29 series, any time window, with change, CAGR, volatility, distribution and CSV export |
| **Forecast lab** | Backtest-driven model selection, confidence bands, a full leaderboard, and comparison against IMF projections |
| **Compare** | Indexed multi-series comparison, correlation matrix and scatter inspection |
| **My dashboard** | Build and export your own set of tiles and charts |

## Data sources

All sources are public and key-free. Every series is cached locally for 12
hours and can be refreshed on demand.

| Provider | Used for | Licence |
|----------|----------|---------|
| **South African Reserve Bank** — official Web API | Policy and prime rates, bond yields, USD/ZAR, real effective exchange rate, CPI and PPI, retail and manufacturing volumes, mining output, vehicle sales, the leading indicator | Public |
| **Stats SA** (carried via SARB) | Consumer and producer inflation originate here | Public |
| **World Bank** Open Data API | Annual macro, labour, fiscal and social indicators | CC BY 4.0 |
| **IMF** IFS and WEO, via the DBnomics mirror | Long-run history and published forecasts to 2030 | IMF terms |
| **FRED** (St. Louis Fed) | Quarterly real GDP | Public |

Coverage spans 1946 to the present across 32 indicators — roughly 8,100
observations.

### On recency

Recency is tracked per series and always displayed, because "latest" means very
different dates for different indicators:

| Indicator | Latest | Lag |
|-----------|--------|-----|
| Repo rate, prime rate, USD/ZAR, bond yield | Nov 2026 | days |
| Vehicle sales | Oct 2026 | ~1 month |
| **CPI / PPI inflation** | **Sep 2026** | **~2 months** |
| Retail, manufacturing, mining, leading indicator | Aug 2026 | ~3 months |
| Real GDP (quarterly) | Q1 2026 | ~10 months |
| GDP growth, unemployment, population, debt ratio | 2025 (annual) | 1 year+ |

An earlier version of this project read rates and inflation from the DBnomics
IMF mirror, which was frozen at early 2025 — a lag of over 600 days. Those
series now use the SARB Web API and are current to within days. Annual
World Bank indicators remain a year behind by construction: they are published
annual statistics, so treat them as structural benchmarks rather than current
readings.

Stats SA publishes the underlying CPI and PPI but exposes no queryable API, and
its Nesstar endpoint was unreachable during this work. Rather than scrape the
site, the dashboard reads the same figures from SARB, which republishes them
within days of release.


## Forecasting approach

Forecasts are not hard-coded to one algorithm. For each series we run a
**rolling-origin backtest at the requested horizon** and let the data pick the
model:

- Naive and seasonal naive baselines
- Drift
- Damped-trend ETS, with and without seasonality (plus multiplicative seasonality on long monthly series)
- Theta
- SARIMA variants (available in the thorough model pool)

Selection uses **MASE** (mean absolute scaled error), which is scale-free and
comparable across series. Below 1.0 means the model beats a naive forecast;
above 1.0 the dashboard tells you so explicitly rather than hiding it.

A 25-year default history window keeps models away from regimes that no longer
apply (such as pre-inflation-targeting South Africa), and can be overridden.

## Apache Ossie semantic model

The project publishes its metrics and dimensions in
[Apache Ossie](https://ossie.apache.org) (incubating) format — formerly Open
Semantic Interchange — so the business logic stays portable across BI tools,
query engines and AI agents instead of living inside this app.

```
semantic/
  sa_insights.ossie.yaml        # version 0.1.1 (released shape)
  sa_insights.ossie-draft.yaml  # version 0.2.0.dev0 (flat draft)
  ossie-schema.json             # official schema, vendored for validation
```

The model is a small star schema: three `dim_date_*` time dimensions at
monthly, quarterly and annual grain; one `fct_<indicator>` per series; 29
relationships; and 150 metrics, including cross-dataset measures such as
`real_policy_rate` (repo rate less CPI) and `credit_spread`.

Two document shapes are emitted deliberately:

- **0.1.1** is the *released* revision, wrapping the model in a
  `semantic_model` array. This is what current consumers such as dbt expect.
- **0.2.0.dev0** is the *flat draft* shape that the only published JSON Schema
  validates. Upstream marks it unstable, so it is provided for validation and
  forward-compatibility rather than as the primary artefact.

Regenerate after any catalogue or data change:

```bash
python scripts/build_semantic_model.py           # writes semantic/*
python scripts/build_semantic_model.py --check    # validate only
```

Generation fails the build if the draft document does not validate against the
official schema.

## Running locally

```bash
python -m venv .venv
source .venv/Scripts/activate     # Windows; use .venv/bin/activate on Unix
pip install -r requirements.txt

streamlit run app.py
```

Then open <http://localhost:8501>.

## Project layout

```
sa-insights/
├── app.py                        # Streamlit entry point, all five views
├── sa_insights/
│   ├── catalog.py                # declarative series definitions
│   ├── sources.py                # provider fetchers + disk cache
│   ├── forecast.py               # model zoo, backtesting, intervals
│   ├── semantic.py               # Apache Ossie model generator
│   └── ui.py                     # design system, Plotly builders, CSS
├── scripts/
│   └── build_semantic_model.py   # regenerate + validate the semantic model
├── semantic/                     # committed Ossie artefacts
└── docs/                         # GitHub Pages site
```

## Caveats

Forecasts are statistical extrapolations of historical patterns. They do not
know about policy decisions, shocks or structural breaks, and they are not
investment advice. Publication lags vary by series — the dashboard always shows
the period each value refers to, because "latest" for annual GDP and for the
rand mean very different dates.

## Licence

Code is MIT licensed. Data remains under each provider's terms as listed above.
