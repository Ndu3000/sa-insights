# Deployment

**Live static mirror:** <https://ndu3000.github.io/sa-insights/>
**Repository:** <https://github.com/Ndu3000/sa-insights>

This project has two deployment modes, serving different purposes. Read the
difference before assuming the public site is live data — it is not.

| Mode | What it is | Data freshness |
|------|-----------|----------------|
| **Static mirror** (GitHub Pages) | A rendered snapshot of the dashboard | Frozen at the last commit |
| **Containerised app** (Docker) | The real application with a live warehouse | Refreshed daily, downloads on demand |

The Pages site exists because it is free and public. It cannot satisfy the
requirements around live data, live downloads, customisation or model tracking —
those need the container.

---

## Containerised deployment

### What runs

```
┌─────────────┐   ┌──────────────┐   ┌─────────────┐
│  app        │   │  worker      │   │  mlflow     │
│  Streamlit  │   │  APScheduler │   │  tracking   │
│  :8501      │   │  05:15 UTC   │   │  :5000      │
└──────┬──────┘   └──────┬───────┘   └──────┬──────┘
       │                 │                  │
       └────────┬────────┴──────────────────┘
                ▼
        /data/sa_insights.duckdb
```

`app` serves the UI and generates downloads. `worker` owns the refresh schedule.
They are separate so a long foundation-model refresh cannot block the UI and a
crash in one does not take down the other. `mlflow` starts by default — tracking
is part of the product, not an optional extra.

### Quick start

```bash
# full stack: app + worker + MLflow, with Chronos-Bolt
HF_TOKEN=hf_xxx docker compose up -d --build

# lean variant without the foundation model (faster build, smaller image)
SA_INSIGHTS_FM=0 docker compose up -d --build
```

Then:
- Dashboard — <http://localhost:8501>
- MLflow — <http://localhost:5000>

On first start the app detects an empty warehouse and runs an initial refresh
before rendering, so it never opens on a blank page.

### Why DuckDB and not Postgres

The workload is analytical time series: 34 indicators, ~8,000 observations, and
aggregate reads for charts. DuckDB is built for exactly that and needs no server
process. The entire database is one file, which makes backup a `cp` and local
development a path. Postgres would add a service to run, secure and upgrade for
no benefit at this size.

If the project grows to multiple concurrent writers or needs row-level security,
Postgres becomes the right answer — the `store.py` interface is narrow enough
that the swap is contained.

### Why uv

`pyproject.toml` + `uv.lock` gives reproducible installs, and `uv sync --frozen`
is far faster than `pip install -r`. In Docker this matters: the dependency layer
is cached and only rebuilt when the lockfile changes.

### Why the foundation model is optional

Chronos-Bolt adds torch, which adds ~200 MB on top of a ~300 MB base. It lives
behind the `fm` extra and the `WITH_FM` build arg. When absent, `chronos_fm`
degrades to a no-op and `candidates()` simply omits it from the model pool — the
classical models still produce every forecast. Set `SA_INSIGHTS_FM=0` to skip it
at runtime even when installed.

---

## Serverless refresh on Modal

GitHub Actions cannot run on this account (see [Why the Actions workflow does
not run](#why-the-actions-workflow-does-not-run)), so the daily refresh runs on
Modal instead. Modal needs only a `git push` credential, which works
independently of the Actions billing lock.

```bash
uv sync --extra deploy                    # installs the modal CLI
uv run modal setup                        # interactive login
uv run modal secret create github-token GITHUB_TOKEN=<fine-grained PAT>
uv run modal deploy modal_app.py          # cron + optional dashboard
uv run modal run modal_app.py::refresh_and_publish   # trigger now
```

The PAT needs **Contents: write** on the repository. It is read from the Modal
secret at runtime and never baked into an image layer.

- `refresh_and_publish` — cron at 05:15 UTC (07:15 SAST). Fetches, forecasts,
  renders every artefact and pushes to `main`, after which Pages serves the new
  snapshot.
- `dashboard` — optional public Streamlit URL. Idle containers cost nothing; the
  first visit after idle pays a cold start (torch import + Chronos weights).

The DuckDB warehouse lives on a Modal Volume (`sa-insights-data`), so it
survives redeploys — unlike a Hugging Face Space, whose disk resets on restart.

MLflow is deliberately left unconfigured on Modal: without a tracking URI the
tracer degrades to a no-op. Setting one would make the dashboard advertise a UI
URL that no public host can serve, which is the dead-link bug fixed earlier.

## Model tracking (MLflow)

Tracking is **gated on `MLFLOW_TRACKING_URI`**. With no URI set, every tracking
call is a cheap no-op and the app behaves identically — so a laptop run needs no
server.

When enabled, each series refresh logs one run:

| Logged | Contents |
|--------|----------|
| params | series key, horizon, confidence, history window, observation count, chosen model and family |
| metrics | MASE, naive MASE, skill over naive, forecast endpoint, last actual |
| artifact | `leaderboard.json` — every candidate model and its score |
| tags | whether the foundation model was involved |

That answers "which model won CPI in October, against what competition, and by
how much" months later. The run id is stored on `forecast_runs.mlflow_run_id`, so
the database and MLflow cross-reference.

**Why not `mlflow-skinny`:** MLflow 3 deprecated the filesystem tracking backend
in favour of a database backend, and serving the UI needs flask. The full
`mlflow` package is therefore a core dependency. This is the one place the
dependency list is heavier than a read-only app strictly needs.

**Windows note:** MLflow's client logs emoji progress banners. The default
Windows console encoding (cp1252) cannot encode them, and the resulting
`UnicodeEncodeError` was being swallowed by the SDK and surfaced as a tracking
failure — silently losing every run after the first. `tracking.py` now
reconfigures stdout/stderr to UTF-8 with replacement, and logs failures per
series rather than disabling tracking globally.

---

## Data freshness

The refresh pipeline is the single path that makes data current:

```
sa_insights.refresh
   ↓  sources.py — SARB, IMF DataMapper, World Bank, FRED
   ↓  DuckDB observations
   ↓  forecast.py — Chronos-Bolt + classical, rolling-origin backtest
   ↓  DuckDB forecast_runs + forecasts, MLflow tracking
```

Triggered three ways:
- **Scheduled** — `worker` container, `REFRESH_CRON` (default `15 5 * * *` UTC)
- **On demand** — the "Refresh all data" button in the sidebar
- **CLI** — `python -m sa_insights.cli refresh`

`store.prune_forecasts()` retains the 10 most recent runs per series, so the
database does not grow without bound.

### Downloads are generated at request time

Earlier versions committed a pre-built Power BI zip and Ossie YAML, which meant
a download reflected whenever the build last ran. `sa_insights/downloads.py` now
builds every artefact from DuckDB on demand:

- Power BI bundle (star schema CSVs + TMSL model + DAX)
- Apache Ossie YAML and JSON, **validated against the official JSON Schema
  before being served**
- observations CSV, forecasts CSV

Cache keys derive from the last ingest timestamp, so new data invalidates them
immediately rather than waiting for a TTL.

---

## Local development

```bash
uv sync --extra dev          # core + test tooling
uv sync --extra fm           # add Chronos-Bolt
uv run streamlit run app.py
```

Useful commands:

```bash
uv run python -m sa_insights.cli doctor      # dependency and warehouse status
uv run python -m sa_insights.cli collect     # fetch upstream data only
uv run python -m sa_insights.cli forecast    # refit models only
uv run python -m sa_insights.cli export --out dist
uv run python -m sa_insights.scheduler --once
```

---

## Configuration

| Variable | Default | Effect |
|----------|---------|--------|
| `SA_INSIGHTS_FM` | `1` | `0` excludes the foundation model from the pool |
| `HF_TOKEN` | — | Read-only Hugging Face token; raises rate limits for the model download |
| `DATA_DIR` | `./data` | Where the DuckDB file lives |
| `SA_DB_PATH` | `$DATA_DIR/sa_insights.duckdb` | Explicit database path |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | SDK write target — Docker service name, not browser-reachable |
| `MLFLOW_UI_URL` | `http://localhost:5000` | Browser-facing MLflow link shown in the sidebar |
| `MLFLOW_EXPERIMENT` | `sa-insights-forecasting` | Experiment name |
| `REFRESH_CRON` | `15 5 * * *` | Worker schedule, UTC |
| `APP_PORT` / `MLFLOW_PORT` | `8501` / `5000` | Host port mapping |

---

## Static mirror (GitHub Pages)

Published from `docs/` on `main` via branch-based Pages.

**This mode does not refresh itself.** The numbers are whatever was last
committed. The site header states the snapshot time; treat it as a demo, not a
data source.

To update it, generate and commit:

```bash
uv run python -m sa_insights.cli export --out dist   # artefacts
uv run python scripts/export_pages.py                # renders docs/index.html
git add -A && git commit -m "Refresh snapshot" && git push
```

### Why the Actions workflow does not run

`.github/workflows/pages.yml` is complete and correct, but jobs are refused at
dispatch on this account:

> The job was not started because your account is locked due to a billing issue.

That is an account-level block, not a workflow defect. Once cleared, switch
Pages to **GitHub Actions** and the daily refresh runs unattended. Until then,
the container is the way to get live data.

---

## Known gaps

**No authentication.** Saved dashboards are namespaced by a per-session token in
`st.session_state`. That gives each browser its own space but is *not* auth:
there is no login, no protection between users, and a new browser starts empty.
Add `streamlit-authenticator` or an OAuth proxy before exposing saved dashboards
beyond a trusted audience.

**Single writer.** DuckDB permits one writing process. `app` and `worker` share
the volume safely today because the app only writes when a user presses
"Refresh all data" — but two simultaneous refreshes will conflict. If that
becomes likely, move refreshes behind a lock or move the store to Postgres.

**Stats SA has no usable public API.** Their SDMX endpoints return an Incapsula
bot-wall. CPI and PPI reach the dashboard via SARB's republishing; Q2 2026 GDP
specifically is not obtainable from any open API tested.

---

## Troubleshooting

**Container exits immediately** — check `docker compose logs app`. The most
common cause is a failed initial refresh (upstream outage); the app logs the
error and exits rather than serving an empty dashboard.

**Downloads 404 on the static site** — `scripts/export_pages.py` must run before
committing `docs/`; it copies the semantic documents into `docs/semantic/`
because Pages serves only that directory.

**MLflow UI unreachable** — the `tracking` profile must be active:
`docker compose --profile tracking up -d`.

**Chronos download fails** — Hugging Face rate-limits unauthenticated pulls. Set
`HF_TOKEN` to a read-only token. Never commit it; pass it through the
environment or a CI secret.
