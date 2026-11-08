# Deployment

**Live static mirror:** <https://ndu3000.github.io/sa-insights/>
**Live dashboard:** <https://ndu3000--sa-insights-dashboard.us-east.modal.run>
**Chat API:** <https://ndu3000--sa-insights-chat-api.us-east.modal.run/docs>
**MLflow:** <https://ndu3000--sa-insights-mlflow-server.us-east.modal.run>
**Repository:** <https://github.com/Ndu3000/sa-insights>

There are three ways to run this: a public static mirror, a containerised stack,
and a serverless deployment on Modal.

| Mode | What it is | Data freshness | Cost |
|------|-----------|----------------|------|
| **Static mirror** (GitHub Pages) | A rendered snapshot of the dashboard | **Refreshed daily** by a Modal cron | Free |
| **Serverless** (Modal) | Refresh pipeline, public dashboard, chat API and MLflow | Live | ~$0.02/day |
| **Containerised** (Docker) | The full stack on your own machine | Refreshed daily | Your hardware |

The Pages site is no longer a frozen snapshot: a scheduled Modal job refreshes
the warehouse, re-renders every artefact and pushes the result, so the public
URL is current to within a day. What it cannot do is serve a live query or
generate a download on demand — that needs the dashboard or the container.

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

GitHub Actions cannot run on this account — jobs are refused at dispatch by an
account-level billing lock (see [Why the Actions workflow does not
run](#why-the-actions-workflow-does-not-run)). Modal runs the same pipeline on
its own infrastructure and needs only a `git push` credential, which works
independently of that lock.

### What is deployed

| Function | URL | Purpose |
|----------|-----|---------|
| `refresh_and_publish` | cron `15 5 * * *` UTC | fetch → forecast → render → push to `main` |
| `dashboard` | <https://ndu3000--sa-insights-dashboard.us-east.modal.run> | the Streamlit app, public |
| `mlflow_server` | <https://ndu3000--sa-insights-mlflow-server.us-east.modal.run> | MLflow UI, public |
| `chat_api` | <https://ndu3000--sa-insights-chat-api.us-east.modal.run> | grounded Q&A over the warehouse |

Two Volumes back the deployment:

| Volume | Mount | Holds |
|--------|-------|-------|
| `sa-insights-data` | `/data` | DuckDB warehouse, source cache, refresh state |
| `sa-insights-mlflow` | `/mlflow` | MLflow SQLite backend and artefacts |

They are separate because they have different retention needs: MLflow can be
discarded without losing data, the warehouse cannot.

### Deploying

```bash
uv sync --extra deploy                    # installs the modal CLI
uv run modal setup                        # browser login; writes ~/.modal.toml
uv run modal secret create github-token GITHUB_TOKEN=<PAT>
uv run modal deploy modal_app.py
uv run modal run modal_app.py::refresh_and_publish   # trigger now
```

The PAT needs **Contents: write** on the repository. It is injected from the
Modal secret at runtime and never baked into an image layer.

`modal setup` prints a token-flow URL; opening it in a browser already signed in
to Modal completes the exchange. The token is written to `~/.modal.toml`.

**On Windows**, run the CLI with `PYTHONUTF8=1`. It prints `✓` on success, which
the default cp1252 console cannot encode — the deploy has already succeeded by
then, so the failure is cosmetic but alarming.

### Measured behaviour

| | Value |
|---|---|
| Full refresh (34 series incl. Chronos-Bolt) | **205 s** |
| Dashboard cold start | **~8 s** |
| MLflow cold start | ~20 s (after the first build) |
| Image build (cold) | ~4 min; ~70 s with layers cached |
| Image size | 2.2 GB (torch CPU is ~200 MB of it) |
| Daily cost | **~$0.02** |

At that rate the free monthly credit covers the cron many times over.

### Why the foundation model runs here and not in the browser

Chronos-Bolt is a 48M-parameter network. Shipping ~191 MB of weights so
WebAssembly could execute hundreds of forward passes is untenable on mobile, so
the model runs at refresh time and the site ships only the resulting numbers.

### Why MLflow is a Modal function

MLflow 3 deprecated the filesystem backend, so it needs a SQLite file — which
needs storage that outlives a container. A Volume provides that, and a Volume
does not exist without an app, so the server lives here beside the refresh that
writes to it.

The dashboard does **not** hardcode its URL. `_mlflow_env()` resolves it at
startup with `Function.from_name("sa-insights", "mlflow_server").get_web_url()`
and exports both `MLFLOW_TRACKING_URI` and `MLFLOW_UI_URL`. That keeps
`tracking.py`'s two-variable split intact — the reason those are separate is
that in Docker the SDK target is the compose service name and only the UI URL is
browser-reachable — while avoiding a URL baked into an image that goes stale on
the next deploy. If resolution fails, tracking is left unconfigured rather than
pointed at a dead link.

Two things went wrong building it, both worth knowing:

- **Naming the function `mlflow` shadowed the `mlflow` package** inside this
  module. Modal resolves remote functions by name, so it surfaced as
  `module 'modal_app' has no attribute 'mlflow'` in a crash loop. It is
  `mlflow_server`.
- **A stale module shadowed every local edit.** The image's `WORKDIR` is `/repo`,
  which already contains a *cloned* copy of `modal_app.py`. Modal imported that
  clone rather than the file being deployed, so local edits were silently
  ignored and the error message named a function that plainly existed. The
  deployed file is now copied over the clone at both import paths, and
  re-asserted after every `git reset` in `_sync_repo()`.
- **Tracking was silently disabled by import order.** `tracking.py` reads
  `MLFLOW_TRACKING_URI` into a module constant *at import time* and gates every
  call on it. Setting the variable after `from sa_insights import refresh` has no
  effect whatsoever: the first run logged `tracking at <url>` while MLflow stayed
  completely empty. `_apply_mlflow_env()` now runs before any `sa_insights`
  import in that process.

### Why the publish push is retried

A refresh takes ~25 minutes. Syncing the checkout only at the start left a
window the full width of the run: any commit landing on `main` meanwhile made
the final push non-fast-forward, and the run failed *after* doing all the
expensive work. The first live run did exactly that — fetched, forecast,
rendered and committed successfully, then lost the push to a concurrent host
commit. `_publish_with_retry()` now syncs immediately before rendering
(narrowing the window to the seconds a render takes) and retries a rejected push
up to three times. Only push rejections are retried; a render error would fail
identically every time and would just burn 3× the compute.

### Hugging Face token

Chronos weights download anonymously, which logs a rate-limit warning on every
cold start. A read-only `HF_TOKEN` added as a Modal secret removes it; it is not
required.

## Chatbot

Ask questions in plain English and get answers built from the warehouse, with
citations. Live at <https://ndu3000--sa-insights-chat-api.us-east.modal.run/docs>.

```bash
curl -X POST https://ndu3000--sa-insights-chat-api.us-east.modal.run/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is CPI inflation now?","session_id":"demo"}'
```

### The one design decision that matters

**Numbers come from tools, never from retrieval.** The language model is an
optional final step that rewrites prose; it never supplies a figure. Everything
that matters — which series are relevant, what their values are, whether the
answer is defensible — is decided by deterministic code:

```
question ─► graph (what is relevant?) ─► tools (what are the numbers?) ─► gate
                                                                          │
                                            ┌─────────────────────────────┤
                                            ▼                             ▼
                                       verified answer            refusal / clarify
```

`answers.verify_numbers` scans the prose for figures and rejects the answer if
any number does not appear in a tool result. That converts "the model usually
does not hallucinate" into "an invented statistic cannot reach a user". A
fabricated `7.9%` CPI reading is caught and replaced with the verified citations.

### The four layers

| Module | Job |
|--------|-----|
| `chat/graph.py` | Typed property graph over the Apache Ossie model: 33 series, 5 cross-series metrics, 8 categories, 7 themes |
| `chat/tools.py` | Deterministic DuckDB queries, with a read-only SQL guard |
| `chat/answers.py` | Typed `Answer`, citations, and the refusal gate |
| `chat/agent.py` | Routing, follow-ups, clarification policy, prose composition |

### Why a graph, and why not a graph database

The questions are relational: "how has the economy improved" is a set of series
*plus* the definitions connecting them — a real policy rate is the repo rate
relative to inflation. Vector search retrieves similar text; it cannot tell you
that `policy_rate` and `cpi_yoy` combine into `real_policy_rate`.

But 33 series and ~175 metrics fit in memory with room to spare, so Neo4j (which
the `knowledge_graph_rag_citations` template uses) would add a service to
traverse a few hundred nodes. The structure is real; the infrastructure is not
warranted. `Graph.expand` does the bounded multi-hop traversal that matters.

### Theme nodes

"How has the economy improved?" matches no series label — not one contains the
word "economy" — so lexical scoring alone returned *nothing* for the most
natural question possible. Seven theme nodes (`economy`, `cost_of_living`,
`job_market`, `interest_rates`, `currency`, `external`, `fiscal`) give everyday
phrasing somewhere to land, with edges weighted so a theme's most
representative series ranks first.

### Grounding: vocabulary, not a score threshold

Deciding whether a question is *about this data* was the hardest part. A score
threshold failed in both directions: "who won the 2010 world cup" matched a GDP
series because 2010 sits inside its data range, while "is inflation getting
better?" scored **below** any sensible threshold because "inflation" is common
across the corpus.

The fix is conceptual rather than numeric: grounding tests the question against
each node's **label and synonyms** — curated vocabulary — instead of its prose
description. Words in a description are incidental; words in a label are
deliberate.

### Clarification

Asking when the answer is already determinable is the most annoying failure mode
a data chatbot has, so all three conditions must hold:

1. vague evaluative wording ("improved", "doing")
2. no window named by the user
3. the seed was a **theme**, not a series

So "How has the economy improved?" asks a follow-up, while "is inflation getting
better?" answers — CPI is the obvious subject even though the wording is vague.

### Bugs found while building this

Each is now a regression test in `tests/test_chat.py` (24 tests):

- **ISO dates parsed as numbers.** `2026-09-01` became 2026, 09, 01, and the
  09 was reported as a fabricated figure — rejecting every correct answer that
  named a period.
- **Percent change on rate series.** CPI moving 3.4% → 4.5% was rendered as
  "+32%": arithmetically true, and it makes inflation look like it tripled.
  Rate series now report percentage points with `percent: null`.
- **Off-topic questions answered from a data range.** See grounding above.
- **The headline-readings fallback fired on ungrounded questions**, so "what is
  the meaning of life" answered with GDP figures.
- **Follow-ups inherited the wrong subject.** "And unemployment?" reused the
  previous answer's series instead of resolving its own.
- **Clarification routed through the gate**, which returned a normal answer and
  produced the "something went wrong" branch.

### Attribution

The typed-answer pattern — validated object with citations, confidence and an
explicit `answered` flag, refusing before calling the model — is adapted from
[`agentic_typed_rag_pydanticai`](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/rag_tutorials/agentic_typed_rag_pydanticai)
in `Shubhamsaboo/awesome-llm-apps` (Apache-2.0). The gate here is arithmetic
rather than cosine similarity, because the failure that matters in a numeric
domain is not "cited the wrong document" but "stated a number nobody computed".

### Running the tests

```bash
uv run --extra dev python -m pytest tests/test_chat.py -q
```

The suite asserts policies, not prose: answered when answerable, clarified when
genuinely open, refused when off-topic, and never a figure that fails
verification.

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

**This mode is refreshed by the Modal cron**, not by GitHub Actions. The job at
05:15 UTC re-renders `docs/` and pushes; Pages then rebuilds itself. The site
header states the snapshot time, so staleness is always visible rather than
assumed.

To update it by hand:

```bash
uv run python -m sa_insights.cli export --out dist   # artefacts
uv run python scripts/export_pages.py                # renders docs/index.html
git add -A && git commit -m "Refresh snapshot" && git push
```

### Why the Actions workflow does not run

`.github/workflows/pages.yml` is complete and correct, but jobs are refused at
dispatch on this account:

> The job was not started because your account is locked due to a billing issue.

That is an account-level block, not a workflow defect. The same schedule is
served by `refresh_and_publish` on Modal instead, so the daily refresh happens
either way. If the lock is ever cleared, either path works — they produce
identical output.

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

**MLflow UI unreachable** — the `mlflow` service starts by default (it has no
profile); check `docker compose ps mlflow` for health. On Modal, MLflow cold-starts
in ~20s, so the first request after idle can be slow.

**Chronos download fails** — Hugging Face rate-limits unauthenticated pulls. Set
`HF_TOKEN` to a read-only token. Never commit it; pass it through the
environment or a CI secret.
