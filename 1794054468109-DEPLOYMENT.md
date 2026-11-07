# Deployment

## How the live site stays current

The published site is static, but it is not a frozen snapshot. GitHub Actions
re-fetches every source and rebuilds the site on a schedule:

```
05:15 UTC daily
   ↓
scripts/collect.py            ← pulls 32 indicators from SARB, WB, IMF, FRED
   ↓
scripts/check_freshness.py    ← fails the build if a critical series goes stale
   ↓
scripts/export_forecasts.py   ← runs Chronos-Bolt + classical models
   ↓
scripts/build_semantic_model.py
scripts/export_powerbi.py
   ↓
scripts/export_pages.py       ← renders docs/index.html
   ↓
GitHub Pages
```

Everything is recomputed from upstream APIs on every run. Nothing is scraped,
and no numbers are committed by hand.

### Why the foundation model runs in CI, not the browser

Chronos-Bolt is a 48M-parameter network. Running it client-side would mean
shipping ~191 MB of weights so WebAssembly could execute hundreds of forward
passes during a backtest. Instead it runs once in the workflow and the site
ships the resulting numbers — about 29 KB of JSON.

### Why freshness is a build gate

An earlier iteration silently published policy rates and inflation that were
over 600 days old, because the DBnomics IMF mirror had stopped updating while
still returning valid-looking data. `check_freshness.py` now fails the build if
any *critical* indicator (CPI, policy rate, prime rate, bond yield, USD/ZAR)
exceeds its lag threshold. Non-critical breaches are reported but do not block a
deploy — an annual World Bank statistic is legitimately a year behind.

## One-time setup

### 1. Create the repository

```bash
gh auth login          # or sign in via the browser and use a token
gh repo create sa-insights --public --source=. --remote=origin --push
```

Without `gh`, create the repo in the web UI, then:

```bash
git remote add origin https://github.com/<user>/sa-insights.git
git push -u origin main
```

### 2. Enable GitHub Pages

Repository → **Settings** → **Pages**:

| Setting | Value |
|---------|-------|
| Source | *GitHub Actions* |

Do **not** choose "Deploy from a branch". The workflow publishes an artifact,
and a branch-based source will fight it.

### 3. Trigger the first deploy

Actions → **Refresh data and deploy Pages** → **Run workflow**.

The first run takes 5–10 minutes: it downloads torch, the Chronos weights and
every dataset. Later runs reuse the pip cache.

## Local development

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows; .venv/bin/activate on Unix
pip install -r requirements-fm.txt   # or requirements.txt for no foundation model
streamlit run app.py
```

To rebuild the static site without leaving the machine:

```bash
python scripts/collect.py
python scripts/export_forecasts.py
python scripts/build_semantic_model.py
python scripts/export_powerbi.py
python scripts/export_pages.py
```

## Configuration

| Variable | Default | Effect |
|----------|---------|--------|
| `SA_INSIGHTS_FM` | `1` | Set to `0` to exclude the foundation model from the model pool. Useful for fast CI runs or machines without torch. |

## What gets committed

| Path | Committed | Why |
|------|-----------|-----|
| `sa_insights/`, `scripts/`, `app.py` | Yes | Source |
| `docs/` | Yes | The deployed site; Pages serves this directory |
| `semantic/` | Yes | The semantic model is a deliverable, not a build artefact |
| `data/cache/` | No | Regenerated from upstream; committing it would go stale silently |
| `build/` | No | Intermediate Power BI CSVs |
| `.venv/` | No | Environment |

## Troubleshooting

**Build fails on `check_freshness.py`** — an upstream API has changed or gone
quiet. Run `python scripts/check_freshness.py --warn` locally to see the report
without failing.

**Chronos download fails in CI** — Hugging Face rate-limits unauthenticated
downloads. Add a `HF_TOKEN` repository secret and pass it as an env var to the
`export_forecasts.py` step.

**Pages shows a 404 for downloads** — the export script must run before the
upload. Check the build log for `export_pages.py`.
