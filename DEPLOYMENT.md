# Deployment

**Live site:** <https://ndu3000.github.io/sa-insights/>
**Repository:** <https://github.com/Ndu3000/sa-insights>

## Current deployment mode: branch-based

The site is published straight from the `docs/` folder on `main`:

> Settings → Pages → Source: **Deploy from a branch** → `main` / `/docs`

This works because `docs/` is committed already built. It requires no Actions
runtime and has no billing dependency.

### Why not GitHub Actions (yet)

The workflow in `.github/workflows/pages.yml` is complete and correct, but it
**cannot run on this account**: jobs are refused at dispatch with

> The job was not started because your account is locked due to a billing issue.

That is an account-level block on Actions, not a workflow problem. Until it is
cleared, the automatic daily refresh cannot run. Everything below still
describes the intended design and will work as soon as Actions is unblocked.

### Consequence, stated plainly

In branch-based mode the numbers are **whatever was last committed**. The site
does not refresh itself. To update it:

```bash
python scripts/collect.py
python scripts/export_forecasts.py
python scripts/build_semantic_model.py
python scripts/export_powerbi.py
python scripts/export_pages.py
git add -A && git commit -m "Refresh data" && git push
```

Data is current as of the last such run; check the "N series · snapshot DATE"
line in the site header for the exact date.

### Switching to automatic refresh

Once Actions is unblocked:

1. Settings → Pages → Source: **GitHub Actions**
2. Actions → *Refresh data and deploy Pages* → **Run workflow**

From then on, `schedule` (05:15 UTC daily) keeps the site current with no
manual step. The workflow also gates on `check_freshness.py`, so a silently
frozen upstream fails the build instead of publishing stale numbers.

## How the pipeline works

```
scripts/collect.py            ← pulls 34 indicators from SARB, IMF, WB, FRED
   ↓
scripts/check_freshness.py    ← fails if a critical series goes stale
   ↓
scripts/export_forecasts.py   ← Chronos-Bolt + classical models, backtested
   ↓
scripts/build_semantic_model.py
scripts/export_powerbi.py
   ↓
scripts/export_pages.py       ← renders docs/index.html
   ↓
docs/ → GitHub Pages
```

Everything is recomputed from upstream APIs. Nothing is scraped, and no numbers
are edited by hand.

### Why the foundation model runs at build time, not in the browser

Chronos-Bolt is a 48M-parameter network. Client-side inference would mean
shipping ~191 MB of weights so WebAssembly could run hundreds of forward passes
during a backtest — untenable on the mobile connections this targets. Instead it
runs once during export and the site ships ~29 KB of numbers.

### Why freshness is a build gate

An early iteration silently published policy rates and inflation over 600 days
old, because the DBnomics IMF mirror had stopped updating while still returning
valid-looking data. `check_freshness.py` now fails the build if any critical
indicator (CPI, policy rate, prime rate, bond yield, USD/ZAR) exceeds its lag
threshold. Non-critical breaches are reported but do not block — an annual World
Bank statistic is legitimately a year behind.

## Local development

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows; .venv/bin/activate on Unix
pip install -r requirements-fm.txt   # or requirements.txt without the foundation model
streamlit run app.py
```

## Configuration

| Variable | Default | Effect |
|----------|---------|--------|
| `SA_INSIGHTS_FM` | `1` | Set to `0` to exclude the foundation model from the pool. Useful for fast runs or machines without torch. |

## What gets committed

| Path | Committed | Why |
|------|-----------|-----|
| `sa_insights/`, `scripts/`, `app.py` | Yes | Source |
| `docs/` | Yes | The deployed site — required for branch-based Pages |
| `semantic/` | Yes | The semantic model is a deliverable, not an artefact |
| `data/cache/` | No | Regenerated from upstream; committing it would go stale silently |
| `build/` | No | Intermediate Power BI CSVs |
| `.venv/` | No | Environment |

## Troubleshooting

**Actions job never starts** — account locked for billing. See above; use
branch-based Pages until resolved.

**`check_freshness.py` fails** — an upstream API changed or went quiet. Run
`python scripts/check_freshness.py --warn` to see the report without failing.

**Chronos download fails** — Hugging Face rate-limits unauthenticated pulls. Add
an `HF_TOKEN` secret and expose it to the `export_forecasts.py` step.

**Downloads 404** — `export_pages.py` must run before committing `docs/`; it
copies the semantic documents into `docs/semantic/` because Pages serves only
that directory.

