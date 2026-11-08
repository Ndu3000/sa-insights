"""Modal deployment: serverless daily refresh, plus an optional live dashboard.

Why Modal instead of GitHub Actions
-----------------------------------
Actions on this account is locked by an unpaid balance, so the schedule in
``.github/workflows/pages.yml`` never starts. Modal runs the same pipeline on
its own infrastructure and only needs a ``git push`` credential, which works
independently of the Actions billing lock.

What runs
---------
``refresh_and_publish`` (cron, 05:15 UTC)
    fetch -> warehouse -> forecasts incl. Chronos-Bolt -> render artefacts
    -> commit + push to ``main``. GitHub Pages then serves the new snapshot.

``dashboard`` (web endpoint, optional)
    the Streamlit app for interactive exploration. Idle containers cost
    nothing; the first visit after idle pays a cold start because the container
    must import torch and load the Chronos weights.

Deploy
------
    pip install modal
    modal setup                       # interactive login, token stored locally
    modal secret create github-token GITHUB_TOKEN=<fine-grained PAT>
    modal deploy modal_app.py

Trigger a refresh immediately (no waiting for the cron):
    modal run modal_app.py::refresh_and_publish

The PAT needs ``Contents: write`` on the repository; the token is read from the
Modal secret at runtime and never written into an image layer.
"""

from __future__ import annotations

import shlex
import subprocess

import modal

REPO_URL = "https://github.com/Ndu3000/sa-insights.git"
REPO_PATH = "/repo"
DATA_PATH = "/data"
VENV = "/opt/venv"

CRON_UTC = "15 5 * * *"  # 07:15 SAST, after the overnight SARB / Stats SA updates

app = modal.App("sa-insights")

# The DuckDB warehouse, cache and refresh state live here. Unlike Hugging Face
# Spaces (whose disk resets on every restart) a Modal Volume survives redeploys.
data_vol = modal.Volume.from_name("sa-insights-data", create_if_missing=True)


# --------------------------------------------------------------------------- #
# Image
# --------------------------------------------------------------------------- #
# The checkout is cloned at build time rather than shipped as a local mount:
# the deployed cron must not depend on this machine being online. Each run
# resets the checkout to origin/main, so a stale bake only wastes a fetch.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "ca-certificates")
    .run_commands(f"git clone --branch main {REPO_URL} {REPO_PATH}")
    .workdir(REPO_PATH)
    # uv.lock pins every transitive version, so builds are reproducible. The
    # `fm` extra installs CPU torch + chronos-forecasting via [tool.uv.sources].
    #
    # UV_PROJECT_ENVIRONMENT is passed inline rather than via .env(): the venv
    # path must be correct at *build* time, and an image-level env var is only
    # guaranteed to be present at runtime.
    #
    # --no-install-project mirrors the Dockerfile: the package is imported from
    # the checkout's own working directory, so installing it into site-packages
    # would duplicate the code and make a stale copy possible.
    .run_commands(
        "command -v uv >/dev/null 2>&1 || pip install --quiet uv",
        "UV_PROJECT_ENVIRONMENT="
        + VENV
        + " UV_LINK_MODE=copy uv sync --frozen --no-install-project --no-dev --extra fm",
    )
    .env(
        {
            "PATH": f"{VENV}/bin:/usr/local/bin:/usr/bin:/bin",
            "VIRTUAL_ENV": VENV,
            "PYTHONUNBUFFERED": "1",
            "DATA_DIR": DATA_PATH,
            "SA_PUBLISH_REPO": REPO_PATH,
            "SA_INSIGHTS_FM": "1",
            # MLflow is deliberately left unconfigured. Without a tracking URI
            # the tracer degrades to a no-op; setting one would make the
            # dashboard advertise a UI URL that no public host can serve, which
            # is exactly the dead-link bug we already fixed once.
        }
    )
    # Build-time assertion. A missing torch/chronos previously showed up only as
    # a soft "foundation model unavailable" notice at runtime and stayed hidden
    # for days, so fail the build instead.
    #
    # Written as a single literal string on purpose: the closing quote must be
    # part of the command, and building it by concatenating pieces silently
    # dropped it once already (sh: Unterminated quoted string).
    .run_commands(
        f"{VENV}/bin/python -c "
        + shlex.quote(
            "import torch, chronos, duckdb, numpy; "
            "print('imports ok, torch', torch.__version__)"
        )
    )
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sync_repo() -> None:
    """Reset the baked checkout to origin/main.

    The image clones once at build time, so the working tree can lag what the
    host has pushed since. Resetting keeps the code being run — and the commit
    base — current.
    """
    # A build-time checkout is owned by root; git still needs the directory
    # marked safe even when running as root inside the container.
    subprocess.run(
        ["git", "config", "--global", "--add", "safe.directory", REPO_PATH],
        check=False,
        capture_output=True,
    )
    for args in (["fetch", "origin", "main"], ["reset", "--hard", "origin/main"]):
        proc = subprocess.run(
            ["git", *args], cwd=REPO_PATH, capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git {args[0]} failed: {proc.stderr.strip()}")


def _publish_with_retry(attempts: int = 3) -> bool:
    """Publish, re-basing onto origin/main and retrying if the push is rejected.

    A refresh takes roughly 25 minutes, so syncing only at the start of the run
    leaves a window the width of the whole refresh: any commit that lands on
    main meanwhile makes the final push non-fast-forward and the run fails
    after having done all the expensive work. Syncing immediately before
    rendering narrows that window to the few seconds the render takes, and the
    retry covers what remains.
    """
    from sa_insights import publish

    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        _sync_repo()
        try:
            return publish.publish()
        except RuntimeError as exc:
            last = exc
            message = str(exc)
            # Only a rejected push is worth retrying — it is the one failure
            # caused by outside activity. A render error would fail identically
            # every time and retrying would just burn 3x the compute.
            if "git push failed" not in message:
                raise
            print(
                f"publish attempt {attempt}/{attempts} rejected "
                f"({message.splitlines()[0]}); retrying on a fresh base",
                flush=True,
            )
    if last is not None:
        raise last
    return False


# --------------------------------------------------------------------------- #
# Scheduled refresh
# --------------------------------------------------------------------------- #
@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    # The token is only ever read from the secret into the process environment;
    # it is never baked into an image layer or printed.
    secrets=[modal.Secret.from_name("github-token")],
    timeout=60 * 60,
    schedule=modal.Cron(CRON_UTC),
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
)
def refresh_and_publish() -> dict:
    """Refresh every series, then publish the regenerated site to Pages."""
    from sa_insights import refresh, store

    # Sync first so the refresh runs the current code, not whatever was baked
    # into the image when it was last built.
    _sync_repo()

    report = refresh.refresh_all(force=True)

    # Persist the warehouse onto the Volume. Modal only keeps a container's
    # writes after an explicit commit, so skipping this would discard a day's
    # ingest while looking perfectly successful in the logs.
    data_vol.commit()

    changed = _publish_with_retry()

    summary = {
        "series_loaded": report.series_loaded,
        "series_total": report.series_total,
        "observations": report.observations,
        "forecasts_written": report.forecasts_written,
        "duration_seconds": round(report.duration_seconds, 1),
        "failed": len(report.failed),
        "publish_changed": changed,
        "warehouse": str(store.DB_PATH),
    }
    print(f"refresh complete: {summary}")
    return summary


# --------------------------------------------------------------------------- #
# Optional live dashboard
# --------------------------------------------------------------------------- #
@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    timeout=60 * 30,
    # Keep a warm container briefly after the last visitor: long enough that a
    # user clicking around does not re-pay the cold start, short enough that an
    # idle app still scales to zero and costs nothing.
    scaledown_window=300,
    max_containers=2,
)
@modal.web_server(port=8501, startup_timeout=300)
def dashboard() -> None:
    """Serve the Streamlit app publicly on ``*.modal.run``."""
    # Modal caches build layers, so the baked checkout can lag origin/main.
    # Refreshing here keeps the live app on the same code as the published site.
    _sync_repo()

    subprocess.Popen(
        [
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            "8501",
            "--server.address",
            "0.0.0.0",
            "--server.headless",
            "true",
            # Modal terminates TLS and proxies from a different origin; both
            # flags are required or the app loads but every widget call fails.
            "--server.enableCORS",
            "false",
            "--server.enableXsrfProtection",
            "false",
        ],
        cwd=REPO_PATH,
    )
