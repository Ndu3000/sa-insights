"""Modal deployment: serverless refresh, dashboard, MLflow and the chatbot.

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

``mlflow_server`` (web endpoint)
    public MLflow UI, so run links in the dashboard resolve.

``chat_api`` (web endpoint)
    the grounded question-answering API over the warehouse
    (``sa_insights.chat.api``).

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

import os
import shlex
import subprocess

import modal

REPO_URL = "https://github.com/Ndu3000/sa-insights.git"
REPO_PATH = "/repo"
DATA_PATH = "/data"
MLFLOW_PATH = "/mlflow"
VENV = "/opt/venv"

CRON_UTC = "15 5 * * *"  # 07:15 SAST, after the overnight SARB / Stats SA updates

app = modal.App("sa-insights")

# The DuckDB warehouse, cache and refresh state live here. Unlike Hugging Face
# Spaces (whose disk resets on every restart) a Modal Volume survives redeploys.
data_vol = modal.Volume.from_name("sa-insights-data", create_if_missing=True)

# MLflow's SQLite backend and artefacts. Separate from the warehouse: the two
# have different write patterns and different retention needs, and MLflow is the
# one thing here that can be discarded without losing data.
mlflow_vol = modal.Volume.from_name("sa-insights-mlflow", create_if_missing=True)


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
    # Ship this file into the image at both paths it can be imported from.
    #
    # Modal resolves a deployed function by importing the module by name. The
    # image's WORKDIR is /repo, which contains the *cloned* copy of this same
    # file — so edits made locally and not yet pushed were silently shadowed by
    # the older clone, surfacing as "module 'modal_app' has no attribute ..."
    # in a crash loop. Copying the deployed file over the clone makes the file
    # that was deployed authoritative, and /root covers the other sys.path entry.
    .add_local_file(__file__, f"{REPO_PATH}/modal_app.py", copy=True)
    .add_local_file(__file__, "/root/modal_app.py", copy=True)
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
            # Both MLflow URLs are filled in at runtime by _mlflow_env(), because
            # they depend on the URL Modal assigns to the mlflow function. A
            # value baked in here would be wrong on the next deploy (Modal
            # version-suffixes some URLs) and would recreate the dead-link bug
            # that separating TRACKING_URI from UI_URL was meant to fix.
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

    ``modal_app.py`` itself is exempt: the image ships the deployed copy over the
    clone (see the image definition), so including it here would let a reset
    quietly restore an older version of this file mid-run.
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

    # Re-assert the deployed copy after the reset overwrites the working tree.
    #
    # The path is resolved before comparing: __file__ can be relative or carry a
    # different case on Windows, and comparing it directly to the destination
    # produced "are the same file" — which silently skipped the restore.
    try:
        import shutil

        source = os.path.realpath(__file__)
        target = os.path.realpath(f"{REPO_PATH}/modal_app.py")
        if source != target:
            shutil.copyfile(source, target)
    except OSError as exc:
        print(f"warning: could not restore modal_app.py after reset ({exc})")

    # Record which commit is now checked out, and say so in the logs.
    #
    # A warm container keeps running whatever this reset produced the moment it
    # started, so a deploy can look successful while the public URL still serves
    # old code. Without the SHA in the log there is no way to distinguish
    # "running the new build" from "running a build from two hours ago" — the
    # same configured-versus-working confusion that once hid missing MLflow runs.
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_PATH,
        capture_output=True,
        text=True,
    )
    if head.returncode == 0:
        print(f"repo: synced to {head.stdout.strip()}", flush=True)


def _mlflow_env() -> dict[str, str]:
    """Resolve MLflow's public URL and configure tracking for this process.

    Looking the URL up at call time (rather than baking it into the image) means
    a redeploy that changes the URL cannot leave the dashboard advertising a
    stale address. If the lookup fails, tracking is left unconfigured: the
    tracer degrades to a no-op, which is strictly better than a link that 404s.
    """
    url = None
    try:
        url = modal.Function.from_name(app.name, "mlflow_server").get_web_url()
    except Exception as exc:  # noqa: BLE001 - tracking must never break a refresh
        print(f"mlflow: could not resolve public URL ({exc}); tracking disabled")

    if not url:
        return {}

    # The same public URL serves both roles here, which is the unusual case: in
    # Docker the SDK target is the compose service name and only the UI URL is
    # browser-reachable, so tracking.py keeps them as separate variables.
    return {
        "MLFLOW_TRACKING_URI": url,
        "MLFLOW_UI_URL": url,
        "MLFLOW_EXPERIMENT": "sa-insights-forecasting",
    }


def _apply_mlflow_env() -> None:
    """Export the resolved MLflow URLs into this process's environment."""
    env = _mlflow_env()
    if env:
        os.environ.update(env)
        print(f"mlflow: tracking at {env['MLFLOW_TRACKING_URI']}", flush=True)


# --------------------------------------------------------------------------- #
# MLflow tracking server
# --------------------------------------------------------------------------- #
@app.function(
    image=image,
    volumes={MLFLOW_PATH: mlflow_vol},
    timeout=60 * 30,
    # Tracking is only written to when a refresh runs, which is once a day. There
    # is nothing to keep warm for, so this scales fully to zero and costs
    # nothing while idle.
    scaledown_window=120,
    max_containers=1,
)
@modal.web_server(port=5000, startup_timeout=120)
def mlflow_server() -> None:
    """Serve the MLflow UI publicly so run links in the dashboard resolve.

    Named ``mlflow_server``, not ``mlflow``: a module-level function with that
    name shadows the ``mlflow`` package for every other function in this file,
    and Modal's runner resolves remote functions by name, so the collision
    surfaced as "module 'modal_app' has no attribute 'mlflow'" in a crash loop.
    """
    subprocess.Popen(
        [
            "mlflow",
            "server",
            "--host",
            "0.0.0.0",
            "--port",
            "5000",
            # SQLite, not the filesystem backend: MLflow 3 deprecated the
            # latter, and a single file on the Volume survives redeploys.
            "--backend-store-uri",
            f"sqlite:///{MLFLOW_PATH}/mlflow.db",
            "--default-artifact-root",
            f"{MLFLOW_PATH}/artifacts",
            # Modal terminates TLS and proxies from a different origin; without
            # these the UI loads but every API call fails its host check.
            "--allowed-hosts",
            "*",
            "--cors-allowed-origins",
            "*",
        ],
        cwd=REPO_PATH,
    )


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
    # Sync first so the refresh runs the current code, not whatever was baked
    # into the image when it was last built.
    _sync_repo()

    # This must happen before `sa_insights` is imported anywhere in this process.
    # tracking.py reads MLFLOW_TRACKING_URI into a module constant at import time
    # and gates every call on it, so a URI set afterwards has no effect at all -
    # the first run of this function logged "tracking at <url>" while MLflow
    # stayed empty, because the import below had already disabled it.
    _apply_mlflow_env()

    from sa_insights import refresh, store

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

    # Streamlit inherits this process's environment, so the sidebar advertises a
    # reachable MLflow link rather than an empty one.
    _apply_mlflow_env()

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


# --------------------------------------------------------------------------- #
# Chatbot API
# --------------------------------------------------------------------------- #
@app.function(
    image=image,
    # The chat reads the warehouse, so it needs the same Volume the refresh
    # writes. Without it every question would refuse: no data, nothing to say.
    volumes={DATA_PATH: data_vol},
    timeout=60 * 10,
    # Short window: each container holds a cached graph and session map, and a
    # question costs well under a second to answer, so there is nothing to keep
    # warm for long.
    scaledown_window=180,
    max_containers=4,
)
@modal.web_server(port=8000, startup_timeout=120)
def chat_api() -> None:
    """Serve the grounded Q&A API on ``*.modal.run``.

    No MLflow and no language-model secret: the answer path is fully
    deterministic, so this runs with no external credentials at all. The model,
    when enabled later, is optional by construction — ``agent.Agent`` produces
    the same verified answer without one.
    """
    _sync_repo()

    subprocess.Popen(
        [
            "uvicorn",
            "sa_insights.chat.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        cwd=REPO_PATH,
    )
