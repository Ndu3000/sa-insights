"""Optional MLflow tracking for forecast runs.

Design
------
Tracking is **gated**: with no ``MLFLOW_TRACKING_URI`` set, every function here
is a cheap no-op and the dashboard behaves identically. That keeps the app
runnable on a laptop or in a container without a tracking server, while a
deployment that does set the URI gets full lineage.

What we log per series per refresh:

* **params** — series, horizon, history window, observation count, chosen model
* **metrics** — MASE, naive MASE, skill over naive, forecast endpoint
* **artifacts** — the full leaderboard (every candidate model and its score)
* **tags** — whether the foundation model was involved

That is enough to answer "which model won CPI in October, against what
competition, and by how much" months later — which is the point of tracking.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

# Read once at import; a missing value disables tracking entirely.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "").strip()
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "sa-insights-forecasting")

_disabled_reason: str | None = None
if not TRACKING_URI:
    _disabled_reason = "MLFLOW_TRACKING_URI not set"

# MLflow's client logs progress banners containing emoji. On Windows the default
# console encoding is cp1252, which cannot encode them, and the resulting
# UnicodeEncodeError is swallowed by the SDK and reported as a tracking failure
# — a confusing way to lose an entire experiment. Reconfiguring stdout/stderr to
# UTF-8 with a replacement handler makes logging unable to break tracking.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

# MLflow also emits an advisory about optional tracing skills on every import.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

LOG = logging.getLogger("sa-insights.tracking")
# Keep MLflow's own logger quiet; errors are caught and surfaced by us instead.
logging.getLogger("mlflow").setLevel(logging.ERROR)


def enabled() -> bool:
    return bool(TRACKING_URI) and _disabled_reason is None


def status() -> dict:
    """Human-readable tracking state, surfaced in the dashboard sidebar."""
    return {
        "enabled": enabled(),
        "tracking_uri": TRACKING_URI or None,
        "experiment": EXPERIMENT if enabled() else None,
        "reason": _disabled_reason,
    }


def log_forecast_run(
    *,
    series_key: str,
    result: Any,
    window_obs: int | None,
    n_obs: int,
    horizon: int,
) -> str | None:
    """Log one forecast run. Returns the MLflow run id, or None if disabled.

    Never raises: a tracking outage must not break a data refresh.
    """
    if not enabled():
        return None

    try:
        import mlflow

        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT)

        with mlflow.start_run(run_name=f"{series_key}-h{horizon}") as run:
            mlflow.log_params(
                {
                    "series_key": series_key,
                    "horizon": horizon,
                    "confidence": getattr(result, "confidence", None),
                    "window_obs": window_obs,
                    "n_obs": n_obs,
                    "frequency": getattr(result, "frequency", None),
                    "model": getattr(result, "model", None),
                    "family": getattr(result, "family", None),
                }
            )

            metrics = {}
            for name, value in (
                ("mase", getattr(result, "mase", None)),
                ("naive_mase", getattr(result, "naive_mase", None)),
                ("skill_vs_naive", getattr(result, "skill_vs_naive", None)),
                ("forecast_end", getattr(result, "end_value", None)),
                ("last_value", getattr(result, "last_actual", None)),
            ):
                # MLflow rejects None and NaN.
                if value is not None and value == value:
                    metrics[name] = float(value)
            if metrics:
                mlflow.log_metrics(metrics)

            table = getattr(result, "table", None)
            if table is not None and len(table):
                mlflow.log_table(table, artifact_file="leaderboard.json")

            mlflow.set_tags(
                {
                    "foundation_model": str(
                        getattr(result, "family", "") == "foundation model"
                    ).lower(),
                    "source": "sa-insights",
                }
            )
            return run.info.run_id

    except Exception as exc:  # noqa: BLE001 - tracking is best-effort
        # Log locally rather than mutating the global status: one bad series
        # should not disable tracking for every subsequent series in the run.
        LOG.warning("mlflow logging failed for %s: %s: %s", series_key, type(exc).__name__, exc)
        return None
