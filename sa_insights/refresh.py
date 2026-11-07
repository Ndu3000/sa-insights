"""The refresh pipeline: upstream APIs → DuckDB → forecasts → MLflow.

This is the single code path that makes the dashboard current. It is called by
three things:

* the scheduler in the ``worker`` container (daily)
* the CLI (``python -m sa_insights.refresh``)
* the Streamlit app's "Refresh now" button, via ``refresh_all``

Ordering matters: observations are committed before forecasting, so a forecast
failure never leaves the store without data, and a partial run is always
visible in ``ingest_runs`` / ``forecast_runs``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from . import store, tracking
from .catalog import SERIES, SeriesSpec, short_label
from .forecast import ForecastError, infer_frequency, run_forecast
from .sources import load_all

# One horizon per frequency: a year of months, two years of quarters, five years.
HORIZONS = {"M": 12, "Q": 8, "A": 5}
CONFIDENCE = 0.80


def _utcnow() -> datetime:
    # Stored naive-UTC: DuckDB TIMESTAMP is timezone-free and mixing aware and
    # naive values in one column causes silent comparison bugs.
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class RefreshReport:
    ingest_run_id: str
    started_at: datetime
    finished_at: datetime
    series_loaded: int = 0
    series_total: int = 0
    observations: int = 0
    forecasts_written: int = 0
    forecasts_failed: list[dict] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    tracking: dict = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    def as_dict(self) -> dict:
        return {
            "ingest_run_id": self.ingest_run_id,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "finished_at": self.finished_at.isoformat(timespec="seconds"),
            "duration_seconds": round(self.duration_seconds, 1),
            "series_loaded": self.series_loaded,
            "series_total": self.series_total,
            "observations": self.observations,
            "forecasts_written": self.forecasts_written,
            "forecasts_failed": self.forecasts_failed,
            "failed": self.failed,
            "tracking": self.tracking,
        }


def _metadata_rows(data: dict[str, pd.Series]) -> list[dict]:
    """Build indicator rows for the metadata table."""
    rows = []
    for spec in SERIES:
        series = data.get(spec.key)
        if series is None or series.empty:
            continue
        rows.append(
            {
                "series_key": spec.key,
                "label": spec.label,
                "short_label": short_label(spec),
                "category": spec.category,
                "unit": spec.unit,
                "frequency": infer_frequency(series.index),
                "provider": spec.provider,
                "description": spec.description,
                "higher_is_better": (
                    "" if spec.higher_is_better is None else str(spec.higher_is_better).lower()
                ),
                "decimals": spec.decimals,
                "first_period": series.index[0].date(),
                "last_period": series.index[-1].date(),
                "observation_count": int(len(series)),
                "updated_at": _utcnow(),
            }
        )
    return rows


def collect(force: bool = True) -> tuple[dict[str, pd.Series], list[dict], RefreshReport]:
    """Fetch every series and commit it to the store."""
    started = _utcnow()
    run_id = str(uuid.uuid4())

    data, statuses = load_all(SERIES, force=force)
    observations = store.write_observations(data, ingest_run_id=run_id)
    store.write_indicator_metadata(_metadata_rows(data))

    failed = [
        {"key": s["key"], "label": s.get("label"), "error": s.get("error")}
        for s in statuses
        if s["state"] == "failed"
    ]
    finished = _utcnow()

    store.record_ingest(
        run_id=run_id,
        started_at=started,
        finished_at=finished,
        series_loaded=len(data),
        series_total=len(SERIES),
        observations=observations,
        failed=failed,
        mode="force" if force else "cache",
    )

    report = RefreshReport(
        ingest_run_id=run_id,
        started_at=started,
        finished_at=finished,
        series_loaded=len(data),
        series_total=len(SERIES),
        observations=observations,
        failed=failed,
        tracking=tracking.status(),
    )
    return data, statuses, report


def forecast_all(
    data: dict[str, pd.Series] | None = None,
    *,
    ingest_run_id: str | None = None,
    progress=None,
) -> tuple[int, list[dict]]:
    """Forecast every forecastable series, persist runs, log to MLflow."""
    if data is None:
        data = store.load_observations()

    candidates: list[SeriesSpec] = [
        spec
        for spec in SERIES
        if spec.forecast and spec.key in data and len(data[spec.key]) >= 24
    ]

    written = 0
    failures: list[dict] = []

    for index, spec in enumerate(candidates, start=1):
        series = data[spec.key]
        horizon = HORIZONS.get(spec.frequency, 6)
        try:
            result = run_forecast(series, horizon, confidence=CONFIDENCE)
        except ForecastError as exc:
            failures.append({"key": spec.key, "error": str(exc)})
            if progress:
                progress(spec, index, len(candidates), None, str(exc))
            continue
        except Exception as exc:  # noqa: BLE001 - one bad series must not stop the run
            failures.append({"key": spec.key, "error": f"{type(exc).__name__}: {exc}"})
            if progress:
                progress(spec, index, len(candidates), None, str(exc))
            continue

        mlflow_run_id = tracking.log_forecast_run(
            series_key=spec.key,
            result=result,
            window_obs=len(result.history),
            n_obs=len(series),
            horizon=horizon,
        )

        path = pd.DataFrame(
            {
                "period": [p.date() for p in result.forecast.index],
                "forecast": [float(v) for v in result.forecast],
                "lower": [float(v) for v in result.lower],
                "upper": [float(v) for v in result.upper],
            }
        )

        try:
            store.write_forecast_run(
                run_id=str(uuid.uuid4()),
                series_key=spec.key,
                generated_at=_utcnow(),
                horizon=horizon,
                confidence=float(result.confidence),
                model=result.model,
                family=result.family,
                mase=result.mase,
                naive_mase=result.naive_mase,
                skill_vs_naive=result.skill_vs_naive,
                window_obs=len(result.history),
                n_obs=len(series),
                forecast_end=result.end_value,
                last_value=result.last_actual,
                mlflow_run_id=mlflow_run_id,
                leaderboard=result.table,
                forecast_frame=path,
                ingest_run_id=ingest_run_id,
            )
        except Exception as exc:  # noqa: BLE001 - a lock clash must not abort the run
            failures.append(
                {"key": spec.key, "error": f"store write failed: {type(exc).__name__}: {exc}"}
            )
            if progress:
                progress(spec, index, len(candidates), None, str(exc))
            continue

        written += 1

        if progress:
            progress(spec, index, len(candidates), result, None)

    return written, failures


def refresh_all(*, force: bool = True, do_forecast: bool = True, progress=None) -> RefreshReport:
    """Full pipeline: collect, store, forecast, prune. Returns a report."""
    data, _statuses, report = collect(force=force)

    if do_forecast and data:
        written, failures = forecast_all(
            data, ingest_run_id=report.ingest_run_id, progress=progress
        )
        report.forecasts_written = written
        report.forecasts_failed = failures
        store.prune_forecasts(keep_per_series=10)

    report.finished_at = _utcnow()
    return report


def bootstrap_if_empty(force: bool = True) -> RefreshReport | None:
    """Populate the store on first run, so the app never opens on an empty page."""
    if store.is_initialised():
        return None
    return refresh_all(force=force)
