"""DuckDB-backed store.

Why DuckDB
----------
The workload is analytical time series: a few dozen indicators, a few thousand
observations, and aggregate reads for charts. That is exactly what DuckDB is
built for, and it needs no server process — the whole database is one file,
which makes backup and local development trivial. Postgres would add an
operational dependency for no benefit at this size.

Layout
------
* ``indicators``      — one row per catalogued series (metadata)
* ``observations``    — the measurements themselves
* ``forecast_runs``   — one row per (series, refresh) with accuracy metrics
* ``forecasts``       — the projected path belonging to a run
* ``ingest_runs``     — audit trail of each collection pass
* ``dashboards``      — user-saved layouts

Everything is idempotent: refreshes replace whole partitions rather than
appending forever, so the file does not grow without bound.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

# Where the database lives. Overridable so containers can mount a volume.
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
DB_PATH = Path(os.getenv("SA_DB_PATH", DATA_DIR / "sa_insights.duckdb"))

# DuckDB permits one writer process. The `app` and `worker` containers share the
# same volume, so an on-demand refresh in the UI can collide with a scheduled
# one and fail with "Conflicting lock is held". A lock file serialises writers
# across processes; readers are never blocked.
LOCK_PATH = DB_PATH.with_suffix(".lock")
LOCK_TIMEOUT_SECONDS = float(os.getenv("SA_DB_LOCK_TIMEOUT", "120"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS indicators (
    series_key        VARCHAR PRIMARY KEY,
    label             VARCHAR NOT NULL,
    short_label       VARCHAR,
    category          VARCHAR,
    unit              VARCHAR,
    frequency         VARCHAR,
    provider          VARCHAR,
    description       VARCHAR,
    higher_is_better  VARCHAR,
    decimals          INTEGER,
    first_period      DATE,
    last_period       DATE,
    observation_count INTEGER,
    updated_at        TIMESTAMP
);

CREATE TABLE IF NOT EXISTS observations (
    series_key VARCHAR NOT NULL,
    period     DATE    NOT NULL,
    value      DOUBLE  NOT NULL,
    PRIMARY KEY (series_key, period)
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id        VARCHAR PRIMARY KEY,
    started_at    TIMESTAMP,
    finished_at   TIMESTAMP,
    series_loaded INTEGER,
    series_total  INTEGER,
    observations  INTEGER,
    failed        VARCHAR,
    mode          VARCHAR
);

CREATE TABLE IF NOT EXISTS forecast_runs (
    run_id           VARCHAR PRIMARY KEY,
    series_key       VARCHAR NOT NULL,
    generated_at     TIMESTAMP NOT NULL,
    horizon          INTEGER,
    confidence       DOUBLE,
    model            VARCHAR,
    family           VARCHAR,
    mase             DOUBLE,
    naive_mase       DOUBLE,
    skill_vs_naive   DOUBLE,
    window_obs       INTEGER,
    n_obs            INTEGER,
    forecast_end     DOUBLE,
    last_value       DOUBLE,
    mlflow_run_id    VARCHAR,
    leaderboard      VARCHAR,
    ingest_run_id    VARCHAR
);

CREATE TABLE IF NOT EXISTS forecasts (
    run_id    VARCHAR NOT NULL,
    series_key VARCHAR NOT NULL,
    period    DATE    NOT NULL,
    forecast  DOUBLE  NOT NULL,
    lower     DOUBLE,
    upper     DOUBLE,
    PRIMARY KEY (run_id, period)
);

CREATE TABLE IF NOT EXISTS dashboards (
    dashboard_id VARCHAR PRIMARY KEY,
    owner        VARCHAR NOT NULL,
    name         VARCHAR NOT NULL,
    config       VARCHAR NOT NULL,
    created_at   TIMESTAMP,
    updated_at   TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_obs_series ON observations (series_key);
CREATE INDEX IF NOT EXISTS idx_fc_series  ON forecasts (series_key);
CREATE INDEX IF NOT EXISTS idx_runs_series ON forecast_runs (series_key, generated_at);
CREATE INDEX IF NOT EXISTS idx_dash_owner ON dashboards (owner);
"""


def ensure_dir() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _write_lock(timeout: float = LOCK_TIMEOUT_SECONDS):
    """Serialise writers across processes using an atomic lock file.

    DuckDB's own locking is single-process, so two containers writing the same
    file produce "Conflicting lock is held in PID 0". An O_CREAT|O_EXCL lock
    file gives us a portable cross-process mutex; stale locks from a crashed
    writer are reclaimed after `timeout` seconds based on the file's mtime.
    """
    ensure_dir()
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            # Reclaim a lock left behind by a crashed process.
            try:
                age = time.time() - LOCK_PATH.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > timeout:
                try:
                    LOCK_PATH.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"could not acquire write lock on {DB_PATH} within {timeout}s "
                    f"(another refresh is probably running)"
                )
            time.sleep(0.25)

    try:
        yield
    finally:
        try:
            LOCK_PATH.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def connect(read_only: bool = False):
    """Yield a DuckDB connection, guaranteeing schema and closing cleanly.

    Writers take the cross-process lock first; readers do not, so the dashboard
    never blocks behind a refresh. Note that DuckDB does not support
    simultaneous multi-process access: a read that lands while another process
    holds the file raises IOException. Callers that read should use
    ``query_df``, which retries.
    """
    ensure_dir()

    if read_only:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        try:
            yield con
        finally:
            con.close()
        return

    with _write_lock():
        con = duckdb.connect(str(DB_PATH), read_only=False)
        try:
            con.execute(SCHEMA)
            yield con
        finally:
            con.close()


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------
def query_df(sql: str, params: list | tuple | None = None) -> pd.DataFrame:
    """Run a read query and return a DataFrame. Never writes.

    Retries on lock contention: DuckDB is single-process, so a read issued while
    the worker is committing will fail with IOException. A short backoff resolves
    it without failing the request.
    """
    last: Exception | None = None
    for attempt in range(6):
        try:
            with connect(read_only=True) as con:
                return con.execute(sql, params or []).fetch_df()
        except (duckdb.IOException, duckdb.ConnectionException) as exc:
            last = exc
            time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"database stayed locked after 6 attempts: {last}")


def is_initialised() -> bool:
    if not DB_PATH.exists():
        return False
    try:
        with connect(read_only=True) as con:
            count = con.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        return count > 0
    except Exception:  # noqa: BLE001 - missing tables simply means "not ready"
        return False


def load_observations(series_keys: list[str] | None = None) -> dict[str, pd.Series]:
    """Return {series_key: Series indexed by period} for the dashboard."""
    sql = "SELECT series_key, period, value FROM observations"
    params: list = []
    if series_keys:
        placeholders = ",".join(["?"] * len(series_keys))
        sql += f" WHERE series_key IN ({placeholders})"
        params = list(series_keys)
    sql += " ORDER BY series_key, period"

    frame = query_df(sql, params)
    if frame.empty:
        return {}

    out: dict[str, pd.Series] = {}
    for key, group in frame.groupby("series_key", sort=False):
        series = pd.Series(
            group["value"].to_numpy(dtype="float64"),
            index=pd.DatetimeIndex(group["period"]),
        )
        series = series[~series.index.duplicated(keep="last")].sort_index()
        series.name = None
        out[str(key)] = series
    return out


def indicator_metadata() -> pd.DataFrame:
    return query_df("SELECT * FROM indicators ORDER BY category, label")


def latest_forecasts(series_key: str | None = None) -> pd.DataFrame:
    """The most recent forecast run per series, with its projected path joined on."""
    where = ""
    params: list = []
    if series_key:
        where = "WHERE r.series_key = ?"
        params = [series_key]

    return query_df(
        f"""
        WITH latest AS (
            SELECT series_key, MAX(generated_at) AS generated_at
            FROM forecast_runs
            {where}
            GROUP BY series_key
        )
        SELECT r.*, f.period, f.forecast, f.lower, f.upper
        FROM forecast_runs r
        JOIN latest l
          ON l.series_key = r.series_key AND l.generated_at = r.generated_at
        LEFT JOIN forecasts f ON f.run_id = r.run_id
        ORDER BY r.series_key, f.period
        """,
        params,
    )


def data_summary() -> dict:
    """Headline counts used by the dashboard header and health checks."""
    if not is_initialised():
        return {"series": 0, "observations": 0, "last_ingest": None, "forecasts": 0}
    row = query_df(
        """
        SELECT
            (SELECT COUNT(DISTINCT series_key) FROM observations)   AS series,
            (SELECT COUNT(*) FROM observations)                     AS observations,
            (SELECT MAX(finished_at) FROM ingest_runs)              AS last_ingest,
            (SELECT COUNT(*) FROM forecast_runs)                    AS forecasts
        """
    ).iloc[0]
    return {
        "series": int(row["series"]),
        "observations": int(row["observations"]),
        "last_ingest": None if pd.isna(row["last_ingest"]) else row["last_ingest"],
        "forecasts": int(row["forecasts"]),
    }


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------
def _upsert_frame(con, table: str, frame: pd.DataFrame, keys: list[str]) -> None:
    """Replace rows matching the frame's keys, then insert.

    DuckDB's DELETE ... USING is the cheapest correct approach here; tables are
    small enough that a merge join on a temp table is unnecessary.
    """
    if frame.empty:
        return
    con.register("_incoming", frame)
    try:
        key_match = " AND ".join(f"{table}.{k} = _incoming.{k}" for k in keys)
        con.execute(f"DELETE FROM {table} USING _incoming WHERE {key_match}")
        con.execute(f"INSERT INTO {table} SELECT * FROM _incoming")
    finally:
        con.unregister("_incoming")


def write_observations(
    data: dict[str, pd.Series],
    *,
    ingest_run_id: str | None = None,
) -> int:
    """Replace the stored observations for the given series."""
    if not data:
        return 0

    rows = []
    for key, series in data.items():
        if series is None or series.empty:
            continue
        clean = series.dropna()
        for period, value in clean.items():
            rows.append(
                {
                    "series_key": key,
                    "period": pd.Timestamp(period).to_pydatetime().date(),
                    "value": float(value),
                }
            )

    if not rows:
        return 0

    frame = pd.DataFrame(rows)
    with connect() as con:
        for key in data:
            con.execute("DELETE FROM observations WHERE series_key = ?", [key])
        con.register("_obs", frame)
        con.execute("INSERT INTO observations SELECT series_key, period, value FROM _obs")
        con.unregister("_obs")
    return len(frame)


def write_indicator_metadata(rows: list[dict]) -> int:
    if not rows:
        return 0
    frame = pd.DataFrame(rows)
    with connect() as con:
        _upsert_frame(con, "indicators", frame, ["series_key"])
    return len(frame)


def record_ingest(
    *,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    series_loaded: int,
    series_total: int,
    observations: int,
    failed: list[dict],
    mode: str,
) -> None:
    frame = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "started_at": started_at,
                "finished_at": finished_at,
                "series_loaded": series_loaded,
                "series_total": series_total,
                "observations": observations,
                "failed": json.dumps(failed),
                "mode": mode,
            }
        ]
    )
    with connect() as con:
        con.register("_ing", frame)
        con.execute("INSERT INTO ingest_runs SELECT * FROM _ing")
        con.unregister("_ing")


def write_forecast_run(
    *,
    run_id: str,
    series_key: str,
    generated_at: datetime,
    horizon: int,
    confidence: float,
    model: str,
    family: str,
    mase: float | None,
    naive_mase: float | None,
    skill_vs_naive: float | None,
    window_obs: int | None,
    n_obs: int,
    forecast_end: float,
    last_value: float,
    mlflow_run_id: str | None,
    leaderboard: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    ingest_run_id: str | None = None,
) -> None:
    """Persist one series' forecast run plus its projected path."""
    head = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "series_key": series_key,
                "generated_at": generated_at,
                "horizon": horizon,
                "confidence": confidence,
                "model": model,
                "family": family,
                "mase": mase,
                "naive_mase": naive_mase,
                "skill_vs_naive": skill_vs_naive,
                "window_obs": window_obs,
                "n_obs": n_obs,
                "forecast_end": forecast_end,
                "last_value": last_value,
                "mlflow_run_id": mlflow_run_id,
                "leaderboard": leaderboard.to_json(orient="records"),
                "ingest_run_id": ingest_run_id,
            }
        ]
    )

    path = forecast_frame.copy()
    path.insert(0, "series_key", series_key)
    path.insert(0, "run_id", run_id)

    with connect() as con:
        con.register("_fr", head)
        con.execute("INSERT INTO forecast_runs SELECT * FROM _fr")
        con.unregister("_fr")

        con.register("_fc", path)
        con.execute("INSERT INTO forecasts SELECT run_id, series_key, period, forecast, lower, upper FROM _fc")
        con.unregister("_fc")


# ---------------------------------------------------------------------------
# Saved dashboards
# ---------------------------------------------------------------------------
def save_dashboard(owner: str, name: str, config: dict) -> str:
    """Create or update a named dashboard for an owner."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    existing = query_df(
        "SELECT dashboard_id FROM dashboards WHERE owner = ? AND name = ?",
        [owner, name],
    )
    config_json = json.dumps(config)

    with connect() as con:
        if not existing.empty:
            dashboard_id = str(existing.iloc[0]["dashboard_id"])
            con.execute(
                "UPDATE dashboards SET config = ?, updated_at = ? WHERE dashboard_id = ?",
                [config_json, now, dashboard_id],
            )
        else:
            dashboard_id = str(uuid.uuid4())
            con.execute(
                "INSERT INTO dashboards VALUES (?, ?, ?, ?, ?, ?)",
                [dashboard_id, owner, name, config_json, now, now],
            )
    return dashboard_id


def list_dashboards(owner: str) -> pd.DataFrame:
    return query_df(
        "SELECT dashboard_id, name, updated_at FROM dashboards "
        "WHERE owner = ? ORDER BY updated_at DESC",
        [owner],
    )


def get_dashboard(owner: str, name: str) -> dict | None:
    frame = query_df(
        "SELECT config FROM dashboards WHERE owner = ? AND name = ?",
        [owner, name],
    )
    if frame.empty:
        return None
    try:
        return json.loads(frame.iloc[0]["config"])
    except (json.JSONDecodeError, TypeError):
        return None


def delete_dashboard(owner: str, name: str) -> bool:
    with connect() as con:
        before = con.execute(
            "SELECT COUNT(*) FROM dashboards WHERE owner = ? AND name = ?",
            [owner, name],
        ).fetchone()[0]
        if not before:
            return False
        con.execute(
            "DELETE FROM dashboards WHERE owner = ? AND name = ?", [owner, name]
        )
    return True


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------
def prune_forecasts(keep_per_series: int = 10) -> int:
    """Retain only the most recent runs per series, so the file stays small."""
    with connect() as con:
        stale = con.execute(
            """
            SELECT run_id FROM (
                SELECT run_id,
                       ROW_NUMBER() OVER (PARTITION BY series_key
                                          ORDER BY generated_at DESC) AS rn
                FROM forecast_runs
            )
            WHERE rn > ?
            """,
            [keep_per_series],
        ).fetchall()

        if not stale:
            return 0

        ids = [row[0] for row in stale]
        placeholders = ",".join(["?"] * len(ids))
        con.execute(f"DELETE FROM forecasts WHERE run_id IN ({placeholders})", ids)
        con.execute(f"DELETE FROM forecast_runs WHERE run_id IN ({placeholders})", ids)
    return len(ids)
