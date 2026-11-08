"""Deterministic query tools over the DuckDB warehouse.

The contract
------------
Every number that appears in a chatbot answer originates from a function in this
module. Retrieval (``graph``) may suggest *what* to look up, but it never
supplies a figure. That separation is the whole point: a language model asked
"what is CPI now?" from memory will invent a plausible number, whereas a tool
either returns the stored value or raises.

Guardrails
----------
- Read-only. ``store.query_df`` opens DuckDB read-only, and ``_guard`` rejects
  anything that is not a single SELECT so a generated statement cannot mutate.
- Bounded. Row caps and explicit column lists stop a broad question from
  pulling 8,000 rows into a prompt.
- Honest about staleness. Every value carries its period, because "latest" for
  annual GDP and for the rand mean very different dates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable

from .. import store

# Hard ceilings. A prompt containing 8,000 observations is both expensive and
# useless: the model cannot read it, and the answer stops being checkable.
MAX_POINTS = 120
MAX_SERIES_PER_CALL = 8

_WRITE_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|truncate|attach|copy|replace|grant|revoke)\b",
    re.IGNORECASE,
)


class ToolError(RuntimeError):
    """A tool could not answer. Surfaced to the agent, never guessed around."""


def _guard(sql: str) -> None:
    """Reject anything that is not a single read-only SELECT.

    Not a full parser — it only needs to stop the classes of statement that
    could change state, and a generated query is the realistic threat here.
    """
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        raise ToolError("only a single statement is permitted")
    if not stripped.lower().startswith(("select", "with")):
        raise ToolError("only SELECT statements are permitted")
    if _WRITE_PATTERN.search(stripped):
        raise ToolError("statement contains a data-modifying keyword")


def run_select(sql: str, params: Iterable[Any] | None = None) -> list[dict]:
    """Execute a guarded read query and return plain records."""
    _guard(sql)
    frame = store.query_df(sql, list(params or []))
    return frame.to_dict("records")


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #
@dataclass
class SeriesInfo:
    key: str
    label: str
    short_label: str | None
    unit: str | None
    frequency: str | None
    category: str | None
    description: str | None
    higher_is_better: str | None
    last_period: str | None
    observation_count: int | None


def list_series(category: str | None = None) -> list[SeriesInfo]:
    """Every series, optionally filtered to one category."""
    sql = "SELECT * FROM indicators"
    params: list[Any] = []
    if category:
        sql += " WHERE lower(category) = lower(?)"
        params.append(category)
    sql += " ORDER BY category, label"

    rows = run_select(sql, params)
    return [
        SeriesInfo(
            key=row["series_key"],
            label=row["label"],
            short_label=row.get("short_label"),
            unit=row.get("unit"),
            frequency=row.get("frequency"),
            category=row.get("category"),
            description=row.get("description"),
            higher_is_better=row.get("higher_is_better"),
            last_period=_iso(row.get("last_period")),
            observation_count=row.get("observation_count"),
        )
        for row in rows
    ]


def categories() -> list[str]:
    rows = run_select(
        "SELECT category, COUNT(*) AS n FROM indicators "
        "WHERE category IS NOT NULL GROUP BY category ORDER BY category"
    )
    return [f"{row['category']} ({row['n']})" for row in rows]


def _iso(value: Any) -> str | None:
    """Render a period as YYYY-MM-DD.

    DuckDB returns DATE as pandas Timestamp via fetch_df, so checking only for
    ``datetime.date`` let values through as "2026-09-01T00:00:00" — which then
    appeared verbatim in answers and looked like a bug to the reader.
    """
    if value is None:
        return None
    if hasattr(value, "date") and not isinstance(value, str):
        try:
            return value.date().isoformat()
        except (AttributeError, ValueError):
            pass
    return str(value)[:10]


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #
@dataclass
class Change:
    """A movement between two observations, with both endpoints named.

    Both endpoints are carried because a bare percentage is uncheckable — "CPI
    is 4.5%" says nothing about whether it improved.
    """

    absolute: float | None
    percent: float | None
    from_period: str
    from_value: float
    to_period: str
    to_value: float


@dataclass
class SeriesSnapshot:
    key: str
    label: str
    unit: str | None
    frequency: str | None
    higher_is_better: bool | None = None
    points: list[dict] = field(default_factory=list)
    latest: dict | None = None
    yoy: Change | None = None
    since_start: Change | None = None
    min_point: dict | None = None
    max_point: dict | None = None

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "unit": self.unit,
            "frequency": self.frequency,
            "higher_is_better": self.higher_is_better,
            "latest": self.latest,
            "year_on_year": _change_dict(self.yoy),
            "change_since_first_point": _change_dict(self.since_start),
            "min": self.min_point,
            "max": self.max_point,
            "points": self.points,
        }


def _change_dict(change: Change | None) -> dict | None:
    if change is None:
        return None
    return {
        "absolute": None if change.absolute is None else round(change.absolute, 4),
        "percent": None if change.percent is None else round(change.percent, 2),
        "from_period": change.from_period,
        "from_value": round(change.from_value, 4),
        "to_period": change.to_period,
        "to_value": round(change.to_value, 4),
    }


def _change(points: list[dict], *, is_rate: bool = False) -> Change | None:
    """Movement between the first and last point.

    ``is_rate`` suppresses the percentage for series that are already rates.
    A CPI reading going 3.4% -> 4.5% is a 1.1 percentage-point rise; calling it
    "+32%" is arithmetically true and completely misleading, and a reader who
    sees it will reasonably conclude inflation tripled in level.
    """
    if len(points) < 2:
        return None
    first, last = points[0], points[-1]
    absolute = last["value"] - first["value"]
    if is_rate:
        percent = None
    else:
        # Percent change is undefined through zero; returning None is honest,
        # while dividing anyway produces a spectacular meaningless number.
        percent = (absolute / abs(first["value"]) * 100.0) if first["value"] else None
    return Change(
        absolute=absolute,
        percent=percent,
        from_period=first["period"],
        from_value=first["value"],
        to_period=last["period"],
        to_value=last["value"],
    )


def _is_rate(unit: str | None, label: str) -> bool:
    """Is this series already expressed as a percentage?

    Determined from the unit and label rather than hardcoded, so a new rate
    series added to the catalogue behaves correctly without a code change.
    """
    text = f"{unit or ''} {label}".lower()
    return any(word in text for word in ("rate", "%", "percent", "inflation", "yoy"))


def get_series(
    key: str,
    *,
    periods: int | None = None,
    since: str | None = None,
    max_points: int = MAX_POINTS,
) -> SeriesSnapshot:
    """Fetch one series' observations, with the movements most questions ask about.

    ``since`` is an ISO date (``"2020-01-01"``) or a bare year (``"2020"``);
    ``periods`` takes the most recent N observations instead.
    """
    meta = run_select("SELECT * FROM indicators WHERE series_key = ?", [key])
    if not meta:
        raise ToolError(f"unknown series '{key}'")
    info = meta[0]

    sql = "SELECT period, value FROM observations WHERE series_key = ?"
    params: list[Any] = [key]
    if since:
        sql += " AND period >= ?"
        params.append(_coerce_since(since))
    sql += " ORDER BY period"

    rows = run_select(sql, params)
    if not rows:
        raise ToolError(f"no observations for '{key}'")

    points = [{"period": _iso(r["period"]), "value": float(r["value"])} for r in rows]
    if periods:
        points = points[-max(1, min(periods, max_points)):]
    elif len(points) > max_points:
        # Thin evenly rather than truncating: the shape of a 30-year monthly
        # series matters more than its first 120 rows.
        step = len(points) / max_points
        thinned = [points[int(i * step)] for i in range(max_points)]
        if thinned[-1] != points[-1]:
            thinned.append(points[-1])
        points = thinned

    latest = points[-1]
    as_rate = _is_rate(info.get("unit"), str(info.get("label") or ""))
    yoy: Change | None = None
    # Year-on-year only where the cadence makes it meaningful. Comparing an
    # annual series to a point one row back would be called "yoy" wrongly.
    cadence = (info.get("frequency") or "").upper()
    if cadence in {"M", "Q"}:
        back = 12 if cadence == "M" else 4
        sql_yoy = (
            "SELECT period, value FROM observations WHERE series_key = ? "
            "ORDER BY period DESC LIMIT 1 OFFSET ?"
        )
        prior = run_select(sql_yoy, [key, back])
        if prior:
            before = {"period": _iso(prior[0]["period"]), "value": float(prior[0]["value"])}
            yoy = _change([before, latest], is_rate=as_rate)

    values = [p["value"] for p in points]
    return SeriesSnapshot(
        key=key,
        label=info["label"],
        unit=info.get("unit"),
        frequency=info.get("frequency"),
        higher_is_better=_flag(info.get("higher_is_better")),
        points=points,
        latest=latest,
        yoy=yoy,
        since_start=_change(points, is_rate=as_rate),
        min_point=points[values.index(min(values))] if points else None,
        max_point=points[values.index(max(values))] if points else None,
    )


def _flag(raw: Any) -> bool | None:
    """Tri-state parse of the stored ``higher_is_better`` column.

    None is a real answer: some series (the exchange rate, unemployment) are not
    simply better in one direction, and pretending otherwise would make the
    chatbot assert an improvement it cannot justify.
    """
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _coerce_since(value: str) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d{4}", text):
        return f"{text}-01-01"
    if not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", text):
        raise ToolError(f"could not parse '{value}' as a date or year")
    return text if len(text) > 7 else f"{text}-01"


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def compare_series(
    keys: list[str],
    *,
    since: str | None = None,
    max_points: int = MAX_POINTS,
) -> dict:
    """Side-by-side movements. Units differ, so percentages are the comparison.

    Absolute levels are included per series but deliberately not ranked against
    each other: a percentage-point rate change and a rand exchange rate are not
    commensurable, and sorting them together would be meaningless.
    """
    if not keys:
        raise ToolError("no series requested")
    if len(keys) > MAX_SERIES_PER_CALL:
        raise ToolError(f"at most {MAX_SERIES_PER_CALL} series per comparison")

    snapshots = [get_series(k, since=since, max_points=max_points) for k in keys]
    ranked = sorted(
        (
            {
                "key": snap.key,
                "label": snap.label,
                "percent_change": (snap.since_start.percent if snap.since_start else None),
                "absolute_change": (snap.since_start.absolute if snap.since_start else None),
                "unit": snap.unit,
                "direction": _direction(snap),
            }
            for snap in snapshots
        ),
        # Percentage is None for rate series, so rank those on the absolute move
        # instead of dropping them to the bottom as unrankable.
        key=lambda row: (
            -(row["percent_change"] if row["percent_change"] is not None
              else abs(row["absolute_change"] or 0))
        ),
    )
    return {"series": [s.to_dict() for s in snapshots], "ranked_by_movement": ranked}


def _direction(snapshot: SeriesSnapshot) -> str:
    """Which way a series moved, and whether that direction is good.

    ``higher_is_better`` is part of the series definition, so "improved" is
    answered from metadata rather than inferred from the sign of the change.
    Year-on-year is preferred when available because a single month is noise;
    otherwise the whole window is used.
    """
    change = snapshot.yoy or snapshot.since_start
    if change is None or change.absolute is None:
        return "unknown"
    if abs(change.absolute) < 1e-9:
        return "unchanged"

    rose = change.absolute > 0
    better = snapshot.higher_is_better
    if better is None:
        return "rose" if rose else "fell"
    return f"{'rose' if rose else 'fell'} ({'improved' if rose == better else 'worsened'})"


# --------------------------------------------------------------------------- #
# Forecasts
# --------------------------------------------------------------------------- #
def get_forecast(key: str) -> dict:
    """The most recent forecast for one series, plus the model competition."""
    import json

    rows = run_select(
        "SELECT run_id, generated_at, horizon, model, family, mase, naive_mase, "
        "skill_vs_naive, last_value, leaderboard, mlflow_run_id "
        "FROM forecast_runs WHERE series_key = ? "
        "ORDER BY generated_at DESC LIMIT 1",
        [key],
    )
    if not rows:
        raise ToolError(f"no forecast recorded for '{key}'")

    run = rows[0]
    path = run_select(
        "SELECT period, forecast, lower, upper FROM forecasts WHERE run_id = ? ORDER BY period",
        [run["run_id"]],
    )

    try:
        board = json.loads(run.get("leaderboard") or "[]")
    except (json.JSONDecodeError, TypeError):
        board = []

    return {
        "key": key,
        "generated_at": _iso(run.get("generated_at")),
        "horizon": run.get("horizon"),
        "model": run.get("model"),
        "family": run.get("family"),
        "mase": run.get("mase"),
        "skill_vs_naive": run.get("skill_vs_naive"),
        "last_actual": run.get("last_value"),
        "mlflow_run_id": run.get("mlflow_run_id"),
        "path": [
            {
                "period": _iso(p["period"]),
                "forecast": p["forecast"],
                "lower": p.get("lower"),
                "upper": p.get("upper"),
            }
            for p in path
        ],
        # Normalised: the warehouse stores DataFrame column names, and reading
        # only one casing silently dropped the runner-up on the published page.
        "leaderboard": [
            {
                "model": row.get("Model", row.get("model")),
                "family": row.get("Family", row.get("family")),
                "mase": row.get("MASE", row.get("mase")),
            }
            for row in board
            if isinstance(row, dict)
        ],
    }


def forecast_board(limit: int = 20) -> list[dict]:
    """Which model won which series, and by how much — the honest scoreboard."""
    return run_select(
        """
        WITH latest AS (
            SELECT series_key, MAX(generated_at) AS generated_at
            FROM forecast_runs GROUP BY series_key
        )
        SELECT r.series_key, i.label, r.model, r.family, r.mase,
               r.skill_vs_naive, r.generated_at
        FROM forecast_runs r
        JOIN latest l ON l.series_key = r.series_key AND l.generated_at = r.generated_at
        LEFT JOIN indicators i ON i.series_key = r.series_key
        ORDER BY r.skill_vs_naive DESC NULLS LAST
        LIMIT ?
        """,
        [max(1, min(limit, 50))],
    )


# --------------------------------------------------------------------------- #
# Orientation
# --------------------------------------------------------------------------- #
def latest_readings(keys: list[str] | None = None) -> list[dict]:
    """Current value, period and recent direction for headline indicators.

    This is the tool a broad question ("how is the economy doing?") should hit
    first: it is cheap, bounded, and gives the model real anchors before it
    decides what to look at next.
    """
    if keys:
        targets = keys[:MAX_SERIES_PER_CALL * 2]
    else:
        from ..catalog import HEADLINE

        targets = list(HEADLINE)

    out: list[dict] = []
    for key in targets:
        try:
            snap = get_series(key, periods=13)
        except ToolError:
            continue
        out.append(
            {
                "key": snap.key,
                "label": snap.label,
                "unit": snap.unit,
                "latest_period": snap.latest["period"] if snap.latest else None,
                "latest_value": snap.latest["value"] if snap.latest else None,
                "year_on_year": _change_dict(snap.yoy),
                "direction": _direction(snap),
            }
        )
    return out


def search_series(query: str, limit: int = 8) -> list[dict]:
    """Lexical lookup over labels, keys and descriptions.

    SQL LIKE rather than the graph's scorer: this is the tool for "which series
    are about bonds", where an exact substring is what the user means.
    """
    pattern = f"%{query.strip().lower()}%"
    return run_select(
        """
        SELECT series_key, label, category, unit, last_period
        FROM indicators
        WHERE lower(label) LIKE ? OR lower(series_key) LIKE ?
           OR lower(coalesce(description, '')) LIKE ?
           OR lower(coalesce(category, '')) LIKE ?
        ORDER BY label
        LIMIT ?
        """,
        [pattern, pattern, pattern, pattern, max(1, min(limit, 25))],
    )


def data_freshness() -> dict:
    """How current the warehouse is, and which series are thinnest.

    Answers "is this up to date?" honestly instead of implying live data.
    """
    summary = run_select(
        "SELECT COUNT(*) AS series, "
        "SUM(observation_count) AS observations, "
        "MAX(updated_at) AS last_updated FROM indicators"
    )
    oldest = run_select(
        "SELECT series_key, label, last_period FROM indicators "
        "ORDER BY last_period ASC NULLS LAST LIMIT 5"
    )
    newest = run_select(
        "SELECT series_key, label, last_period FROM indicators "
        "ORDER BY last_period DESC NULLS LAST LIMIT 5"
    )
    return {
        "totals": summary[0] if summary else {},
        "stalest": oldest,
        "most_current": newest,
    }
