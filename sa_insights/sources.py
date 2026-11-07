"""Fetchers for each upstream provider, plus a disk-cached resolution layer.

All upstream calls are key-free and read-only. Every series resolves to a tidy
pandas Series with a DatetimeIndex, so downstream code never has to care which
provider it came from.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .catalog import SeriesSpec

DBNOMICS = "https://api.db.nomics.world/v22"
WORLDBANK = "https://api.worldbank.org/v2"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
# The official South African Reserve Bank Web API. Publishes the primary data
# for rates, exchange rates, production, sales and cycle indicators, and is
# updated within days of each release — unlike the DBnomics IMF mirror, which
# lags by a year or more.
SARB_API = "https://custom.resbank.co.za/SarbWebApi"
# The IMF DataMapper API. Unlike the DBnomics WEO mirror (frozen at an old
# vintage) and the retired IMF SDMX-JSON service, this is live and carries the
# current World Economic Outlook vintage, including projections several years
# ahead.
IMF_DATAMAPPER = "https://www.imf.org/external/datamapper/api/v1"

# Cache location. Must honour DATA_DIR: in a container the package lives under
# /app, which is read-only for the unprivileged user, while the writable volume
# is mounted at /data. Resolving relative to the package file ignored that and
# caused "Permission denied: '/app/data'" on every fetch.
_DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
CACHE_DIR = Path(os.getenv("SA_CACHE_DIR", _DATA_DIR / "cache"))
RAW_DIR = CACHE_DIR / "raw"
DEFAULT_TTL_HOURS = 12

# A neutral user-agent is deliberate. FRED's edge rejects requests that
# advertise a real browser (it treats them as automation) but happily serves
# plain clients, so a simple descriptive UA works across all providers.
USER_AGENT = f"sa-insights/{__import__('sa_insights').__version__} (+python-requests)"


class FetchError(RuntimeError):
    """Raised when a series cannot be retrieved from any provider."""


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _get(
    url: str,
    params: dict | None = None,
    *,
    retries: int = 3,
    timeout: int = 45,
    expect_json: bool = True,
    headers: dict | None = None,
):
    """GET with bounded exponential backoff. Raises FetchError on failure.

    Hosts differ on what they accept: FRED's edge throttles anything claiming to
    be a browser, while the IMF DataMapper rejects a custom user-agent outright
    but serves the requests library default. Callers that know their host pass
    explicit `headers`; the default is the project user-agent.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers=headers if headers is not None else {"User-Agent": USER_AGENT, "Accept": "*/*"},
            )
            if resp.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            resp.raise_for_status()
            if not expect_json:
                return resp.text
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - network layer is best-effort
            last = exc
            time.sleep(1.2 * (attempt + 1))
    raise FetchError(f"{url} failed after {retries} attempts: {last}")


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
def _period_to_timestamp(period) -> pd.Timestamp | None:
    """Convert DBnomics-style period strings to a month-start timestamp."""
    if period is None:
        return None
    raw = str(period).strip()
    if not raw or raw.upper() == "NA":
        return None
    try:
        if "-Q" in raw:  # 2024-Q3
            year, qtr = raw.split("-Q")
            return pd.Timestamp(year=int(year), month=(int(qtr) - 1) * 3 + 1, day=1)
        if len(raw) == 4:  # 2024
            return pd.Timestamp(year=int(raw), month=1, day=1)
        if len(raw) == 7:  # 2024-03
            return pd.Timestamp(raw + "-01")
        if len(raw) == 10:
            return pd.Timestamp(raw)
    except (ValueError, TypeError):
        return None
    return None


def _clean(periods, values) -> pd.Series:
    """Build a sorted float Series, dropping NA / non-numeric / duplicate stamps."""
    pairs: list[tuple[pd.Timestamp, float]] = []
    for period, value in zip(periods, values):
        stamp = _period_to_timestamp(period)
        if stamp is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
        if pd.isna(value):
            continue
        pairs.append((stamp, float(value)))
    if not pairs:
        return pd.Series(dtype="float64")
    series = pd.Series(
        [v for _, v in pairs],
        index=pd.DatetimeIndex([s for s, _ in pairs]),
        dtype="float64",
    )
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.name = None
    return series


def _fetch_dbnomics(spec: SeriesSpec) -> pd.Series:
    """Resolve a DBnomics dataset via its dimensions (robust across versions)."""
    fetch = spec.fetch
    provider, dataset = fetch["provider"], fetch["dataset"]
    dimensions = fetch["dimensions"]

    if dataset == "WEO" and not str(fetch.get("dataset", "")).count(":"):
        # Resolve the newest WEO vintage that actually has our subject.
        datasets = _get(f"{DBNOMICS}/datasets/{provider}", {"limit": 200})
        codes = sorted(
            d["code"] for d in datasets.get("datasets", {}).get("docs", [])
            if d["code"].startswith("WEO:")
        )
        last_err: Exception | None = None
        for code in reversed(codes[-6:]):
            try:
                return _dbnomics_series(provider, code, dimensions, fetch.get("prefer_unit"))
            except FetchError as exc:
                last_err = exc
                continue
        raise FetchError(f"no WEO vintage had data ({last_err})")

    return _dbnomics_series(provider, dataset, dimensions, fetch.get("prefer_unit"))


def _dbnomics_series(provider, dataset, dimensions, prefer_unit) -> pd.Series:
    import urllib.parse

    params = {
        "dimensions": json.dumps(dimensions, separators=(",", ":")),
        "observations": 1,
        "limit": 50,
    }
    url = f"{DBNOMICS}/series/{provider}/{urllib.parse.quote(dataset, safe=':/')}"
    payload = _get(url, params)

    docs = payload.get("series", {}).get("docs") or []
    if not docs:
        msg = payload.get("message") or "no series returned"
        raise FetchError(f"{provider}/{dataset}: {msg}")

    if prefer_unit and len(docs) > 1:
        for doc in docs:
            if str(doc.get("unit") or "").lower() == prefer_unit.lower():
                return _clean(doc.get("period", []), doc.get("value", []))
        for doc in docs:
            unit = str(doc.get("unit") or "").lower()
            if prefer_unit.lower().startswith(unit) and unit:
                return _clean(doc.get("period", []), doc.get("value", []))

    doc = docs[0]
    return _clean(doc.get("period", []), doc.get("value", []))


def _fetch_worldbank(spec: SeriesSpec) -> pd.Series:
    fetch = spec.fetch
    url = f"{WORLDBANK}/country/{fetch.get('country', 'ZAF')}/indicator/{fetch['indicator']}"
    params = {"format": "json", "per_page": 1000, "date": "1960:2035"}

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            payload = _get(url, params, retries=2)
        except FetchError as exc:
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
            continue
        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            last_err = FetchError("empty payload")
            time.sleep(1.5 * (attempt + 1))
            continue
        rows = payload[1]
        stamps, values = [], []
        for row in rows:
            if row.get("value") is None:
                continue
            stamps.append(_period_to_timestamp(row.get("date")))
            values.append(row["value"])
        series = pd.Series(
            [float(v) for v in values],
            index=pd.DatetimeIndex([s for s in stamps if s is not None]),
            dtype="float64",
        )
        if len(series):
            return series[~series.index.duplicated(keep="last")].sort_index()
        last_err = FetchError("no non-null observations")
        time.sleep(1.5 * (attempt + 1))
    raise FetchError(f"World Bank {fetch['indicator']}: {last_err}")


def _fetch_fred(spec: SeriesSpec) -> pd.Series:
    sid = spec.fetch["series_id"]
    text = _get(f"{FRED_CSV}?id={sid}", expect_json=False)
    if "observation_date" not in text[:200]:
        raise FetchError(f"FRED {sid}: unexpected response (bot challenge?)")
    from io import StringIO

    frame = pd.read_csv(StringIO(text))
    frame.columns = ["date", "value"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna()
    if frame.empty:
        raise FetchError(f"FRED {sid}: no observations")
    out = pd.Series(frame["value"].to_numpy(dtype="float64"), index=frame["date"])
    out = out[out.index.duplicated(keep="last") == False].sort_index()
    out.name = None
    return out


def _fetch_imf_datamapper(spec: SeriesSpec) -> pd.Series:
    """Fetch a series from the IMF DataMapper API.

    Payload shape::

        {"values": {"<INDICATOR>": {"ZAF": {"2024": 0.5, "2025": 1.1, ...},
                                    "SDN": {...}, ...}}}

    Some indicators honour the ``/{indicator}/{country}`` path and return just
    one country; others ignore it and return all ~229. Both shapes are handled
    by always looking up the country key rather than assuming a single block.

    Values include published projections, so the series legitimately extends
    into the future. Those are kept — they are the IMF's own outlook, which the
    dashboard shows as a benchmark rather than as observed data.
    """
    indicator = spec.fetch["indicator"]
    country = spec.fetch.get("country", "ZAF")
    url = f"{IMF_DATAMAPPER}/{indicator}/{country}"

    # The IMF API rejects custom user-agents with a 403 but serves the requests
    # library default, so this call deliberately sends no User-Agent header.
    last: Exception | None = None
    payload = None
    for attempt in range(4):
        try:
            payload = _get(url, timeout=60, retries=1, headers={"Accept": "*/*"})
            break
        except FetchError as exc:
            last = exc
            time.sleep(2.0 * (attempt + 1))
    if payload is None:
        raise FetchError(f"IMF {indicator}/{country}: {last}")

    if not isinstance(payload, dict):
        raise FetchError(f"IMF {indicator}: unexpected payload")

    by_country = payload.get("values", {}).get(indicator, {})
    if not isinstance(by_country, dict) or not by_country:
        raise FetchError(f"IMF {indicator}: no values returned")

    # The response may be {ZAF: {...}} or {SDN: {...}, ZAF: {...}, ...}.
    block = by_country.get(country)
    if block is None and len(by_country) == 1:
        block = next(iter(by_country.values()))
    if not isinstance(block, dict) or not block:
        raise FetchError(f"IMF {indicator}: no data for {country}")

    stamps: list[pd.Timestamp] = []
    values: list[float] = []
    for period, raw in block.items():
        if raw is None:
            continue
        try:
            year = int(str(period)[:4])
            value = float(raw)
        except (TypeError, ValueError):
            continue
        stamps.append(pd.Timestamp(year=year, month=1, day=1))
        values.append(value)

    if not stamps:
        raise FetchError(f"IMF {indicator}/{country}: no numeric observations")

    series = pd.Series(values, index=pd.DatetimeIndex(stamps), dtype="float64")
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.name = None
    return series


def _fetch_sarb(spec: SeriesSpec) -> pd.Series:
    """Fetch a timeseries from the official SARB Web API.

    Endpoint: ``/WebIndicators/Shared/GetTimeseriesObservations/{code}/{from}/{to}``

    The unbounded variant returns only a short default window (about two years),
    so we always pass explicit dates to get the full history. Returns a JSON
    array of ``{Period, Timeseries, Value, ...}`` objects.

    The API reports daily granularity even for monthly series (each month is a
    run of business-day rows sharing the same value), so observations are
    collapsed to one per period at the series' native frequency.
    """
    code = spec.fetch["timeseries_code"]
    start = spec.fetch.get("start", "1980-01-01")
    end = spec.fetch.get("end", "2035-12-31")
    url = (
        f"{SARB_API}/WebIndicators/Shared/GetTimeseriesObservations/{code}/{start}/{end}"
    )
    payload = _get(url, timeout=90)

    if not isinstance(payload, list) or not payload:
        raise FetchError(f"SARB {code}: unexpected payload shape")

    rows: list[tuple[pd.Timestamp, float]] = []
    for item in payload:
        period = item.get("Period")
        raw = item.get("Value")
        if period is None or raw is None:
            continue
        try:
            if isinstance(raw, str):
                # SARB uses a decimal comma in some payloads.
                raw = raw.replace(" ", "").replace(",", ".")
            value = float(raw)
        except (TypeError, ValueError):
            continue
        try:
            stamp = pd.Timestamp(str(period)[:10])
        except (ValueError, TypeError):
            continue
        if pd.isna(stamp):
            continue
        rows.append((stamp, value))

    if not rows:
        raise FetchError(f"SARB {code}: no usable observations")

    frame = pd.DataFrame(rows, columns=["date", "value"])
    if spec.frequency == "M":
        # Collapse the intra-month business-day rows to one value per month.
        frame["month"] = frame["date"].dt.to_period("M")
        frame = frame.groupby("month", as_index=False).last()
        frame["date"] = frame["month"].dt.to_timestamp()
    elif spec.frequency == "Q":
        frame["quarter"] = frame["date"].dt.to_period("Q")
        frame = frame.groupby("quarter", as_index=False).last()
        frame["date"] = frame["quarter"].dt.to_timestamp()
    else:
        frame["year"] = frame["date"].dt.year
        frame = frame.groupby("year", as_index=False).last()
        frame["date"] = pd.to_datetime(frame["year"].astype(str) + "-01-01")

    series = pd.Series(
        frame["value"].to_numpy(dtype="float64"),
        index=pd.DatetimeIndex(frame["date"]),
    )
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.name = None
    return series


_FETCHERS = {
    "dbnomics": _fetch_dbnomics,
    "worldbank": _fetch_worldbank,
    "fred": _fetch_fred,
    "sarb": _fetch_sarb,
    "imf": _fetch_imf_datamapper,
}


# ---------------------------------------------------------------------------
# Caching layer
# ---------------------------------------------------------------------------
def _cache_paths(key: str) -> tuple[Path, Path]:
    return CACHE_DIR / f"{key}.csv", RAW_DIR / f"{key}.json"


def read_cache(key: str) -> pd.Series | None:
    path, _ = _cache_paths(key)
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, parse_dates=["date"])
        frame = frame.dropna(subset=["value"])
        if frame.empty:
            return None
        series = pd.Series(
            frame["value"].to_numpy(dtype="float64"),
            index=pd.DatetimeIndex(frame["date"]),
        )
        series.name = None
        return series.sort_index()
    except Exception:  # noqa: BLE001 - corrupt cache is simply a cache miss
        return None


def cache_meta(key: str) -> dict:
    _, meta_path = _cache_paths(key)
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def cache_age_hours(key: str) -> float | None:
    path, _ = _cache_paths(key)
    if not path.exists():
        return None
    age = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
    return age / 3600.0


def _write_cache(key: str, series: pd.Series, spec: SeriesSpec) -> None:
    path, meta_path = _cache_paths(key)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": series.index, "value": series.to_numpy()}).to_csv(
        path, index=False
    )
    meta = {
        "key": key,
        "label": spec.label,
        "unit": spec.unit,
        "frequency": spec.frequency,
        "category": spec.category,
        "provider": spec.provider,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "observations": int(len(series)),
        "first_period": series.index[0].strftime("%Y-%m-%d") if len(series) else None,
        "last_period": series.index[-1].strftime("%Y-%m-%d") if len(series) else None,
        "last_value": float(series.iloc[-1]) if len(series) else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_series(
    spec: SeriesSpec,
    *,
    force: bool = False,
    ttl_hours: float = DEFAULT_TTL_HOURS,
) -> tuple[pd.Series, dict]:
    """Return (series, status) preferring fresh cache, then upstream, then stale cache."""
    path, _ = _cache_paths(spec.key)
    age = cache_age_hours(spec.key)
    cached = read_cache(spec.key) if path.exists() else None

    if cached is not None and not force and age is not None and age < ttl_hours:
        return cached, {"key": spec.key, "state": "cache", "age_hours": age, "error": None}

    try:
        series = _FETCHERS[spec.provider](spec)
        if series is None or series.empty:
            raise FetchError("provider returned no usable observations")
        _write_cache(spec.key, series, spec)
        return series, {
            "key": spec.key,
            "state": "live",
            "age_hours": 0.0,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - fall back to whatever we already have
        if cached is not None and not cached.empty:
            return cached, {
                "key": spec.key,
                "state": "stale",
                "age_hours": age,
                "error": str(exc)[:220],
            }
        return pd.Series(dtype="float64"), {
            "key": spec.key,
            "state": "failed",
            "age_hours": age,
            "error": str(exc)[:220],
        }


def load_all(
    specs: list[SeriesSpec],
    *,
    force: bool = False,
    ttl_hours: float = DEFAULT_TTL_HOURS,
    progress=None,
) -> tuple[dict[str, pd.Series], list[dict]]:
    """Load many series, reporting progress through an optional callback."""
    data: dict[str, pd.Series] = {}
    statuses: list[dict] = []
    for spec in specs:
        series, status = load_series(spec, force=force, ttl_hours=ttl_hours)
        status["label"] = spec.label
        status["category"] = spec.category
        if not series.empty:
            data[spec.key] = series
        statuses.append(status)
        if progress is not None:
            progress(spec, status, len(statuses), len(specs))
    return data, statuses
