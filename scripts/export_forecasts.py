#!/usr/bin/env python
"""Precompute forecasts for the static GitHub Pages build.

Why precompute
--------------
Chronos-Bolt is a 48M-parameter neural network. Running it *in the browser*
would mean shipping ~191 MB of weights so WebAssembly can execute hundreds of
forward passes during a rolling-origin backtest. On the mobile connections this
dashboard targets that is simply not viable, and GitHub Pages serves static
files only anyway.

So the model runs here, once, at build time. The published site ships only the
resulting numbers as compact JSON — roughly 30 KB — and needs no ML runtime at
all. The page loads instantly while still showing genuine foundation-model
output.

Usage:
    python scripts/export_forecasts.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from sa_insights import chronos_fm  # noqa: E402
from sa_insights.catalog import BY_KEY, SERIES, SeriesSpec  # noqa: E402
from sa_insights.forecast import ForecastError, run_forecast  # noqa: E402
from sa_insights.sources import read_cache  # noqa: E402

OUT = Path(os.getenv("SA_FORECASTS_OUT", ROOT / "docs" / "data" / "forecasts.json"))

# One horizon per frequency: a year of months, two years of quarters, five years.
HORIZONS = {"D": 10, "M": 12, "Q": 8, "A": 5}

# Series featured on the static page.
FEATURED = ["cpi_yoy", "policy_rate", "usd_zar", "gdp_growth", "unemployment"]


def _round(values, places: int = 4) -> list[float]:
    return [round(float(v), places) for v in values]


def _clean(value, places: int = 4) -> float | None:
    """Round a metric, mapping None and NaN to None so JSON stays honest."""
    if value is None or value != value:
        return None
    return round(float(value), places)


def _spec_for(key: str) -> SeriesSpec | None:
    """Catalogue spec, or a labelled descriptor for the GDP nowcast estimate."""
    spec = BY_KEY.get(key)
    if spec is not None:
        return spec
    if key == "gdp_nowcast":
        # The nowcast is an estimate, not a catalogued observation, so it has
        # no SeriesSpec. Give it a minimal labelled descriptor instead of
        # dropping it, and mark it clearly as a nowcast downstream.
        return SeriesSpec(
            key="gdp_nowcast",
            label="GDP growth nowcast (estimate)",
            unit="%",
            frequency="Q",
            category="Growth & Output",
            description="Bridge-equation estimate of the next quarter's YoY GDP growth.",
            provider="derived",
            decimals=2,
            higher_is_better=True,
        )
    return None


def _leaderboard(head) -> list[dict]:
    """Normalise the stored leaderboard to lowercase keys.

    The warehouse stores the DataFrame's own column names ("Model", "Family",
    "MASE"); consumers here expect the lowercase form, and emitting both
    conventions caused a silent blank column on the page.
    """
    try:
        raw = json.loads(head.get("leaderboard") or "[]")
    except (json.JSONDecodeError, TypeError):
        raw = []
    return [
        {
            "model": row.get("Model", row.get("model")),
            "family": row.get("Family", row.get("family")),
            "mase": row.get("MASE", row.get("mase")),
        }
        for row in raw
        if isinstance(row, dict)
    ]


def _entry_for(key: str, spec: SeriesSpec, group, last_periods: dict) -> dict | None:
    """One series' payload entry from its latest forecast run rows."""
    head = group.iloc[0]
    path = group.dropna(subset=["forecast"]).sort_values("period")
    if path.empty:
        return None

    # A nowcast has no observation history, so the indicators table cannot
    # supply its last_period. Its anchor is the last *observed* quarter,
    # which is the quarter before the one being estimated.
    last_period = last_periods.get(key, "")
    if not last_period and key == "gdp_nowcast":
        last_period = (
            pd.Timestamp(path["period"].iloc[0]) - pd.offsets.QuarterBegin(1)
        ).strftime("%Y-%m-%d")

    return {
        "label": spec.label,
        "unit": spec.unit,
        "decimals": spec.decimals,
        "frequency": spec.frequency,
        "higher_is_better": spec.higher_is_better,
        "model": head.get("model"),
        "family": head.get("family"),
        "horizon": int(head.get("horizon") or 0),
        "mase": _clean(head.get("mase")),
        "naive_mase": _clean(head.get("naive_mase")),
        "skill_vs_naive": _clean(head.get("skill_vs_naive"), 1),
        "last_period": last_period,
        "last_value": round(float(head.get("last_value") or 0.0), 4),
        "periods": [pd.Timestamp(p).strftime("%Y-%m-%d") for p in path["period"]],
        "forecast": _round(path["forecast"]),
        "lower": _round(path["lower"]),
        "upper": _round(path["upper"]),
        "leaderboard": _leaderboard(head),
    }


def _from_warehouse(only_featured: bool) -> dict | None:
    """Build the payload from forecasts the worker already computed.

    The container's refresh pipeline stores every run in DuckDB. Reusing those
    means publishing the static site costs no model refitting at all — which
    matters when the publisher is a small container on a schedule, not a CI
    runner. Returns None when the warehouse has nothing usable, so the caller
    can fall back to computing fresh forecasts.
    """
    from sa_insights import store

    try:
        if not store.is_initialised():
            return None
        latest = store.latest_forecasts()
    except Exception as exc:  # noqa: BLE001 - fall back rather than fail
        print(f"warehouse unavailable ({exc}); computing forecasts instead")
        return None

    if latest.empty:
        return None

    # `forecast_runs` has no last_period column, so reading it from the run row
    # silently produced "" for every series on the published page — the site
    # could not say how current its own data was. The indicators table records
    # the real last observation per series, which is the authoritative answer.
    try:
        meta = store.indicator_metadata()
        last_periods = {
            str(r["series_key"]): (
                pd.Timestamp(r["last_period"]).strftime("%Y-%m-%d")
                if r.get("last_period") is not None
                else ""
            )
            for _, r in meta.iterrows()
        }
    except Exception:  # noqa: BLE001 - a missing table must not blank the page
        last_periods = {}

    payload_series: dict[str, dict] = {}

    for key, group in latest.groupby("series_key", sort=False):
        if only_featured and key not in FEATURED:
            continue
        spec = _spec_for(str(key))
        if spec is None:
            continue
        entry = _entry_for(str(key), spec, group, last_periods)
        if entry is not None:
            payload_series[key] = entry

    # The worker that produced these runs may have had the foundation model
    # while this publisher does not, so derive activity from the runs.
    if not payload_series:
        return None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "foundation_model": chronos_fm.MODEL_ID,
        # Derived from the runs themselves, not from this process's torch
        # availability: the worker that produced them may have had the model
        # while the publisher does not, so probing locally reports a false
        # negative.
        "foundation_model_active": any(
            v.get("family") == "foundation model" for v in payload_series.values()
        ),
        "forecast_note": (
            "Taken from the live warehouse — the same runs the dashboard shows."
        ),
        "series": payload_series,
    }


def build(*, only_featured: bool = False, prefer_warehouse: bool = True) -> dict:
    """Assemble the export payload.

    Order of preference:
      1. Forecasts already stored by the refresh worker (no refitting).
      2. Freshly computed forecasts from the local series cache.
    """
    if prefer_warehouse:
        payload = _from_warehouse(only_featured)
        if payload is not None:
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            return payload
        print("no warehouse forecasts; computing fresh")

    chronos_status = chronos_fm.describe()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "foundation_model": chronos_status["model_id"],
        "foundation_model_active": bool(chronos_status["available"]),
        "forecast_note": (
            "Precomputed at build time with the Apache Chronos-Bolt foundation "
            "model competing against classical baselines on a rolling-origin "
            "backtest. The browser only reads these numbers."
        ),
        "series": {},
    }

    for spec in SERIES:
        if not spec.forecast:
            continue
        if only_featured and spec.key not in FEATURED:
            continue

        series = read_cache(spec.key)
        if series is None or len(series) < 12:
            continue

        horizon = HORIZONS[spec.frequency]
        try:
            result = run_forecast(series, horizon, confidence=0.8)
        except ForecastError:
            continue

        payload["series"][spec.key] = {
            "label": spec.label,
            "unit": spec.unit,
            "decimals": spec.decimals,
            "frequency": spec.frequency,
            "higher_is_better": spec.higher_is_better,
            "model": result.model,
            "family": result.family,
            "horizon": horizon,
            "mase": None if result.mase is None else round(result.mase, 4),
            "naive_mase": None if result.naive_mase is None else round(result.naive_mase, 4),
            "skill_vs_naive": (
                None if result.skill_vs_naive is None else round(result.skill_vs_naive, 1)
            ),
            "last_period": result.history.index[-1].strftime("%Y-%m-%d"),
            "last_value": round(float(result.last_actual), 4),
            "periods": [p.strftime("%Y-%m-%d") for p in result.forecast.index],
            "forecast": _round(result.forecast),
            "lower": _round(result.lower),
            "upper": _round(result.upper),
            "leaderboard": [
                {
                    "model": str(row["Model"]),
                    "family": str(row["Family"]),
                    "mase": None if row["MASE"] != row["MASE"] else round(float(row["MASE"]), 4),
                }
                for _, row in result.table.iterrows()
            ],
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Compact separators: this file is a wire format, not something to read.
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--featured",
        action="store_true",
        help="only export the series featured on the static page",
    )
    args = parser.parse_args()

    payload = build(only_featured=args.featured)

    size = OUT.stat().st_size
    series = payload["series"]
    fm_wins = sum(1 for s in series.values() if s["family"] == "foundation model")
    beating_naive = sum(
        1 for s in series.values() if (s["skill_vs_naive"] or 0) > 0
    )

    print(f"wrote {OUT.relative_to(ROOT)}  ({size:,} bytes)")
    print(f"series exported : {len(series)}")
    print(f"naive beaten by : {beating_naive}/{len(series)}")
    print(f"Chronos wins    : {fm_wins}")
    print(f"foundation model: {payload['foundation_model']}")
    print(f"  active        : {payload['foundation_model_active']}")
    if not payload["foundation_model_active"]:
        print(f"  reason        : {chronos_fm.unavailable_reason()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
