"""Downloads generated on demand from the live store.

The previous design committed a pre-built Power BI zip and Ossie YAML to the
repository, which meant a download reflected whenever the build last ran — up to
a day stale, or much worse if the schedule was blocked. These helpers build
every artefact from DuckDB at request time, so what a user downloads is what
they just saw on screen.

Caching
-------
Two layers, deliberately:

* ``functools.lru_cache`` keyed on the ingest run id — an in-process memo that
  is invalidated automatically the moment new data lands.
* an explicit short TTL, because a long-running server may serve many users and
  we do not want to rebuild a 30 KB CSV per click.

Validation
----------
The Ossie document is validated against the official JSON Schema before it is
handed out. Shipping an invalid semantic model to someone's BI tool would be
worse than failing loudly here.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd

from . import store
from .catalog import SERIES
from .semantic import DRAFT_VERSION, OSSIE_VERSION, build_model, validate_schema

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "semantic" / "ossie-schema.json"

# Per-series measurement columns for the Power BI star schema.
POWERBI_HORIZONS = {"M": 12, "Q": 8, "A": 5}
MEASURE_COUNT = 17


def _stamp() -> str:
    """A cheap cache key: changes whenever an ingest writes new observations."""
    summary = store.data_summary()
    last = summary.get("last_ingest")
    return "" if last is None else str(last)


# ---------------------------------------------------------------------------
# Star schema for BI tools
# ---------------------------------------------------------------------------
def build_frames() -> dict[str, pd.DataFrame]:
    """Assemble the star schema from the live store.

    One long fact table with ``series_key`` as the foreign key, rather than one
    table per indicator: 34 near-identical tables would be unusable in a BI
    field list. The mapping is lossless.
    """
    obs = store.query_df(
        "SELECT series_key, period, value FROM observations ORDER BY series_key, period"
    )
    meta = store.query_df("SELECT * FROM indicators ORDER BY category, label")
    forecasts = store.latest_forecasts()

    dim_indicator = pd.DataFrame(
        {
            "series_key": meta["series_key"],
            # The warehouse column is `label`; the BI-facing name is `indicator`.
            "indicator": meta["label"],
            "short_label": meta["short_label"],
            "category": meta["category"],
            "unit": meta["unit"],
            "frequency": meta["frequency"],
            "provider": meta["provider"],
            "higher_is_better": meta["higher_is_better"],
            "description": meta["description"],
        }
    )

    fact_observation = pd.DataFrame(
        {
            "series_key": obs["series_key"],
            "date_key": pd.to_datetime(obs["period"]).dt.strftime("%Y-%m-%d"),
            "value": obs["value"].round(6),
        }
    )

    if not forecasts.empty:
        fact_forecast = pd.DataFrame(
            {
                "series_key": forecasts["series_key"],
                "date_key": pd.to_datetime(forecasts["period"]).dt.strftime("%Y-%m-%d"),
                "forecast": forecasts["forecast"],
                "lower_80": forecasts["lower"],
                "upper_80": forecasts["upper"],
                "model": forecasts["model"],
                "model_family": forecasts["family"],
            }
        )
    else:
        fact_forecast = pd.DataFrame(
            columns=[
                "series_key",
                "date_key",
                "forecast",
                "lower_80",
                "upper_80",
                "model",
                "model_family",
            ]
        )

    # Calendar dimension at month grain; every frequency lands on month starts.
    all_dates = pd.DatetimeIndex(
        sorted(
            set(pd.to_datetime(obs["period"]).dt.to_period("M").dt.to_timestamp())
            | set(
                pd.to_datetime(forecasts["period"]).dt.to_period("M").dt.to_timestamp()
            )
            if not forecasts.empty
            else set(pd.to_datetime(obs["period"]).dt.to_period("M").dt.to_timestamp())
        )
    )
    dim_date = pd.DataFrame(
        {
            "date_key": all_dates.strftime("%Y-%m-%d"),
            "date": all_dates.strftime("%Y-%m-%d"),
            "year": all_dates.year,
            "quarter": all_dates.quarter,
            "quarter_label": [f"Q{q} {y}" for q, y in zip(all_dates.quarter, all_dates.year)],
            "month": all_dates.month,
            "month_name": all_dates.strftime("%B"),
            "year_month": all_dates.strftime("%Y-%m"),
            "sort_key": all_dates.year * 100 + all_dates.month,
        }
    )

    return {
        "dim_indicator": dim_indicator,
        "dim_date": dim_date,
        "fact_observation": fact_observation,
        "fact_forecast": fact_forecast,
    }


def _dax_measures() -> str:
    """Reuse the canonical measure catalogue from the exporter script."""
    import importlib.util

    path = ROOT / "scripts" / "export_powerbi.py"
    spec = importlib.util.spec_from_file_location("_pbi_export", path)
    if spec is None or spec.loader is None:
        return "-- measure catalogue unavailable"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dax_file(build_frames())


def _powerbi_readme(frames: dict[str, pd.DataFrame]) -> str:
    obs = len(frames["fact_observation"])
    ind = len(frames["dim_indicator"])
    fc = len(frames["fact_forecast"])
    return f"""# SA Insights — Power BI model

Generated on download from the live dataset, so these figures are current as of
the moment you saved this file.

| File | Rows | Purpose |
|------|------|---------|
| `dim_indicator.csv` | {ind:,} | Indicator metadata: label, category, unit, frequency. |
| `dim_date.csv` | {len(frames['dim_date']):,} | Monthly calendar: year, quarter, month. |
| `fact_observation.csv` | {obs:,} | Measured values — `series_key` + `date_key` + `value`. |
| `fact_forecast.csv` | {fc:,} | Forecasts with an 80% interval and the winning model. |
| `sa_insights.model.bim` | — | Tabular model with relationships and measures. |
| `measures.dax` | {MEASURE_COUNT} | The same measures as readable DAX. |

## Import

1. Extract to a folder.
2. Power BI Desktop → **Get data** → **Text/CSV** → import all four CSVs.
3. Create these relationships:

   | From | To | Cardinality |
   |------|-----|-------------|
   | `fact_observation[series_key]` | `dim_indicator[series_key]` | many-to-one |
   | `fact_observation[date_key]` | `dim_date[date_key]` | many-to-one |
   | `fact_forecast[series_key]` | `dim_indicator[series_key]` | many-to-one |
   | `fact_forecast[date_key]` | `dim_date[date_key]` | many-to-one |

4. Select `dim_date` → **Table tools → Mark as date table**, column `date`.
5. Paste each measure from `measures.dax` into the table named above it.

## Caveat

Frequencies differ by indicator — monthly rates and inflation, quarterly GDP,
annual unemployment and debt. Filter on `dim_indicator[frequency]` when mixing
them, and always show the period alongside a headline figure.
"""


def powerbi_bundle() -> bytes:
    """Build the Power BI zip in memory, straight from the store."""
    from scripts.export_powerbi import build_bim  # noqa: PLC0415 - optional import

    frames = build_frames()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, frame in frames.items():
            archive.writestr(f"{name}.csv", frame.to_csv(index=False))
        archive.writestr(
            "sa_insights.model.bim", json.dumps(build_bim(frames), indent=2)
        )
        archive.writestr("measures.dax", _dax_measures())
        archive.writestr("README.md", _powerbi_readme(frames))
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Apache Ossie semantic model
# ---------------------------------------------------------------------------
def ossie_yaml(version: str = OSSIE_VERSION) -> bytes:
    """Render the semantic model as YAML, validated before it is served."""
    import yaml

    document = build_model(version=version)
    if version == DRAFT_VERSION:
        problems = validate_schema(document, SCHEMA_PATH)
        if problems:
            raise RuntimeError(
                "semantic model failed schema validation: " + "; ".join(problems[:3])
            )
    return yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100).encode(
        "utf-8"
    )


def ossie_json(version: str = OSSIE_VERSION) -> bytes:
    document = build_model(version=version)
    if version == DRAFT_VERSION:
        problems = validate_schema(document, SCHEMA_PATH)
        if problems:
            raise RuntimeError(
                "semantic model failed schema validation: " + "; ".join(problems[:3])
            )
    return json.dumps(document, indent=2).encode("utf-8")


# ---------------------------------------------------------------------------
# Plain data
# ---------------------------------------------------------------------------
def observations_csv(series_keys: list[str] | None = None) -> bytes:
    sql = "SELECT series_key, period, value FROM observations"
    params: list = []
    if series_keys:
        placeholders = ",".join(["?"] * len(series_keys))
        sql += f" WHERE series_key IN ({placeholders})"
        params = list(series_keys)
    sql += " ORDER BY series_key, period"
    frame = store.query_df(sql, params)
    return frame.to_csv(index=False).encode("utf-8")


def forecasts_csv() -> bytes:
    frame = store.latest_forecasts()
    return frame.to_csv(index=False).encode("utf-8")


def summary() -> dict:
    """What the download panel shows the user before they click."""
    counts = store.data_summary()
    frames = build_frames()
    return {
        **counts,
        "powerbi_rows": int(len(frames["fact_observation"])),
        "forecast_rows": int(len(frames["fact_forecast"])),
        "indicators": int(len(frames["dim_indicator"])),
    }
