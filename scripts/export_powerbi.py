#!/usr/bin/env python
"""Export the semantic model as a Power BI project bundle.

Power BI cannot read an Apache Ossie document directly, so this script
translates the same semantics into things Power BI actually consumes:

* ``dim_indicator.csv`` / ``dim_date.csv`` / ``fact_observation.csv`` /
  ``fact_forecast.csv`` - a conventional star schema, importable anywhere.
* ``sa_insights.model.bim`` - a Tabular Model Definition (TMSL) document that
  wires the tables together with relationships, hierarchies and measures. Open
  it in Tabular Editor, or point Power BI Desktop at the extracted CSVs.
* ``measures.dax`` - the same measures as readable, copy-pasteable DAX.
* ``README.md`` - step-by-step import instructions.

Everything is packaged into a single zip that the GitHub Pages site offers as a
download, so a Power BI author can start from the published model rather than
rebuilding it.

Design choice
-------------
The Ossie model uses one fact table per indicator. Power BI users are far
better served by a single long fact table plus an indicator dimension: it keeps
the field list short, makes slicers natural, and avoids 29 near-identical
tables. The mapping is lossless - ``series_key`` is the foreign key.

Usage:
    python scripts/export_powerbi.py
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from sa_insights.catalog import SERIES, short_label  # noqa: E402
from sa_insights.forecast import infer_frequency  # noqa: E402
from sa_insights.sources import read_cache  # noqa: E402

# Honour the DOCS override so the container publisher writes into its mounted
# checkout. BUILD_DIR is scratch and can stay wherever the process may write.
_DOCS = Path(os.getenv("SA_DOCS_OUT", ROOT / "docs"))
OUT_DIR = _DOCS / "download"
BUILD_DIR = Path(os.getenv("SA_BUILD_DIR", ROOT / "build" / "powerbi"))
FC_PATH = _DOCS / "data" / "forecasts.json"
ZIP_NAME = "sa-insights-powerbi.zip"

MODEL_NAME = "sa_insights"


# ---------------------------------------------------------------------------
# Star schema
# ---------------------------------------------------------------------------
def build_frames() -> dict[str, pd.DataFrame]:
    """Assemble the dimension and fact tables from the cached series."""
    indicator_rows = []
    observation_rows = []
    all_dates: set[pd.Timestamp] = set()

    for spec in SERIES:
        series = read_cache(spec.key)
        if series is None or series.empty:
            continue
        freq = infer_frequency(series.index)

        indicator_rows.append(
            {
                "series_key": spec.key,
                "indicator": spec.label,
                "short_label": short_label(spec),
                "category": spec.category,
                "unit": spec.unit,
                "frequency": freq,
                "provider": spec.provider,
                "higher_is_better": (
                    "" if spec.higher_is_better is None else str(spec.higher_is_better).lower()
                ),
                "description": spec.description,
            }
        )

        for stamp, value in series.items():
            period = pd.Timestamp(stamp).to_period("M").to_timestamp()
            all_dates.add(period)
            observation_rows.append(
                {
                    "series_key": spec.key,
                    "date_key": period.strftime("%Y-%m-%d"),
                    "value": round(float(value), 6),
                }
            )

    dim_indicator = pd.DataFrame(indicator_rows)
    fact_observation = pd.DataFrame(observation_rows)

    # Forecasts, when the forecast export has run.
    forecast_rows = []
    if FC_PATH.exists():
        payload = json.loads(FC_PATH.read_text(encoding="utf-8"))
        for key, entry in payload.get("series", {}).items():
            for i, period in enumerate(entry["periods"]):
                stamp = pd.Timestamp(period).to_period("M").to_timestamp()
                all_dates.add(stamp)
                forecast_rows.append(
                    {
                        "series_key": key,
                        "date_key": stamp.strftime("%Y-%m-%d"),
                        "forecast": entry["forecast"][i],
                        "lower_80": entry["lower"][i],
                        "upper_80": entry["upper"][i],
                        "model": entry["model"],
                        "model_family": entry["family"],
                    }
                )

    fact_forecast = pd.DataFrame(
        forecast_rows,
        columns=[
            "series_key",
            "date_key",
            "forecast",
            "lower_80",
            "upper_80",
            "model",
            "model_family",
        ],
    )

    # Date dimension, one row per first-of-month. Monthly, quarterly and annual
    # observations all land on month starts, so a single grain serves all three.
    dates = pd.DatetimeIndex(sorted(all_dates))
    dim_date = pd.DataFrame(
        {
            "date_key": dates.strftime("%Y-%m-%d"),
            "date": dates.strftime("%Y-%m-%d"),
            "year": dates.year,
            "quarter": dates.quarter,
            "quarter_label": [f"Q{q} {y}" for q, y in zip(dates.quarter, dates.year)],
            "month": dates.month,
            "month_name": dates.strftime("%B"),
            "year_month": dates.strftime("%Y-%m"),
            "sort_key": dates.year * 100 + dates.month,
        }
    )

    return {
        "dim_indicator": dim_indicator,
        "dim_date": dim_date,
        "fact_observation": fact_observation,
        "fact_forecast": fact_forecast,
    }


# ---------------------------------------------------------------------------
# DAX measures
# ---------------------------------------------------------------------------
MEASURES: list[tuple[str, str, str, str]] = [
    # (name, table, dax, description)
    (
        "Observation Count",
        "fact_observation",
        "Observation Count = COUNTROWS ( fact_observation )",
        "Number of observations in the current filter context.",
    ),
    (
        "Latest Period",
        "fact_observation",
        """Latest Period =
VAR LatestKey = CALCULATE ( MAX ( fact_observation[date_key] ), ALLSELECTED ( fact_observation ) )
RETURN
    LOOKUPVALUE ( dim_date[date], dim_date[date_key], LatestKey )""",
        "Most recent observation date in the current selection.",
    ),
    (
        "Latest Value",
        "fact_observation",
        """Latest Value =
VAR LatestKey = CALCULATE ( MAX ( fact_observation[date_key] ), ALLSELECTED ( fact_observation ) )
RETURN
    CALCULATE ( SUM ( fact_observation[value] ), fact_observation[date_key] = LatestKey )""",
        "The most recent published value for each selected indicator.",
    ),
    (
        "Previous Value",
        "fact_observation",
        """Previous Value =
VAR LatestKey = CALCULATE ( MAX ( fact_observation[date_key] ), ALLSELECTED ( fact_observation ) )
VAR PrevKey = CALCULATE ( MAX ( fact_observation[date_key] ), fact_observation[date_key] < LatestKey )
RETURN
    CALCULATE ( SUM ( fact_observation[value] ), fact_observation[date_key] = PrevKey )""",
        "The observation immediately before the latest one.",
    ),
    (
        "Change",
        "fact_observation",
        """Change = [Latest Value] - [Previous Value]""",
        "Absolute change between the last two observations.",
    ),
    (
        "Change %",
        "fact_observation",
        """Change % =
VAR Prev = [Previous Value]
RETURN
    IF ( ABS ( Prev ) > 0, DIVIDE ( [Change], ABS ( Prev ) ), BLANK () )""",
        "Percentage change between the last two observations.",
    ),
    (
        "Value YoY Change",
        "fact_observation",
        """Value YoY Change =
VAR Current = [Latest Value]
VAR Prior = CALCULATE ( [Latest Value], DATEADD ( dim_date[date], -1, YEAR ) )
RETURN
    IF ( NOT ISBLANK ( Prior ), Current - Prior )""",
        "Change versus the same period a year earlier. Requires dim_date marked as a date table.",
    ),
    (
        "Value YoY %",
        "fact_observation",
        """Value YoY % =
VAR Current = [Latest Value]
VAR Prior = CALCULATE ( [Latest Value], DATEADD ( dim_date[date], -1, YEAR ) )
RETURN
    IF ( NOT ISBLANK ( Prior ) && ABS ( Prior ) > 0, DIVIDE ( Current - Prior, ABS ( Prior ) ), BLANK () )""",
        "Year-on-year percentage change.",
    ),
    (
        "Average Value",
        "fact_observation",
        "Average Value = AVERAGE ( fact_observation[value] )",
        "Mean of the selected observations.",
    ),
    (
        "Volatility",
        "fact_observation",
        "Volatility = STDEV.P ( fact_observation[value] )",
        "Population standard deviation of the selected observations.",
    ),
    (
        "Coverage Years",
        "fact_observation",
        """Coverage Years =
VAR FirstYear = CALCULATE ( MIN ( dim_date[year] ), ALLSELECTED ( dim_date ) )
VAR LastYear = CALCULATE ( MAX ( dim_date[year] ), ALLSELECTED ( dim_date ) )
RETURN
    LastYear - FirstYear + 1""",
        "Span of the selected data, in years.",
    ),
    (
        "Indicator Count",
        "dim_indicator",
        "Indicator Count = DISTINCTCOUNT ( dim_indicator[series_key] )",
        "Number of distinct indicators in the current selection.",
    ),
    (
        "Real Policy Rate",
        "fact_observation",
        """Real Policy Rate =
VAR Repo = CALCULATE ( [Latest Value], dim_indicator[series_key] = "policy_rate" )
VAR Cpi  = CALCULATE ( [Latest Value], dim_indicator[series_key] = "cpi_yoy" )
RETURN
    IF ( NOT ISBLANK ( Repo ) && NOT ISBLANK ( Cpi ), Repo - Cpi )""",
        "Repo rate less consumer inflation. Positive means policy is restrictive in real terms.",
    ),
    (
        "Credit Spread",
        "fact_observation",
        """Credit Spread =
VAR Lending = CALCULATE ( [Latest Value], dim_indicator[series_key] = "lending_rate" )
VAR Repo    = CALCULATE ( [Latest Value], dim_indicator[series_key] = "policy_rate" )
RETURN
    IF ( NOT ISBLANK ( Lending ) && NOT ISBLANK ( Repo ), Lending - Repo )""",
        "Bank lending rate less the policy rate, in percentage points.",
    ),
    (
        "Trade Balance (US$ m)",
        "fact_observation",
        """Trade Balance (US$ m) =
VAR Exports = CALCULATE ( [Latest Value], dim_indicator[series_key] = "exports_usd" )
VAR Imports = CALCULATE ( [Latest Value], dim_indicator[series_key] = "imports_usd" )
RETURN
    IF ( NOT ISBLANK ( Exports ) && NOT ISBLANK ( Imports ), Exports - Imports )""",
        "Goods exports less goods imports, in millions of US dollars.",
    ),
    (
        "Forecast End Value",
        "fact_forecast",
        """Forecast End Value =
VAR LastKey = CALCULATE ( MAX ( fact_forecast[date_key] ), ALLSELECTED ( fact_forecast ) )
RETURN
    CALCULATE ( SUM ( fact_forecast[forecast] ), fact_forecast[date_key] = LastKey )""",
        "Final projected value at the end of the forecast horizon.",
    ),
    (
        "Forecast Interval Width",
        "fact_forecast",
        "Forecast Interval Width = AVERAGE ( fact_forecast[upper_80] ) - AVERAGE ( fact_forecast[lower_80] )",
        "Average width of the 80% prediction band. Wider means less certainty.",
    ),
]


def dax_file(frames: dict[str, pd.DataFrame]) -> str:
    lines = [
        "/*",
        " * SA Insights - DAX measure catalogue",
        " * Generated " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        " *",
        " * Prerequisites in the Power BI model:",
        " *   1. Mark dim_date as the model date table (Table tools > Mark as date table,",
        " *      column: date). Several measures use DATEADD, which requires it.",
        " *   2. Create the relationships listed in README.md.",
        " *",
        " * Paste each measure into the table named in its comment.",
        " */",
        "",
    ]
    current_table = None
    for name, table, dax, description in MEASURES:
        if table != current_table:
            lines.append(f"// ===== {table} " + "=" * max(4, 60 - len(table)))
            current_table = table
        lines.append(f"// {description}")
        lines.append(dax)
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TMSL model definition
# ---------------------------------------------------------------------------
def _col(name: str, dtype: str, source: str, *, hidden: bool = False) -> dict:
    return {
        "name": name,
        "dataType": dtype,
        "sourceColumn": source,
        "lineageTag": f"{source}.col",
        "annotations": [{"name": "SummarizationSetBy", "value": "Automatic"}],
        **({"isHidden": True} if hidden else {}),
    }


def _m_partition(table: str, query: str) -> dict:
    return {
        "name": table,
        "mode": "import",
        "source": {"type": "m", "expression": query},
    }


def _csv_query(file_name: str) -> list[str]:
    """Power Query M that reads a CSV from the extracted bundle folder."""
    return [
        "let",
        f'    Source = Csv.Document(File.Contents(DataFolder & "\\{file_name}"),'
        "[Delimiter=\",\", Columns=null, Encoding=65001, QuoteStyle=QuoteStyle.Csv]),",
        "    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),",
        (
            "    Typed = Table.TransformColumnTypes(Promoted,"
            "{{"
            + ", ".join(f'"{c}", type {t}' for c, t in _POWERQUERY_TYPES[file_name])
            + "})"
        ),
        "in",
        "    Typed",
    ]


_POWERQUERY_TYPES: dict[str, list[tuple[str, str]]] = {
    "dim_indicator.csv": [
        ("series_key", "text"),
        ("indicator", "text"),
        ("short_label", "text"),
        ("category", "text"),
        ("unit", "text"),
        ("frequency", "text"),
        ("provider", "text"),
        ("higher_is_better", "text"),
        ("description", "text"),
    ],
    "dim_date.csv": [
        ("date_key", "text"),
        ("date", "date"),
        ("year", "Int64.Type"),
        ("quarter", "Int64.Type"),
        ("quarter_label", "text"),
        ("month", "Int64.Type"),
        ("month_name", "text"),
        ("year_month", "text"),
        ("sort_key", "Int64.Type"),
    ],
    "fact_observation.csv": [
        ("series_key", "text"),
        ("date_key", "text"),
        ("value", "Double.Type"),
    ],
    "fact_forecast.csv": [
        ("series_key", "text"),
        ("date_key", "text"),
        ("forecast", "Double.Type"),
        ("lower_80", "Double.Type"),
        ("upper_80", "Double.Type"),
        ("model", "text"),
        ("model_family", "text"),
    ],
}


def build_bim(frames: dict[str, pd.DataFrame]) -> dict:
    """Build a TMSL (model.bim) document for the star schema."""

    def measure(name: str, dax: str, description: str) -> dict:
        body = dax
        # TMSL stores the expression without the leading "Name = " prefix.
        if body.startswith(name + " ="):
            body = body[len(name) + 2 :].strip()
        return {
            "name": name,
            "expression": body,
            "description": description,
            "lineageTag": f"{name}.measure",
        }

    tables = []

    # ----- dim_indicator -------------------------------------------------
    di = frames["dim_indicator"]
    tables.append(
        {
            "name": "dim_indicator",
            "lineageTag": "dim_indicator.tbl",
            "columns": [
                _col("series_key", "string", "series_key"),
                _col("Indicator", "string", "indicator"),
                _col("Short Label", "string", "short_label"),
                _col("Category", "string", "category"),
                _col("Unit", "string", "unit"),
                _col("Frequency", "string", "frequency"),
                _col("Provider", "string", "provider"),
                _col("Higher Is Better", "string", "higher_is_better"),
                _col("Description", "string", "description"),
            ],
            "partitions": [_m_partition("dim_indicator", "\n".join(_csv_query("dim_indicator.csv")))],
            "measures": [measure("Indicator Count", MEASURES[11][2], MEASURES[11][3])],
            "annotations": [{"name": "PBI_ResultType", "value": "Table"}],
        }
    )

    # ----- dim_date ------------------------------------------------------
    tables.append(
        {
            "name": "dim_date",
            "lineageTag": "dim_date.tbl",
            "columns": [
                _col("date_key", "string", "date_key", hidden=True),
                _col("date", "dateTime", "date"),
                _col("year", "int64", "year"),
                _col("quarter", "int64", "quarter"),
                _col("quarter_label", "string", "quarter_label"),
                _col("month", "int64", "month"),
                _col("month_name", "string", "month_name"),
                _col("year_month", "string", "year_month"),
                _col("sort_key", "int64", "sort_key", hidden=True),
            ],
            "partitions": [_m_partition("dim_date", "\n".join(_csv_query("dim_date.csv")))],
            "hierarchies": [
                {
                    "name": "Calendar",
                    "levels": [
                        {"name": "Year", "ordinal": 0, "column": "year"},
                        {"name": "Quarter", "ordinal": 1, "column": "quarter"},
                        {"name": "Month", "ordinal": 2, "column": "month_name"},
                    ],
                }
            ],
            "annotations": [{"name": "PBI_ResultType", "value": "Table"}],
        }
    )

    # ----- fact_observation ----------------------------------------------
    obs_measures = [
        measure(name, dax, desc)
        for name, table, dax, desc in MEASURES
        if table == "fact_observation"
    ]
    tables.append(
        {
            "name": "fact_observation",
            "lineageTag": "fact_observation.tbl",
            "columns": [
                _col("series_key", "string", "series_key", hidden=True),
                _col("date_key", "string", "date_key", hidden=True),
                _col("value", "double", "value"),
            ],
            "partitions": [
                _m_partition("fact_observation", "\n".join(_csv_query("fact_observation.csv")))
            ],
            "measures": obs_measures,
            "annotations": [{"name": "PBI_ResultType", "value": "Table"}],
        }
    )

    # ----- fact_forecast --------------------------------------------------
    fc_measures = [
        measure(name, dax, desc)
        for name, table, dax, desc in MEASURES
        if table == "fact_forecast"
    ]
    tables.append(
        {
            "name": "fact_forecast",
            "lineageTag": "fact_forecast.tbl",
            "columns": [
                _col("series_key", "string", "series_key", hidden=True),
                _col("date_key", "string", "date_key", hidden=True),
                _col("forecast", "double", "forecast"),
                _col("lower_80", "double", "lower_80"),
                _col("upper_80", "double", "upper_80"),
                _col("model", "string", "model"),
                _col("model_family", "string", "model_family"),
            ],
            "partitions": [
                _m_partition("fact_forecast", "\n".join(_csv_query("fact_forecast.csv")))
            ],
            "measures": fc_measures,
            "annotations": [{"name": "PBI_ResultType", "value": "Table"}],
        }
    )

    relationships = [
        {
            "name": "obs_indicator",
            "fromTable": "fact_observation",
            "fromColumn": "series_key",
            "toTable": "dim_indicator",
            "toColumn": "series_key",
            "crossFilteringBehavior": "oneDirection",
        },
        {
            "name": "obs_date",
            "fromTable": "fact_observation",
            "fromColumn": "date_key",
            "toTable": "dim_date",
            "toColumn": "date_key",
            "crossFilteringBehavior": "oneDirection",
        },
        {
            "name": "fc_indicator",
            "fromTable": "fact_forecast",
            "fromColumn": "series_key",
            "toTable": "dim_indicator",
            "toColumn": "series_key",
            "crossFilteringBehavior": "oneDirection",
        },
        {
            "name": "fc_date",
            "fromTable": "fact_forecast",
            "fromColumn": "date_key",
            "toTable": "dim_date",
            "toColumn": "date_key",
            "crossFilteringBehavior": "oneDirection",
        },
    ]

    return {
        "name": MODEL_NAME,
        "compatibilityLevel": 1567,
        "model": {
            "culture": "en-ZA",
            "defaultPowerBIDataSourceVersion": "powerBI_V3",
            "dataSources": [
                {
                    "name": "DataFolder",
                    "type": "m",
                    "connectionDetails": "{\"protocol\":\"folder\"}",
                    "credential": {"AuthenticationKind": "Anonymous"},
                }
            ],
            "expressions": [
                {
                    "name": "DataFolder",
                    "kind": "m",
                    "expression": [
                        "// Set this to the folder containing the exported CSV files.",
                        '"C:\\\\sa-insights\\data"',
                    ],
                    "annotations": [
                        {
                            "name": "PBI_IsParameter",
                            "value": "true",
                        }
                    ],
                }
            ],
            "tables": tables,
            "relationships": relationships,
            "annotations": [
                {"name": "PBI_QueryOrder", "value": json.dumps(list(frames))},
            ],
        },
    }


# ---------------------------------------------------------------------------
# README + packaging
# ---------------------------------------------------------------------------
def readme(frames: dict[str, pd.DataFrame]) -> str:
    obs = len(frames["fact_observation"])
    fc = len(frames["fact_forecast"])
    ind = len(frames["dim_indicator"])
    dates = len(frames["dim_date"])
    generated = datetime.now(timezone.utc).strftime("%d %B %Y")

    return f"""# SA Insights — Power BI model

A ready-to-import semantic model for the SA Insights South African macro data.
Generated {generated} from the same definitions used by the
[Apache Ossie](../semantic/sa_insights.ossie.yaml) model.

## What is in this bundle

| File | Rows | Purpose |
|------|------|---------|
| `dim_indicator.csv` | {ind:,} | One row per indicator: label, category, unit, frequency, provider. |
| `dim_date.csv` | {dates:,} | Monthly calendar: year, quarter, month, sort keys. |
| `fact_observation.csv` | {obs:,} | Measured values. `series_key` + `date_key` + `value`. |
| `fact_forecast.csv` | {fc:,} | Forecasts with an 80% interval and the model that produced them. |
| `sa_insights.model.bim` | — | Tabular model: tables, relationships, hierarchy and measures. |
| `measures.dax` | {len(MEASURES)} | The same measures as readable DAX. |

## Why one long fact table

The Ossie document uses one dataset per indicator. In Power BI that would mean
29 near-identical tables and a field list nobody can navigate. This export
instead uses a single `fact_observation` table with `series_key` as the foreign
key, which gives you one compact model and natural slicers. No information is
lost — every indicator is still individually selectable.

## Option A — open the model definition (fastest)

1. Open **Tabular Editor 2** (free) or Power BI Desktop with the *Tabular
   Editor 3* external tool installed.
2. Open `sa_insights.model.bim`.
3. In the model, edit the `DataFolder` expression to point at this folder.
4. Click **Process** → **Process Full** to load the CSVs.
5. Save as a Power BI file, or deploy to a workspace.

## Option B — import the CSVs manually

1. Extract this bundle to a folder.
2. Power BI Desktop → **Get data** → **Text/CSV**.
3. Import all four CSVs. Choose *Transform Data* and confirm `date` is a Date
   type and `value` / `forecast` / `lower_80` / `upper_80` are Decimal Number.
4. In Model view, create these relationships:

   | From | To | Cardinality |
   |------|-----|-------------|
   | `fact_observation[series_key]` | `dim_indicator[series_key]` | many-to-one |
   | `fact_observation[date_key]` | `dim_date[date_key]` | many-to-one |
   | `fact_forecast[series_key]` | `dim_indicator[series_key]` | many-to-one |
   | `fact_forecast[date_key]` | `dim_date[date_key]` | many-to-one |

5. Select `dim_date`, then **Table tools → Mark as date table**, column `date`.
   This is required by the year-on-year measures.
6. Open `measures.dax` and paste each measure into the table named in its
   comment.

## The measures

| Measure | What it answers |
|---------|-----------------|
| `Latest Value` | The most recent published figure for the selection. |
| `Latest Period` | What date that figure refers to. |
| `Previous Value`, `Change`, `Change %` | Movement since the prior observation. |
| `Value YoY Change`, `Value YoY %` | Movement versus a year earlier. |
| `Average Value`, `Volatility` | Central tendency and dispersion. |
| `Coverage Years` | How much history is behind the number. |
| `Indicator Count` | How many indicators are in scope. |
| `Real Policy Rate` | Repo rate less CPI — is policy tight in real terms? |
| `Credit Spread` | Lending rate less repo rate. |
| `Trade Balance (US$ m)` | Exports less imports. |
| `Forecast End Value` | Where the projection lands at the horizon. |
| `Forecast Interval Width` | How uncertain that projection is. |

Watch out for `Latest Value` in a table visual: it returns a single value per
indicator, so always put `dim_indicator[Indicator]` on rows. Slicing by
`dim_date[year]` changes which observation counts as "latest" within that year,
which is usually what you want.

## Suggested starting report

- **Card** — `Latest Value` filtered to one indicator, with `Latest Period` as
  the subtitle.
- **Line chart** — `dim_date[date]` on the axis, `Average Value` as values,
  `dim_indicator[Indicator]` as the legend.
- **Table** — `dim_indicator[Indicator]` on rows, with `Latest Value`,
  `Value YoY %` and `Coverage Years`.
- **Slicers** — `dim_indicator[Category]`, `dim_indicator[Frequency]`,
  `dim_date[year]`.

## Caveats

Frequency differs by indicator: CPI and the repo rate are monthly, real GDP is
quarterly, unemployment and the debt ratio are annual. Power BI will happily
plot them on the same axis, but that can imply precision that does not exist.
Filter by `dim_indicator[Frequency]` when mixing indicators, and always show
`Latest Period` alongside a headline number so nobody mistakes an annual figure
for a current one.

Forecasts are statistical extrapolations chosen by a rolling-origin backtest,
not predictions of policy or shocks. `fact_forecast[model]` records which model
won for each series, so the provenance travels with the data.
"""


def build() -> Path:
    frames = build_frames()

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(BUILD_DIR / f"{name}.csv", index=False, encoding="utf-8")

    (BUILD_DIR / "sa_insights.model.bim").write_text(
        json.dumps(build_bim(frames), indent=2), encoding="utf-8"
    )
    (BUILD_DIR / "measures.dax").write_text(dax_file(frames), encoding="utf-8")
    (BUILD_DIR / "README.md").write_text(readme(frames), encoding="utf-8")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUT_DIR / ZIP_NAME
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(BUILD_DIR.iterdir()):
            archive.write(path, arcname=f"sa-insights-powerbi/{path.name}")

    return zip_path


if __name__ == "__main__":
    path = build()
    frames = build_frames()
    print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size:,} bytes)")
    for name, frame in frames.items():
        print(f"  {name + '.csv':26s} {len(frame):>7,} rows")
    print(f"  measures                  {len(MEASURES):>7,}")
