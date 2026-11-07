"""Generate an Apache Ossie semantic model for the SA Insights data.

Apache Ossie (incubating), formerly Open Semantic Interchange, is a
vendor-neutral specification for exchanging semantic models between BI tools,
AI agents and query engines. This module emits the project's metrics and
dimensions in that format so the model stays portable and reviewable as the
project grows.

Target version
--------------
We emit **0.1.1** — the released revision, which wraps the model in a
`semantic_model` array and is what dbt v1.12 and other consumers accept. The
in-development `0.2.0.dev0` schema is explicitly marked unstable upstream, so
we do not target it.

Modelling shape
---------------
A small star schema:

* `dim_date_month` / `dim_date_quarter` / `dim_date_year` — one time
  dimension per grain, with derived calendar fields.
* `fct_<series>` — one fact dataset per indicator, holding `period` and
  `value`, plus the series' own metadata as descriptive fields.
* Relationships from each fact to its grain's time dimension.
* Metrics: dataset-scoped measures on each fact (`value_latest`,
  `value_average`, ...) and a handful of model-scoped measures that span
  datasets, such as the real policy rate.

Run via:  python scripts/build_semantic_model.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .catalog import SERIES, SeriesSpec, short_label
from .sources import cache_meta

OSSIE_VERSION = "0.1.1"
DRAFT_VERSION = "0.2.0.dev0"
MODEL_NAME = "sa_insights"

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "semantic"
SCHEMA_PATH = ROOT / "semantic" / "ossie-schema.json"

GRAIN_DATASET = {"M": "dim_date_month", "Q": "dim_date_quarter", "A": "dim_date_year"}

# Per-grain extra calendar fields, expressed as portable Ossie SQL.
GRAIN_FIELDS = {
    "M": [
        ("year", "EXTRACT(YEAR FROM period)", "Integer"),
        ("quarter", "EXTRACT(QUARTER FROM period)", "Integer"),
        ("month_of_year", "EXTRACT(MONTH FROM period)", "Integer"),
    ],
    "Q": [
        ("year", "EXTRACT(YEAR FROM period)", "Integer"),
        ("quarter", "EXTRACT(QUARTER FROM period)", "Integer"),
    ],
    "A": [
        ("year", "EXTRACT(YEAR FROM period)", "Integer"),
    ],
}


def _slug(key: str) -> str:
    return f"fct_{key}"


def _dimension_datasets() -> list[dict]:
    out = []
    for grain, dataset in GRAIN_DATASET.items():
        fields = [
            {
                "name": "period",
                "expression": {"sql": "period"},
                "datatype": "Date",
                "dimension": {"is_time": True},
                "description": "First day of the period this row represents.",
            },
            {
                "name": "grain",
                "expression": {"sql": "grain"},
                "datatype": "String",
                "description": "Frequency of the series this period belongs to.",
            },
        ]
        for name, sql, datatype in GRAIN_FIELDS[grain]:
            fields.append(
                {
                    "name": name,
                    "expression": {"sql": sql},
                    "datatype": datatype,
                    "dimension": {"is_time": False},
                    "description": f"Calendar {name.replace('_', ' ')} of the period.",
                }
            )
        out.append(
            {
                "name": dataset,
                "source": f"sa_insights.{dataset}",
                "primary_key": ["period"],
                "unique_keys": [["period"]],
                "description": {
                    "dim_date_month": "Monthly calendar dimension.",
                    "dim_date_quarter": "Quarterly calendar dimension.",
                    "dim_date_year": "Annual calendar dimension.",
                }[dataset],
                "ai_context": {
                    "synonyms": ["calendar", "date dimension", "time dimension"],
                    "examples": [f"Group {dataset.split('_')[-1]} data by year"],
                },
                "fields": fields,
            }
        )
    return out


def _fact_dataset(spec: SeriesSpec, meta: dict) -> dict:
    dataset = _slug(spec.key)
    fields = [
        {
            "name": "period",
            "expression": {"sql": "period"},
            "datatype": "Date",
            "dimension": {"is_time": True},
            "description": f"Observation period for {spec.label}.",
            "ai_context": {"synonyms": ["date", "month", "quarter", "year"]},
        },
        {
            "name": "value",
            "expression": {"sql": "value"},
            "datatype": "Float",
            "dimension": {"is_time": False},
            "description": f"Observed value of {spec.label}, measured in {spec.unit}.",
            "ai_context": {"synonyms": ["reading", "observation", "level"]},
        },
        {
            "name": "series_key",
            "expression": {"sql": "series_key"},
            "datatype": "String",
            "dimension": {"is_time": False},
            "description": "Stable identifier of the indicator.",
        },
        {
            "name": "unit",
            "expression": {"sql": "unit"},
            "datatype": "String",
            "dimension": {"is_time": False},
            "description": "Unit of measurement for the value field.",
        },
    ]

    metrics = [
        {
            "name": "value_average",
            "expression": {"sql": "AVG(value)"},
            "datatype": "Float",
            "description": f"Mean {spec.label} across the selected periods.",
            "ai_context": {"synonyms": ["mean", "average"]},
        },
        {
            "name": "value_min",
            "expression": {"sql": "MIN(value)"},
            "datatype": "Float",
            "description": f"Lowest {spec.label} in the selected periods.",
        },
        {
            "name": "value_max",
            "expression": {"sql": "MAX(value)"},
            "datatype": "Float",
            "description": f"Highest {spec.label} in the selected periods.",
        },
        {
            "name": "value_standard_deviation",
            "expression": {"sql": "STDDEV(value)"},
            "datatype": "Float",
            "description": f"Dispersion of {spec.label} across the selected periods.",
            "ai_context": {"synonyms": ["volatility", "standard deviation"]},
        },
        {
            "name": "observation_count",
            "expression": {"sql": "COUNT(value)"},
            "datatype": "Integer",
            "description": "Number of non-missing observations in the selection.",
        },
    ]

    coverage = ""
    if meta.get("first_period") and meta.get("last_period"):
        coverage = f" Covers {meta['first_period'][:7]} to {meta['last_period'][:7]}."

    return {
        "name": dataset,
        "source": f"sa_insights.{dataset}",
        "primary_key": ["period"],
        "unique_keys": [["period"]],
        "description": f"{spec.description}{coverage}",
        "ai_context": {
            "synonyms": _synonyms(spec),
            "examples": _examples(spec),
        },
        "fields": fields,
        "metrics": metrics,
        "custom_extensions": [
            {
                "vendor_name": "SA_INSIGHTS",
                "data": json.dumps(
                    {
                        "series_key": spec.key,
                        "label": spec.label,
                        "short_label": short_label(spec),
                        "category": spec.category,
                        "unit": spec.unit,
                        "frequency": spec.frequency,
                        "provider": spec.provider,
                        "higher_is_better": spec.higher_is_better,
                        "upstream_indicator": spec.fetch.get("indicator")
                        or spec.fetch.get("series_id")
                        or (spec.fetch.get("dimensions", {}) or {}),
                        "observations": meta.get("observations"),
                        "first_period": meta.get("first_period"),
                        "last_period": meta.get("last_period"),
                        "last_value": meta.get("last_value"),
                    },
                    default=str,
                ),
            }
        ],
    }


def _synonyms(spec: SeriesSpec) -> list[str]:
    base = [spec.label.lower(), short_label(spec).lower()]
    extra = {
        "cpi_yoy": ["inflation", "cost of living", "price growth"],
        "policy_rate": ["repo rate", "interest rate", "MPC rate"],
        "usd_zar": ["rand", "exchange rate", "dollar rand"],
        "gdp_growth": ["economic growth", "GDP", "output growth"],
        "unemployment": ["joblessness", "unemployment rate"],
        "equity_index": ["stock market", "JSE", "share prices"],
        "govt_debt": ["debt to GDP", "public debt"],
        "current_account": ["trade balance", "external balance"],
    }.get(spec.key, [])
    seen, out = set(), []
    for term in base + extra:
        if term not in seen:
            seen.add(term)
            out.append(term)
    return out


def _examples(spec: SeriesSpec) -> list[str]:
    grain = {"M": "month", "Q": "quarter", "A": "year"}.get(spec.frequency, "period")
    examples = [f"What is the latest {short_label(spec).lower()}?"]
    examples.append(f"Show {spec.label.lower()} by {grain} over the last five years")
    if spec.higher_is_better is not None:
        trend = "improving" if spec.higher_is_better else "rising pressure"
        examples.append(f"Is {spec.label.lower()} {trend}?")
    return examples


def _relationships(fact_names: list[tuple[str, str]]) -> list[dict]:
    out = []
    for dataset, grain in fact_names:
        dim = GRAIN_DATASET[grain]
        out.append(
            {
                "name": f"{dataset}_to_{dim}",
                "from": dataset,
                "to": dim,
                "from_columns": ["period"],
                "to_columns": ["period"],
                "ai_context": {
                    "instructions": (
                        "Join on period to attach calendar attributes such as year "
                        "and quarter to this indicator."
                    )
                },
            }
        )
    return out


def _model_metrics(present: set[str], grains: dict[str, str]) -> list[dict]:
    """Model-scoped measures that genuinely span datasets."""
    metrics: list[dict] = []

    monthly = {"policy_rate", "cpi_yoy", "usd_zar", "equity_index", "lending_rate", "bond_yield"}
    if {"policy_rate", "cpi_yoy"} <= present and grains.get("policy_rate") == grains.get("cpi_yoy"):
        metrics.append(
            {
                "name": "real_policy_rate",
                "expression": {
                    "sql": "AVG(fct_policy_rate.value) - AVG(fct_cpi_yoy.value)"
                },
                "datatype": "Float",
                "description": (
                    "The repo rate less consumer inflation, in percentage points. "
                    "Positive values mean monetary policy is restrictive in real terms."
                ),
                "ai_context": {
                    "synonyms": ["real repo rate", "real interest rate"],
                    "examples": ["Is monetary policy tight in real terms?"],
                },
            }
        )

    if {"lending_rate", "policy_rate"} <= present and grains.get("lending_rate") == grains.get(
        "policy_rate"
    ):
        metrics.append(
            {
                "name": "credit_spread",
                "expression": {
                    "sql": "AVG(fct_lending_rate.value) - AVG(fct_policy_rate.value)"
                },
                "datatype": "Float",
                "description": (
                    "Bank lending rate less the policy rate, in percentage points."
                ),
                "ai_context": {"synonyms": ["lending spread", "bank margin"]},
            }
        )

    if {"bond_yield", "cpi_yoy"} <= present and grains.get("bond_yield") == grains.get("cpi_yoy"):
        metrics.append(
            {
                "name": "real_bond_yield",
                "expression": {
                    "sql": "AVG(fct_bond_yield.value) - AVG(fct_cpi_yoy.value)"
                },
                "datatype": "Float",
                "description": (
                    "Long government bond yield less consumer inflation, in "
                    "percentage points."
                ),
                "ai_context": {"synonyms": ["real yield"]},
            }
        )

    if {"exports_usd", "imports_usd"} <= present and grains.get("exports_usd") == grains.get(
        "imports_usd"
    ):
        metrics.append(
            {
                "name": "trade_balance",
                "expression": {"sql": "AVG(fct_exports_usd.value) - AVG(fct_imports_usd.value)"},
                "datatype": "Float",
                "description": (
                    "Goods exports less goods imports, in millions of US dollars."
                ),
                "ai_context": {
                    "synonyms": ["net exports", "trade surplus", "trade deficit"],
                    "examples": ["Does South Africa run a trade surplus?"],
                },
            }
        )

    if present:
        first = sorted(_slug(k) for k in present)[0]
        metrics.append(
            {
                "name": "indicator_count",
                "expression": {"sql": f"COUNT(DISTINCT {first}.series_key)"},
                "datatype": "Integer",
                "description": (
                    "Number of distinct indicators represented in the model. "
                    "Per-dataset counts are available as each fact's "
                    "observation_count metric."
                ),
            }
        )
    return metrics


def build_model(*, include_all: bool = True, version: str = OSSIE_VERSION) -> dict:
    """Construct the Ossie document as a plain dict.

    `version` selects the document shape:

    * ``0.1.1`` (default) — wraps the model in a ``semantic_model`` array.
    * ``0.2.0.dev0`` — the flat draft shape that the published JSON Schema
      validates; the model sits directly at the root.

    The model contents are identical; only the wrapper and version change.
    """
    facts: list[dict] = []
    fact_grains: list[tuple[str, str]] = []
    present: set[str] = set()
    grains: dict[str, str] = {}

    for spec in SERIES:
        meta = cache_meta(spec.key)
        if not meta.get("observations"):
            continue
        facts.append(_fact_dataset(spec, meta))
        fact_grains.append((_slug(spec.key), spec.frequency))
        present.add(spec.key)
        grains[spec.key] = spec.frequency

    datasets = _dimension_datasets() + facts

    model = {
        "name": MODEL_NAME,
        "description": (
            "South African macroeconomic indicators with forecasting. One fact "
            "dataset per indicator over monthly, quarterly or annual time "
            "dimensions, sourced from the World Bank, IMF and FRED."
        ),
        "ai_context": {
            "instructions": (
                "Use this model for questions about the South African economy: "
                "growth, inflation, employment, interest rates, the rand, trade "
                "and the fiscal position. Each fact dataset holds one indicator. "
                "Join to the matching dim_date_* dataset on period for calendar "
                "attributes. Values are published statistics, not real-time market "
                "data; check the observation coverage before drawing conclusions."
            ),
            "synonyms": [
                "South Africa",
                "SA macro",
                "SA economy",
                "South African statistics",
            ],
            "examples": [
                "What is South Africa's inflation rate?",
                "How has the rand moved against the dollar?",
                "Compare GDP growth and unemployment over the last decade",
                "Is the repo rate restrictive in real terms?",
            ],
        },
        "datasets": datasets,
        "relationships": _relationships(fact_grains),
        "metrics": _model_metrics(present, grains),
        "custom_extensions": [
            {
                "vendor_name": "SA_INSIGHTS",
                "data": json.dumps(
                    {
                        "generated_at": datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                        "document_version": version,
                        "series_count": len(facts),
                        "generator": "scripts/build_semantic_model.py",
                        "note": (
                            "0.1.1 is the released shape (semantic_model array). "
                            "0.2.0.dev0 is the flat draft that the published JSON "
                            "Schema validates; the upstream schema is unstable."
                        ),
                    }
                ),
            }
        ],
    }

    if version == OSSIE_VERSION:
        return {"version": OSSIE_VERSION, "semantic_model": [model]}

    return {"version": version, **model}


def write_model(directory: Path = OUT_DIR, *, include_all: bool = True) -> list[Path]:
    """Write YAML and JSON in both document shapes. Returns the paths written."""
    import yaml

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for label, version in (
        ("sa_insights.ossie", OSSIE_VERSION),
        ("sa_insights.ossie-draft", DRAFT_VERSION),
    ):
        document = build_model(include_all=include_all, version=version)
        json_path = directory / f"{label}.json"
        yaml_path = directory / f"{label}.yaml"
        json_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        yaml_path.write_text(
            yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )
        written += [json_path, yaml_path]

    return written


def validate_schema(document: dict, schema_path: Path | None = None) -> list[str]:
    """Validate the flat draft document against the official Ossie JSON Schema.

    The schema only exists for ``0.2.0.dev0``; it is optional so the project
    still builds offline. Returns a list of human-readable problems.
    """
    if schema_path is None:
        schema_path = SCHEMA_PATH
    if not schema_path.exists():
        return ["schema file not present — skipped official schema validation"]
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema not installed — skipped official schema validation"]

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    problems = []
    for error in sorted(validator.iter_errors(document), key=lambda e: list(e.path)):
        location = "/".join(str(p) for p in error.absolute_path) or "<root>"
        problems.append(f"{location}: {error.message}")
    return problems


def validate(document: dict) -> list[str]:
    """Lightweight structural checks against the 0.1.1 shape."""
    problems: list[str] = []
    if document.get("version") != OSSIE_VERSION:
        problems.append(f"version is {document.get('version')!r}, expected {OSSIE_VERSION!r}")

    models = document.get("semantic_model")
    if not isinstance(models, list) or not models:
        return problems + ["semantic_model must be a non-empty array"]

    model = models[0]
    for required in ("name", "datasets"):
        if not model.get(required):
            problems.append(f"model missing required field {required!r}")

    dataset_names = set()
    for ds in model.get("datasets", []):
        for required in ("name", "source"):
            if not ds.get(required):
                problems.append(f"dataset {ds.get('name')!r} missing {required!r}")
        name = ds.get("name")
        if name in dataset_names:
            problems.append(f"duplicate dataset name {name!r}")
        dataset_names.add(name)

        field_names = set()
        for field in ds.get("fields", []):
            if not field.get("name") or not field.get("expression", {}).get("sql"):
                problems.append(f"{name}: field missing name or expression.sql")
            if field.get("name") in field_names:
                problems.append(f"{name}: duplicate field {field.get('name')!r}")
            field_names.add(field.get("name"))

        for key in ds.get("primary_key", []) or []:
            if key not in field_names:
                problems.append(f"{name}: primary_key {key!r} is not a declared field")

    for rel in model.get("relationships", []):
        if rel.get("from") not in dataset_names:
            problems.append(f"relationship {rel.get('name')}: unknown from dataset")
        if rel.get("to") not in dataset_names:
            problems.append(f"relationship {rel.get('name')}: unknown to dataset")
        if len(rel.get("from_columns", [])) != len(rel.get("to_columns", [])):
            problems.append(f"relationship {rel.get('name')}: column count mismatch")

    return problems
