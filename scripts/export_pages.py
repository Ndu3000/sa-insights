#!/usr/bin/env python
"""Export a static dashboard snapshot for the GitHub Pages site.

The live app is a Streamlit server and cannot be hosted on GitHub Pages, which
only serves static files. This script renders the dashboard's core content as
self-contained HTML: KPI figures, interactive Plotly charts and the forecast
lab output, all inlined so the page needs no backend.

Usage:
    python scripts/export_pages.py
"""

from __future__ import annotations

import html
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from sa_insights import store, ui  # noqa: E402
from sa_insights.catalog import BY_KEY, HEADLINE, SERIES, short_label  # noqa: E402
from sa_insights.forecast import (  # noqa: E402
    ForecastError,
    infer_frequency,
    run_forecast,
)
from sa_insights.sources import cache_meta, read_cache  # noqa: E402

# Output directory, overridable so the Docker worker can render into a git
# checkout mounted at /publish instead of the image's own tree.
DOCS = Path(os.getenv("SA_DOCS_OUT", ROOT / "docs"))

# Forecasts showcased on the static page.
SHOWCASE = [
    ("cpi_yoy", 18),
    ("policy_rate", 12),
    ("usd_zar", 12),
    ("gdp_growth", 5),
]


def fig_html(fig, *, include_js: bool = False, div_id: str | None = None) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn" if include_js else False,
        config={"displayModeBar": False, "responsive": True},
        div_id=div_id,
    )


# The first chart on the page must pull in Plotly's runtime. Track it so every
# chart after the first just reuses the global.
_plotly_loaded = {"done": False}


def chart_html(fig) -> str:
    first = not _plotly_loaded["done"]
    _plotly_loaded["done"] = True
    return fig_html(fig, include_js=first)


def collect() -> dict[str, pd.Series]:
    """Load series for the static page.

    Prefers the DuckDB warehouse, which is the system of record since the
    container migration. Falls back to the CSV cache so this script still works
    in a bare checkout that has only run ``collect``.
    """
    data: dict[str, pd.Series] = {}

    try:
        if store.is_initialised():
            data = store.load_observations()
            if data:
                print(f"source: warehouse ({len(data)} series)")
                return data
    except Exception as exc:  # noqa: BLE001 - fall back rather than crash
        print(f"warehouse unavailable ({exc}); falling back to cache")

    for spec in SERIES:
        series = read_cache(spec.key)
        if series is not None and not series.empty:
            data[spec.key] = series
    print(f"source: cache ({len(data)} series)")
    return data


def _history(series_key: str) -> pd.Series | None:
    """Recent history for a chart, from the warehouse first, then the cache."""
    try:
        if store.is_initialised():
            found = store.load_observations([series_key])
            if found.get(series_key) is not None:
                return found[series_key]
    except Exception:  # noqa: BLE001
        pass
    return read_cache(series_key)


def kpi_row(data: dict[str, pd.Series]) -> str:
    cards = []
    for key in HEADLINE:
        if key not in data:
            continue
        spec = BY_KEY[key]
        series = data[key]
        freq = infer_frequency(series.index)
        ppy = {"M": 12, "Q": 4, "A": 1}.get(freq, 1)
        change = None
        if len(series) > ppy:
            prev = series.iloc[-(ppy + 1)]
            if abs(prev) > 1e-9:
                change = series.iloc[-1] - prev

        delta_html = ""
        if change is not None:
            rising = change > 0
            good = (not rising) if spec.higher_is_better is False else rising
            cls = "up" if good else "down"
            arrow = "▲" if rising else "▼"
            delta_html = (
                f'<span class="kpi-delta {cls}">{arrow} {change:+.1f} y/y</span>'
            )
        unit = "" if spec.unit in ("index", "ZAR", "people") else spec.unit
        unit_html = f'<span class="kpi-unit">{html.escape(unit)}</span>' if unit else ""
        spark = ui.svg_sparkline(series, color=ui.ACCENT if (change or 0) >= 0 else ui.CORAL)

        cards.append(
            f"""
      <div class="kpi">
        <div class="kpi-label">{html.escape(short_label(spec))}</div>
        <div class="kpi-value">{ui.fmt(series.iloc[-1], spec.decimals, spec.unit)}{unit_html}</div>
        {delta_html}
        <div class="kpi-foot">{ui.period_label(series.index[-1], freq)}</div>
        {spark}
      </div>"""
        )
    return '<div class="kpi-grid">' + "".join(cards) + "</div>"


def narrative(data: dict[str, pd.Series]) -> str:
    points = []
    if "gdp_growth" in data:
        s = data["gdp_growth"]
        latest, prior = s.iloc[-1], s.iloc[-2] if len(s) > 1 else s.iloc[-1]
        words = "faster" if latest > prior else "slower"
        points.append(
            f"The economy grew <b>{latest:.2f}%</b> in {s.index[-1].year}, "
            f"{words} than the {prior:.2f}% recorded a year earlier."
        )
    if "cpi_yoy" in data:
        s = data["cpi_yoy"].dropna()
        latest = s.iloc[-1]
        points.append(f"Consumer inflation stands at <b>{latest:.2f}%</b>.")
    if {"policy_rate", "cpi_yoy"} <= data.keys():
        real = data["policy_rate"].iloc[-1] - data["cpi_yoy"].iloc[-1]
        stance = "restrictive" if real > 0 else "accommodative"
        points.append(
            f"With the repo rate at <b>{data['policy_rate'].iloc[-1]:.2f}%</b>, the real "
            f"policy rate is <b>{real:+.2f}%</b> — a {stance} stance."
        )
    if "unemployment" in data:
        points.append(f"Unemployment is <b>{data['unemployment'].iloc[-1]:.1f}%</b>.")
    if "usd_zar" in data:
        points.append(f"The rand trades at <b>R{data['usd_zar'].iloc[-1]:.2f}</b> per dollar.")
    if not points:
        return ""
    return (
        '<div class="insight"><div class="insight-title">Read-through</div>'
        f'<div class="insight-body">{" ".join(points)}</div></div>'
    )


def chart_cards(data: dict[str, pd.Series]) -> str:
    specs = [
        ("cpi_yoy", "Inflation path", "monthly CPI, year-on-year %", ui.AMBER, "Line"),
        ("policy_rate", "Policy rate", "repo rate, %", ui.ACCENT, "Line"),
        ("usd_zar", "The rand", "rand per US dollar", ui.SKY, "Line"),
        ("gdp_growth", "Growth momentum", "real GDP growth, %", ui.ACCENT, "Bars"),
    ]
    cards = []
    for key, title, sub, color, style in specs:
        if key not in data:
            continue
        spec = BY_KEY[key]
        series = data[key]
        freq = infer_frequency(series.index)
        view = series.tail(180)
        fig = (
            ui.bar_chart(view, unit=spec.unit, freq=freq, height=260)
            if style == "Bars"
            else ui.trend_line(view, color=color, unit=spec.unit, freq=freq, height=260)
        )
        cards.append(
            f"""
      <div class="panel">
        <div class="panel-title">{title}</div>
        <div class="panel-sub">{sub}</div>
        {chart_html(fig)}
      </div>"""
        )
    return '<div class="grid-2">' + "".join(cards) + "</div>"


FC_PATH = DOCS / "data" / "forecasts.json"


def _fc_chart(spec, entry: dict, height: int = 330) -> str:
    """Plot a precomputed forecast, with history pulled from the local cache."""
    import plotly.graph_objects as go

    dates = pd.to_datetime(entry["periods"])
    hist = _history(spec.key)
    freq = entry["frequency"]
    date_fmt = "%Y" if freq == "A" else "%b %Y"

    fig = go.Figure()

    # Confidence band, drawn as a filled ribbon.
    fig.add_trace(
        go.Scatter(
            x=list(dates) + list(dates)[::-1],
            y=list(entry["upper"]) + list(entry["lower"])[::-1],
            fill="toself",
            fillcolor="rgba(245,185,75,0.14)",
            line={"width": 0},
            name="80% interval",
            hoverinfo="skip",
        )
    )

    if hist is not None and not hist.empty:
        tail = hist.tail(120)
        fig.add_trace(
            go.Scatter(
                x=tail.index,
                y=tail.to_numpy(),
                mode="lines",
                name="Actual",
                line={"color": "#4FB3FF", "width": 2.3},
                hovertemplate=f"%{{x|{date_fmt}}}<br><b>%{{y:,.2f}}</b> {spec.unit}<extra>Actual</extra>",
            )
        )
        bridge_x = [tail.index[-1]] + list(dates)
        bridge_y = [tail.iloc[-1]] + list(entry["forecast"])
    else:
        bridge_x, bridge_y = list(dates), list(entry["forecast"])

    fig.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_y,
            mode="lines",
            name=f"Forecast · {entry['model']}",
            line={"color": "#F5B94B", "width": 2.4, "dash": "dot"},
            hovertemplate=f"%{{x|{date_fmt}}}<br><b>%{{y:,.2f}}</b> {spec.unit}<extra>Forecast</extra>",
        )
    )

    fig.update_layout(**ui.base_layout(height, show_legend=False))

    # Frame the window on recent history plus the projection. Using the bridge
    # list here would start the range at the *last* history point and hide the
    # entire past, so anchor on the history tail instead.
    HISTORY_SHOWN = 60
    if hist is not None and not hist.empty:
        start = hist.tail(HISTORY_SHOWN).index[0]
    else:
        start = dates[0]
    fig.update_xaxes(range=[start, dates[-1]])
    return chart_html(fig)


def forecast_cards(data: dict[str, pd.Series]) -> str:
    """Render forecasts from the build-time JSON rather than recomputing.

    The foundation model cannot run in a browser, so the page reads precomputed
    numbers. This keeps the site static and instant while still surfacing real
    Chronos-Bolt output.
    """
    if not FC_PATH.exists():
        return (
            '<p class="note">Forecasts have not been generated for this build. '
            "Run <code>python scripts/export_forecasts.py</code> first.</p>"
        )

    payload = json.loads(FC_PATH.read_text(encoding="utf-8"))
    entries = payload.get("series", {})

    cards = []
    for key in ["cpi_yoy", "policy_rate", "gdp_growth", "usd_zar", "unemployment"]:
        entry = entries.get(key)
        if not entry:
            continue
        spec = BY_KEY[key]

        skill = entry.get("skill_vs_naive")
        if skill is None:
            quality, cls = "not backtested", ""
        elif skill >= 5:
            quality, cls = f"beats naive by {skill:.0f}%", "ok"
        elif skill > -5:
            quality, cls = "about equal to naive", "stale"
        else:
            quality, cls = f"worse than naive by {abs(skill):.0f}%", "stale"

        is_fm = entry.get("family") == "foundation model"
        fm_chip = '<span class="chip fm">foundation model</span>' if is_fm else ""

        # Percent signs sit tight against the number; word units get a space.
        if spec.unit == "index":
            unit = ""
        elif spec.unit == "%":
            unit = "%"
        else:
            unit = f" {spec.unit}"
        last = entry["last_value"]
        end = entry["forecast"][-1]
        lo, hi = entry["lower"][-1], entry["upper"][-1]

        cards.append(
            f"""
      <div class="panel forecast">
        <div class="panel-head">
          <div>
            <div class="panel-title">{html.escape(entry['label'])}</div>
            <div class="panel-sub">{entry['horizon']} periods ahead · {html.escape(entry['model'])}</div>
          </div>
          <div class="chip-row"><span class="chip {cls}">{quality}</span>{fm_chip}</div>
        </div>
        <div class="fc-stats">
          <div><span class="fc-label">Now</span><span class="fc-value">{ui.fmt(last, spec.decimals)}{html.escape(unit)}</span></div>
          <div><span class="fc-label">Forecast</span><span class="fc-value accent">{ui.fmt(end, spec.decimals)}{html.escape(unit)}</span></div>
          <div><span class="fc-label">80% range</span><span class="fc-value">{ui.fmt(lo, spec.decimals)} – {ui.fmt(hi, spec.decimals)}</span></div>
        </div>
        {_fc_chart(spec, entry)}
      </div>"""
        )
    return "".join(cards)


def fm_winners_table() -> str:
    """Show exactly which series the foundation model won, with its margin.

    The featured forecast cards use fixed, high-profile indicators, and Chronos
    does not win all of them. Listing its actual wins keeps the claim honest and
    checkable instead of leaving it as an assertion in prose.
    """
    if not FC_PATH.exists():
        return ""
    payload = json.loads(FC_PATH.read_text(encoding="utf-8"))
    entries = payload.get("series", {})

    wins = [
        (k, v)
        for k, v in entries.items()
        if v.get("family") == "foundation model" and v.get("mase") is not None
    ]
    if not wins:
        return (
            '<p class="note">The foundation model did not win any series in this '
            "build — the classical models were more accurate on every indicator.</p>"
        )

    wins.sort(key=lambda kv: -(kv[1].get("skill_vs_naive") or 0))

    rows = []
    for key, v in wins:
        # .get, not []: leaderboards written by an older build (or by the
        # live-compute path) may omit "family", and a missing key must not take
        # down the whole page render.
        runner_up = next(
            (
                m
                for m in v.get("leaderboard", [])
                if m.get("family") != "foundation model" and m.get("mase") is not None
            ),
            None,
        )
        runner_txt = (
            f"{html.escape(str(runner_up.get('model', '—')))} ({runner_up['mase']:.2f})"
            if runner_up
            else "—"
        )
        skill = v.get("skill_vs_naive")
        rows.append(
            f"<tr><td>{html.escape(v['label'])}</td>"
            f"<td>{v['frequency']}</td>"
            f"<td class='num'>{v['mase']:.2f}</td>"
            f"<td>{runner_txt}</td>"
            f"<td class='num accent'>"
            f"{'' if skill is None else f'+{skill:.0f}%'}</td></tr>"
        )

    return f"""
      <div class="table-scroll" style="margin-top:1rem">
        <table class="coverage">
          <thead><tr>
            <th>Series won by Chronos-Bolt</th><th>Freq</th>
            <th class="num">Chronos MASE</th><th>Best classical rival (MASE)</th>
            <th class="num">Skill vs naive</th>
          </tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>"""


def foundation_model_summary() -> str:
    """Explain what the foundation model is doing, using real results."""
    if not FC_PATH.exists():
        return ""
    payload = json.loads(FC_PATH.read_text(encoding="utf-8"))
    entries = payload.get("series", {})
    total = len(entries)
    if not total:
        return ""

    fm_wins = [v for v in entries.values() if v.get("family") == "foundation model"]
    active = payload.get("foundation_model_active")
    model_name = payload.get("foundation_model", "amazon/chronos-bolt-small")

    if not active:
        return (
            '<p class="note">The foundation model was unavailable for this build, '
            "so all series were forecast with the classical model pool.</p>"
        )

    winners = ", ".join(
        f"<b>{html.escape(v['label'])}</b> (+{v['skill_vs_naive']:.0f}%)"
        for v in fm_wins
        if v.get("skill_vs_naive") is not None
    )

    return f"""
      <div class="kpi-grid four">
        <div class="kpi">
          <div class="kpi-label">Model</div>
          <div class="kpi-value" style="font-size:1.02rem">Chronos-Bolt</div>
          <div class="kpi-foot">{html.escape(model_name)}</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Series won</div>
          <div class="kpi-value">{len(fm_wins)}<span class="kpi-unit">/ {total}</span></div>
          <div class="kpi-foot">lowest MASE in the backtest</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Parameters</div>
          <div class="kpi-value">48<span class="kpi-unit">m</span></div>
          <div class="kpi-foot">T5 encoder-decoder</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Runs where</div>
          <div class="kpi-value" style="font-size:1.02rem">At build time</div>
          <div class="kpi-foot">page ships numbers, not weights</div>
        </div>
      </div>
      <div class="insight">
        <div class="insight-title">Why the foundation model wins only some series</div>
        <div class="insight-body">
          Chronos-Bolt forecasts zero-shot, so it contributes most where history is
          shortest. It took {len(fm_wins)} of {total} series on lowest MASE
          {("— " + winners) if winners else ""}.
          On long, well-sampled monthly series such as consumer inflation (800+
          observations) classical ETS still wins, because there is enough data for
          seasonal structure to be estimated directly rather than borrowed from a
          prior. Every result shown here comes from the same rolling-origin
          backtest, so selection is decided by measured accuracy, not preference.
        </div>
      </div>"""


def coverage_table(data: dict[str, pd.Series]) -> str:
    rows = []
    for spec in SERIES:
        if spec.key not in data:
            continue
        meta = cache_meta(spec.key)
        series = data[spec.key]
        freq = infer_frequency(series.index)
        rows.append(
            f"<tr><td>{html.escape(spec.label)}</td><td>{html.escape(spec.category)}</td>"
            f"<td>{spec.freq_label}</td><td class='num'>{len(series):,}</td>"
            f"<td>{series.index[0].strftime('%b %Y')}</td>"
            f"<td>{series.index[-1].strftime('%b %Y')}</td>"
            f"<td class='num'>{ui.fmt(series.iloc[-1], spec.decimals)}</td>"
            f"<td>{html.escape(spec.unit)}</td></tr>"
        )
    return (
        "<table class='coverage'><thead><tr><th>Indicator</th><th>Category</th>"
        "<th>Frequency</th><th class='num'>Obs</th><th>From</th><th>To</th>"
        "<th class='num'>Latest</th><th>Unit</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def downloads_section() -> str:
    """Offer the semantic model as portable, tool-ready downloads."""
    download_dir = DOCS / "download"
    pbi_zip = download_dir / "sa-insights-powerbi.zip"

    pbi_card = ""
    if pbi_zip.exists():
        size_kb = pbi_zip.stat().st_size / 1024
        pbi_card = f"""
      <div class="source">
        <h4>Power BI model <span class="chip ok">ready</span></h4>
        <p>
          Star schema as CSV plus a Tabular model (<code>.bim</code>) with
          relationships, a calendar hierarchy and 17 DAX measures. Open it in
          Tabular Editor, or import the CSVs and paste the measures.
          {size_kb:.0f} KB.
        </p>
        <p style="margin-top:.6rem">
          <a class="btn primary" href="download/sa-insights-powerbi.zip" download>
            Download Power BI bundle
          </a>
        </p>
      </div>"""
    else:
        pbi_card = """
      <div class="source">
        <h4>Power BI model</h4>
        <p>Not generated for this build. Run
        <code>python scripts/export_powerbi.py</code> to create it.</p>
      </div>"""

    # GitHub Pages publishes only docs/, so the semantic documents must be
    # copied inside it. Linking to ../semantic/ would 404 on the live site even
    # though it happens to resolve locally.
    # Honour the DOCS override so the container publisher writes into its
    # mounted checkout rather than the image's read-only tree.
    ossie_dir = DOCS / "semantic"
    ossie_links = []
    for name, label in (
        ("sa_insights.ossie.yaml", "Ossie 0.1.1 · YAML"),
        ("sa_insights.ossie.json", "Ossie 0.1.1 · JSON"),
        ("sa_insights.ossie-draft.yaml", "Ossie 0.2.0 draft · YAML"),
        ("ossie-schema.json", "Official JSON Schema"),
    ):
        src = ROOT / "semantic" / name
        if not src.exists():
            continue
        ossie_dir.mkdir(parents=True, exist_ok=True)
        # copyfile, not copy2: copy2 also copies file metadata, and chmod/utime
        # on a bind-mounted volume raises PermissionError. Only the bytes matter
        # for a download.
        shutil.copyfile(src, ossie_dir / name)
        ossie_links.append(
            f'<a class="btn ghost" href="semantic/{name}" download '
            f'style="margin:.25rem .25rem 0 0">{label}</a>'
        )

    return f"""
  <h3 class="sec">Take the model with you</h3>
  <p class="sec-sub">Download it and use it in your own tools.</p>
  <div class="sources">
    {pbi_card}
    <div class="source">
      <h4>Apache Ossie documents</h4>
      <p>
        The vendor-neutral source of truth. Any compliant tool can read these
        directly; the Power BI bundle is a translation of the same definitions.
      </p>
      <p style="margin-top:.6rem">{''.join(ossie_links) or '<em>not generated</em>'}</p>
    </div>
    <div class="source">
      <h4>Precomputed forecasts</h4>
      <p>
        Every projection behind the charts above, as compact JSON — periods,
        point forecast, 80% band, winning model and the full leaderboard.
      </p>
      <p style="margin-top:.6rem">
        <a class="btn ghost" href="data/forecasts.json" download>forecasts.json</a>
      </p>
    </div>
  </div>
  <p class="note">
    The Power BI bundle uses a single long fact table with
    <code>series_key</code> as the foreign key rather than one table per
    indicator — 29 near-identical tables would be unusable in a field list. The
    mapping is lossless; every indicator remains individually selectable.
  </p>"""


def semantic_summary() -> str:
    path = ROOT / "semantic" / "sa_insights.ossie.json"
    if not path.exists():
        return ""
    doc = json.loads(path.read_text(encoding="utf-8"))
    model = doc["semantic_model"][0]
    datasets = model["datasets"]
    facts = sum(1 for d in datasets if d["name"].startswith("fct_"))
    dims = sum(1 for d in datasets if d["name"].startswith("dim_"))
    metrics = len(model.get("metrics", [])) + sum(len(d.get("metrics", [])) for d in datasets)
    return f"""
      <div class="kpi-grid four">
        <div class="kpi"><div class="kpi-label">Ossie version</div><div class="kpi-value">{doc['version']}</div><div class="kpi-foot">released shape</div></div>
        <div class="kpi"><div class="kpi-label">Datasets</div><div class="kpi-value">{len(datasets)}</div><div class="kpi-foot">{facts} fact · {dims} dimension</div></div>
        <div class="kpi"><div class="kpi-label">Relationships</div><div class="kpi-value">{len(model.get('relationships', []))}</div><div class="kpi-foot">fact to time dimension</div></div>
        <div class="kpi"><div class="kpi-label">Metrics</div><div class="kpi-value">{metrics}</div><div class="kpi-foot">dataset and model scoped</div></div>
      </div>"""


def build() -> Path:
    data = collect()
    if not data:
        raise SystemExit("no cached data found — run the app or collector first")

    total_obs = sum(len(s) for s in data.values())
    now = datetime.now(timezone.utc).strftime("%d %B %Y")

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SA Insights — South African statistics dashboard</title>
<meta name="description" content="Live South African macroeconomic indicators with automated time-series forecasting and an Apache Ossie semantic model.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box}}
:root{{
  --ink:#0B1220; --surface:#111C2E; --surface2:#16243A;
  --border:rgba(148,163,184,.16); --text:#E8EEF7; --muted:#8FA3BF; --faint:#5C7089;
  --accent:#3DDC97; --amber:#F5B94B; --coral:#FF6B6B; --sky:#4FB3FF;
}}
html{{scroll-behavior:smooth}}
body{{
  margin:0; background:var(--ink); color:var(--text);
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  -webkit-font-smoothing:antialiased; line-height:1.6;
}}
.wrap{{max-width:1240px; margin:0 auto; padding:0 1.5rem 5rem}}

/* header */
header{{
  border-bottom:1px solid var(--border); margin-bottom:2rem; padding:1.6rem 0 1.4rem;
}}
.brand{{display:flex; align-items:center; gap:.9rem}}
.brand-mark{{
  width:44px;height:44px;border-radius:12px;flex:0 0 auto;
  background:linear-gradient(140deg,#007A4D 0%,#3DDC97 45%,#4FB3FF 100%);
  display:grid;place-items:center;font-weight:800;font-size:1rem;color:#04180F;
  box-shadow:0 8px 24px rgba(61,220,151,.24);
}}
.brand h1{{margin:0;font-size:1.15rem;font-weight:700;letter-spacing:-.02em}}
.brand p{{margin:.15rem 0 0;font-size:.78rem;color:var(--faint)}}
.header-meta{{margin-left:auto;text-align:right;font-size:.72rem;color:var(--faint)}}
.header-meta b{{color:var(--muted)}}

/* hero */
.hero{{padding:1.6rem 0 2rem}}
.hero h2{{
  font-size:2.05rem;font-weight:750;letter-spacing:-.035em;line-height:1.18;margin:0 0 .7rem;
  background:linear-gradient(120deg,#E8EEF7 20%,#3DDC97 100%);
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
}}
.hero p{{font-size:.95rem;color:var(--muted);max-width:62ch;margin:0 0 1.2rem}}
.cta{{display:flex;gap:.6rem;flex-wrap:wrap}}
.btn{{
  display:inline-block;padding:.55rem 1.15rem;border-radius:9px;text-decoration:none;
  font-size:.83rem;font-weight:600;transition:all .16s ease;
}}
.btn.primary{{background:rgba(61,220,151,.12);color:#3DDC97;border:1px solid rgba(61,220,151,.32)}}
.btn.primary:hover{{background:rgba(61,220,151,.2);transform:translateY(-1px)}}
.btn.ghost{{background:rgba(148,163,184,.08);color:var(--muted);border:1px solid var(--border)}}
.btn.ghost:hover{{background:rgba(148,163,184,.14);color:var(--text)}}

/* sections */
h3.sec{{
  font-size:.95rem;font-weight:650;margin:2.4rem 0 .3rem;padding-bottom:.45rem;
  border-bottom:1px solid var(--border);letter-spacing:-.01em;
}}
p.sec-sub{{font-size:.78rem;color:var(--faint);margin:.35rem 0 1.1rem}}

/* kpi */
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:.7rem;margin:1.1rem 0}}
.kpi-grid.four{{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}}
.kpi{{
  background:linear-gradient(165deg,var(--surface2) 0%,var(--surface) 78%);
  border:1px solid var(--border);border-radius:15px;padding:.95rem 1.05rem .7rem;
  transition:border-color .18s ease,transform .18s ease;
}}
.kpi:hover{{border-color:rgba(61,220,151,.34);transform:translateY(-1px)}}
.kpi-label{{
  font-size:.655rem;font-weight:600;letter-spacing:.045em;text-transform:uppercase;
  color:var(--faint);margin-bottom:.32rem;line-height:1.28;min-height:1.72em;
}}
.kpi-value{{
  font-size:1.5rem;font-weight:750;letter-spacing:-.03em;line-height:1.1;
  font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}}
.kpi-value.accent{{color:var(--accent)}}
.kpi-unit{{font-size:.66rem;font-weight:500;color:var(--faint);margin-left:.16rem}}
.kpi-delta{{
  display:inline-block;font-size:.715rem;font-weight:600;margin-top:.3rem;
  padding:.12rem .42rem;border-radius:6px;white-space:nowrap;
}}
.kpi-delta.up{{color:#3DDC97;background:rgba(61,220,151,.12)}}
.kpi-delta.down{{color:#FF6B6B;background:rgba(255,107,107,.12)}}
.kpi-foot{{font-size:.655rem;color:var(--faint);margin-top:.34rem}}

/* panels */
.grid-2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:.9rem}}
.panel{{
  background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:1rem 1.1rem .6rem;margin-bottom:.9rem;overflow:hidden;
}}
.panel-title{{font-size:.82rem;font-weight:650;margin:0 0 .12rem}}
.panel-sub{{font-size:.71rem;color:var(--faint);margin:0 0 .4rem}}
.panel-head{{display:flex;align-items:flex-start;justify-content:space-between;gap:.6rem}}
.forecast{{padding-bottom:.4rem}}

/* insight */
.insight{{
  background:linear-gradient(135deg,rgba(61,220,151,.09),rgba(79,179,255,.05));
  border:1px solid rgba(61,220,151,.26);border-left:3px solid var(--accent);
  border-radius:11px;padding:.9rem 1.05rem;margin:1.1rem 0;
}}
.insight-title{{
  font-size:.7rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
  color:var(--accent);margin-bottom:.3rem;
}}
.insight-body{{font-size:.86rem;line-height:1.6}}
.insight-body b{{color:#3DDC97;font-weight:650}}

/* chips */
.chip{{
  font-size:.685rem;font-weight:550;color:var(--muted);
  background:rgba(148,163,184,.09);border:1px solid var(--border);
  border-radius:999px;padding:.2rem .62rem;white-space:nowrap;
}}
.chip.ok{{color:#3DDC97;border-color:rgba(61,220,151,.3);background:rgba(61,220,151,.09)}}
.chip.stale{{color:#F5B94B;border-color:rgba(245,185,75,.3);background:rgba(245,185,75,.09)}}
.chip.fm{{color:#9B8CFF;border-color:rgba(155,140,255,.34);background:rgba(155,140,255,.1)}}
.chip-row{{display:flex;gap:.35rem;flex-wrap:wrap;justify-content:flex-end}}

/* forecast stats */
.fc-stats{{display:flex;gap:1.4rem;flex-wrap:wrap;margin:.5rem 0 .2rem}}
.fc-stats>div{{display:flex;flex-direction:column}}
.fc-label{{font-size:.65rem;text-transform:uppercase;letter-spacing:.05em;color:var(--faint)}}
.fc-value{{font-size:.95rem;font-weight:650;font-variant-numeric:tabular-nums}}
.fc-value.accent{{color:var(--accent)}}
/* Percent units read better tight against the number. */
.fc-value:has(+ .fc-unit){{margin-right:.05rem}}

/* coverage table */
.table-scroll{{overflow-x:auto;border:1px solid var(--border);border-radius:12px}}
table.coverage{{width:100%;border-collapse:collapse;font-size:.79rem;min-width:760px}}
table.coverage th{{
  text-align:left;padding:.6rem .8rem;background:var(--surface2);
  font-size:.68rem;font-weight:650;letter-spacing:.05em;text-transform:uppercase;
  color:var(--faint);border-bottom:1px solid var(--border);white-space:nowrap;
}}
table.coverage td{{padding:.5rem .8rem;border-bottom:1px solid rgba(148,163,184,.07)}}
table.coverage tr:last-child td{{border-bottom:none}}
table.coverage tbody tr:hover{{background:rgba(148,163,184,.04)}}
table.coverage td.num{{text-align:right;font-variant-numeric:tabular-nums}}
table.coverage td.accent{{color:var(--accent);font-weight:600}}

/* cards for sources */
.sources{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:.8rem}}
.source{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1rem 1.1rem}}
.source h4{{margin:0 0 .3rem;font-size:.85rem;font-weight:650}}
.source p{{margin:0;font-size:.78rem;color:var(--faint);line-height:1.55}}

/* footer */
footer{{
  margin-top:3rem;padding-top:1.4rem;border-top:1px solid var(--border);
  font-size:.72rem;color:var(--faint);text-align:center;line-height:1.7;
}}
footer a{{color:var(--muted)}}
.note{{
  font-size:.75rem;color:var(--faint);border-left:2px solid var(--border);
  padding:.2rem 0 .2rem .8rem;margin:1rem 0;
}}
@media(max-width:640px){{
  .hero h2{{font-size:1.55rem}}
  .wrap{{padding:0 1rem 3rem}}
  .grid-2{{grid-template-columns:1fr}}
  .header-meta{{display:none}}
}}
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="brand">
      <div class="brand-mark">SA</div>
      <div>
        <h1>SA Insights</h1>
        <p>South African statistics · live data · forecast lab</p>
      </div>
      <div class="header-meta">
        <b>{total_obs:,}</b> observations<br>
        {len(data)} series · snapshot {now}
      </div>
    </div>
  </header>

  <section class="hero">
    <h2>South Africa's economy, measured and projected.</h2>
    <p>
      {len(data)} indicators drawn directly from the South African Reserve Bank,
      Stats SA (via SARB), the World Bank, the IMF and FRED —
      covering growth, prices, labour, rates, the rand and the fiscal position.
      Each forecast is chosen by rolling-origin backtesting rather than assumed,
      and the metrics are published as a portable Apache Ossie semantic model.
    </p>
    <div class="cta">
      <a class="btn primary" href="https://github.com/nduduzo-phili/sa-insights">View source on GitHub</a>
      <a class="btn ghost" href="#forecasts">See the forecasts</a>
      <a class="btn ghost" href="#coverage">Data coverage</a>
    </div>
  </section>

  <h3 class="sec">At a glance</h3>
  <p class="sec-sub">Latest published values. Change compares with the same period a year earlier.</p>
  {kpi_row(data)}
  {narrative(data)}

  <h3 class="sec">Trends</h3>
  <p class="sec-sub">Interactive — hover for values, drag to zoom.</p>
  {chart_cards(data)}

  <h3 class="sec" id="forecasts">Forecast lab</h3>
  <p class="sec-sub">
    One-shot models selected by MASE on rolling-origin backtests at the horizon
    shown. The Apache Chronos-Bolt foundation model competes here alongside the
    classical baselines. Bands are 80% prediction intervals.
  </p>
  {foundation_model_summary()}
  {fm_winners_table()}
  {forecast_cards(data)}
  <p class="note">
    MASE below 1.0 means the model outperformed a naive forecast across the
    backtest windows. Where MASE exceeds 1.0 the dashboard says so rather than
    presenting the projection as reliable.
  </p>

  <h3 class="sec">Semantic model</h3>
  <p class="sec-sub">
    Metrics and dimensions published in Apache Ossie (incubating) format so the
    logic travels between BI tools, query engines and AI agents.
  </p>
  {semantic_summary()}
  <p class="note">
    Two shapes are emitted: <b>0.1.1</b> (released, <code>semantic_model</code> array)
    and <b>0.2.0.dev0</b> (flat draft, validated against the official JSON Schema).
  </p>
  {downloads_section()}

  <h3 class="sec">Sources</h3>
  <p class="sec-sub">All key-free and read-only. Locally cached for 12 hours.</p>
  <div class="sources">
    <div class="source">
      <h4>South African Reserve Bank <span class="chip ok">primary</span></h4>
      <p>
        Official Web API. Policy and prime rates, bond yields, the rand, real
        effective exchange rate, CPI and PPI, retail and manufacturing volumes,
        mining output, vehicle sales and the leading indicator. Updated within
        days of each release.
      </p>
    </div>
    <div class="source">
      <h4>Stats SA (via SARB)</h4>
      <p>
        Consumer and producer inflation originate at Statistics South Africa;
        both are carried on the SARB Web API, which is how this dashboard gets
        them without scraping the Stats SA site.
      </p>
    </div>
    <div class="source">
      <h4>World Bank Open Data</h4>
      <p>Annual macroeconomic, labour, fiscal and social indicators. Published under CC BY 4.0.</p>
    </div>
    <div class="source">
      <h4>IMF — IFS &amp; WEO</h4>
      <p>Long-run history and published projections to 2030, via the DBnomics mirror.</p>
    </div>
    <div class="source">
      <h4>FRED</h4>
      <p>Quarterly real GDP from the Federal Reserve Bank of St. Louis.</p>
    </div>
  </div>
  <p class="note">
    Recency varies by indicator, and the table below always states each series'
    last observed period. Rate, rand and inflation data are current to within a
    few weeks. Annual indicators sourced from the World Bank — GDP growth,
    unemployment, population, the debt ratio — reflect the latest <em>published
    annual</em> figure, which is inherently a year or more behind. Treat an
    annual number as a structural benchmark, not a current reading.
  </p>

  <h3 class="sec" id="coverage">Data coverage</h3>
  <p class="sec-sub">Every series in the model, with its observed span and latest value.</p>
  <div class="table-scroll">
    {coverage_table(data)}
  </div>

  <footer>
    <p>
      SA Insights · built with Streamlit, Plotly and statsmodels ·
      <a href="https://github.com/nduduzo-phili/sa-insights">source</a>
    </p>
    <p>
      Forecasts are statistical extrapolations of past patterns, not predictions of
      policy, shocks or structural breaks. This is not investment advice.
    </p>
    <p>Snapshot generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.</p>
  </footer>

</div>
</body>
</html>
"""

    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "index.html"
    out.write_text(page, encoding="utf-8")

    # Tell Jekyll to serve files as-is.
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")

    return out


if __name__ == "__main__":
    path = build()
    size = path.stat().st_size
    print(f"wrote {path.relative_to(ROOT)}  ({size:,} bytes)")
