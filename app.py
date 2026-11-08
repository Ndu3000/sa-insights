"""SA Insights — dashboard entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from sa_insights import chronos_fm, downloads, refresh, store, tracking, ui
from sa_insights.catalog import (
    BY_KEY,
    CATEGORIES,
    FORECAST_DEFAULT,
    HEADLINE,
    SERIES,
    short_label,
)
from sa_insights.forecast import (
    FM_ENABLED,
    ForecastError,
    default_window,
    infer_frequency,
    model_names,
    run_forecast,
)

st.set_page_config(
    page_title="SA Insights",
    page_icon="🇿🇦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(ui.CSS, unsafe_allow_html=True)

PROVIDER_LABEL = {
    "worldbank": "World Bank",
    "fred": "FRED",
    "dbnomics": "IMF (via DBnomics)",
    "sarb": "SARB",
    "imf": "IMF",
}


# ---------------------------------------------------------------------------
# Data loading — reads the warehouse, never the network on page load
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def load_warehouse(_cache_key: str) -> dict[str, pd.Series]:
    """Read all observations from DuckDB.

    Keyed on the last ingest timestamp so the cache invalidates the moment new
    data lands, rather than waiting for a TTL to expire.
    """
    return store.load_observations()


@st.cache_data(show_spinner=False, ttl=300)
def warehouse_summary(_cache_key: str) -> dict:
    return store.data_summary()


@st.cache_data(show_spinner=False, ttl=300)
def warehouse_metadata(_cache_key: str) -> pd.DataFrame:
    return store.indicator_metadata()


@st.cache_data(show_spinner=False)
def cached_forecast(
    key: str, horizon: int, confidence: float, model: str, window: int, data_token: str
):
    """Fit a forecast on demand for the lab tab.

    Uses the warehouse series, so what is projected matches what is plotted
    elsewhere in the app.
    """
    series = store.load_observations([key]).get(key)
    if series is None or series.empty:
        raise ForecastError(f"no observations stored for {key}")
    return run_forecast(
        series,
        horizon,
        confidence=confidence,
        model=model,
        max_history=window if window > 0 else None,
    )


def data_token() -> str:
    """Cheap cache key derived from the newest ingest."""
    summary = store.data_summary()
    return str(summary.get("last_ingest") or "empty")


def provider_of(spec) -> str:
    if spec.provider == "dbnomics":
        return PROVIDER_LABEL["dbnomics"]
    return PROVIDER_LABEL.get(spec.provider, spec.provider)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def series_frame(mapping: dict[str, pd.Series]) -> pd.DataFrame:
    if not mapping:
        return pd.DataFrame()
    return pd.DataFrame(mapping).sort_index()


def yoy_change(series: pd.Series, periods: int) -> float | None:
    if len(series) <= periods:
        return None
    prev = series.iloc[-(periods + 1)]
    if abs(prev) < 1e-9:
        return None
    return series.iloc[-1] - prev


def pct_change(series: pd.Series, periods: int) -> float | None:
    if len(series) <= periods:
        return None
    prev = series.iloc[-(periods + 1)]
    if abs(prev) < 1e-9:
        return None
    return (series.iloc[-1] / prev - 1.0) * 100.0


def periods_per_year(freq: str) -> int:
    return {"M": 12, "Q": 4, "A": 1}.get(freq, 1)


def _tile_label(spec) -> str:
    """KPI tiles have little room, so use the terse label where one exists."""
    return short_label(spec)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar(summary: dict, meta: pd.DataFrame, stored_keys: set[str]):
    with st.sidebar:
        st.markdown(
            '<p style="font-size:.7rem;font-weight:700;letter-spacing:.1em;'
            'text-transform:uppercase;color:#5C7089;margin-bottom:.2rem">Warehouse</p>',
            unsafe_allow_html=True,
        )

        last = summary.get("last_ingest")
        last_txt = "never" if last is None else pd.Timestamp(last).strftime("%d %b %Y %H:%M")
        st.markdown(
            ui.chips(
                [
                    (f"{summary['series']} series", "ok" if summary["series"] else "fail"),
                    (f"{summary['observations']:,} obs", ""),
                    (f"{summary['forecasts']} runs", ""),
                ]
            ),
            unsafe_allow_html=True,
        )
        st.caption(f"last refresh {last_txt}")

        # Foundation-model status. Surfaced because whether Chronos-Bolt competes
        # materially changes the leaderboard.
        fm = chronos_fm.describe()
        if not FM_ENABLED:
            fm_text, fm_cls = "Foundation model off", "stale"
            fm_hint = "SA_INSIGHTS_FM=0 disables Chronos-Bolt"
        elif fm["available"]:
            fm_text, fm_cls = "Chronos-Bolt ready", "ok"
            fm_hint = fm["model_id"]
        else:
            fm_text, fm_cls = "Foundation model unavailable", "stale"
            fm_hint = (fm["reason"] or "")[:90]
        st.markdown(ui.chips([(fm_text, fm_cls)]), unsafe_allow_html=True)
        st.caption(fm_hint)

        # Tracking status. Distinguishes three states rather than two, because
        # "configured" and "working" are different things: the old code showed a
        # green chip whenever a URI was set, even with no server listening, and
        # rendered the internal Docker hostname as a clickable link that could
        # never resolve from a browser.
        tr = tracking.status()
        if not tr["enabled"]:
            st.markdown(ui.chips([("tracking off", "")]), unsafe_allow_html=True)
            st.caption(tr["reason"] or "")
        elif tracking.probe() is False:
            st.markdown(
                ui.chips([("tracking unreachable", "fail")]), unsafe_allow_html=True
            )
            st.caption(f"cannot reach {tr['tracking_uri']}")
        else:
            st.markdown(ui.chips([("MLflow tracking", "ok")]), unsafe_allow_html=True)
            if tr["ui_url"]:
                st.link_button("Open MLflow", tr["ui_url"], use_container_width=True)
            else:
                st.caption(f"logging to {tr['tracking_uri']}")
                st.caption("set MLFLOW_UI_URL for a browser link")

        if st.button("↻ Refresh all data", use_container_width=True):
            with st.spinner("Fetching upstream data and refitting models…"):
                try:
                    report = refresh.refresh_all(force=True, do_forecast=True)
                except Exception as exc:  # noqa: BLE001 - surfaced in the UI
                    st.error(f"Refresh failed: {exc}")
                else:
                    # Clear every warehouse-backed cache so the next read is fresh.
                    load_warehouse.clear()
                    warehouse_summary.clear()
                    warehouse_metadata.clear()
                    cached_forecast.clear()
                    st.success(
                        f"Refreshed {report.series_loaded}/{report.series_total} series, "
                        f"{report.forecasts_written} forecasts in "
                        f"{report.duration_seconds:.0f}s"
                    )
                    st.rerun()

        st.divider()
        st.markdown(
            '<p style="font-size:.7rem;font-weight:700;letter-spacing:.1em;'
            'text-transform:uppercase;color:#5C7089;margin-bottom:.3rem">Display</p>',
            unsafe_allow_html=True,
        )
        st.session_state.setdefault("smoothing", False)
        st.session_state.setdefault("show_points", False)

        st.toggle("Smooth trend lines", key="smoothing")
        st.toggle("Show data markers", key="show_points")

        st.divider()
        with st.expander("Data sources & licence", expanded=False):
            st.markdown(
                """
**World Bank** — Open Data API, CC BY 4.0.
**IMF** — International Financial Statistics and World Economic Outlook, via the DBnomics mirror.
**FRED** — Federal Reserve Bank of St. Louis, public series.

All calls are key-free and read-only. Values are cached locally for 12 hours;
use **Refresh** to force a re-fetch.
                """
            )
        with st.expander("How forecasts are chosen", expanded=False):
            st.markdown(
                """
Every series is backtested with rolling origins at your chosen horizon. Models
competing include naive, drift, seasonal naive, damped ETS with and without
seasonality, and Theta. The winner is the one with the lowest
**MASE** (mean absolute scaled error) — a scale-free measure where below 1.0
means the model beats a naive forecast.
                """
            )

        missing = [s for s in SERIES if s.key not in stored_keys]
        if missing:
            with st.expander(f"⚠ {len(missing)} series not in the warehouse", expanded=False):
                for spec in missing:
                    st.caption(f"**{spec.label}** — `{spec.key}` has no stored observations")


# ---------------------------------------------------------------------------
# View: Overview
# ---------------------------------------------------------------------------
def view_overview(data: dict[str, pd.Series], specs_by_key):
    st.markdown(
        ui.section("At a glance", "latest published values across the South African economy"),
        unsafe_allow_html=True,
    )

    headline = [k for k in HEADLINE if k in data]
    if not headline:
        st.warning("No headline series are available yet — try **Refresh** in the sidebar.")
        return

    columns = st.columns(len(headline), gap="small")
    for col, key in zip(columns, headline):
        spec = specs_by_key[key]
        series = data[key]
        freq = infer_frequency(series.index)
        ppy = periods_per_year(freq)
        change = yoy_change(series, ppy)
        change_txt = (
            f"{ui.fmt_pct(change)} y/y" if change is not None else "—"
        )
        delta = ui.delta_badge(
            change,
            change_txt,
            invert=(spec.higher_is_better is False),
        )
        spark_color = ui.ACCENT if (change or 0) >= 0 else ui.CORAL
        with col:
            st.markdown(
                ui.kpi_tile(
                    _tile_label(spec),
                    ui.fmt(series.iloc[-1], spec.decimals, spec.unit),
                    "" if spec.unit in ("index", "ZAR", "people") else spec.unit,
                    delta,
                    ui.last_updated(series.index[-1], freq),
                    spark_svg=ui.svg_sparkline(series, color=spark_color),
                ),
                unsafe_allow_html=True,
            )

    # --- Economic read-through -------------------------------------------
    st.markdown(ui.section("What the numbers say"), unsafe_allow_html=True)
    st.markdown(_overview_narrative(data), unsafe_allow_html=True)

    # --- Growth vs inflation ---------------------------------------------
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(ui.panel_open("Growth momentum", "real GDP growth, annual %"), unsafe_allow_html=True)
        if "gdp_growth" in data:
            st.plotly_chart(
                ui.bar_chart(data["gdp_growth"].tail(20), unit="%", height=300),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        st.markdown(ui.PANEL_CLOSE, unsafe_allow_html=True)

    with right:
        st.markdown(ui.panel_open("Inflation path", "monthly CPI, year-on-year %"), unsafe_allow_html=True)
        if "cpi_yoy" in data:
            st.plotly_chart(
                ui.trend_line(data["cpi_yoy"].tail(160), color=ui.AMBER, unit="%", freq="M", height=300),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        st.markdown(ui.PANEL_CLOSE, unsafe_allow_html=True)

    # --- Rates and rand ---------------------------------------------------
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(
            ui.panel_open("Policy rate & the cost of credit", "repo rate against bank lending rate, %"),
            unsafe_allow_html=True,
        )
        rate_cols = {k: data[k] for k in ("policy_rate", "lending_rate", "bond_yield") if k in data}
        if rate_cols:
            frame = series_frame({BY_KEY[k].label: v for k, v in rate_cols.items()})
            st.plotly_chart(
                ui.multi_line(frame.tail(160), unit="%", height=300),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        st.markdown(ui.PANEL_CLOSE, unsafe_allow_html=True)

    with right:
        st.markdown(ui.panel_open("The rand", "rand per US dollar, monthly average"), unsafe_allow_html=True)
        if "usd_zar" in data:
            st.plotly_chart(
                ui.trend_line(data["usd_zar"].tail(160), color=ui.SKY, unit="ZAR", freq="M", height=300),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        st.markdown(ui.PANEL_CLOSE, unsafe_allow_html=True)

    # --- IMF outlook strip ------------------------------------------------
    weo = {k: data[k] for k in data if k.startswith("weo_")}
    if weo:
        st.markdown(
            ui.section("Official outlook", "IMF World Economic Outlook, published projections to 2030"),
            unsafe_allow_html=True,
        )
        cols = st.columns(len(weo), gap="small")
        for col, (key, series) in zip(cols, weo.items()):
            spec = specs_by_key[key]
            horizon_years = max(0, series.index[-1].year - datetime.now().year)
            target = series.iloc[-1]
            current = series[series.index <= pd.Timestamp.now().normalize()]
            now_val = current.iloc[-1] if len(current) else target
            change = target - now_val
            with col:
                st.markdown(
                    ui.kpi_tile(
                        _tile_label(spec),
                        ui.fmt(target, spec.decimals, spec.unit),
                        "% GDP" if spec.unit == "% of GDP" else ("" if spec.unit == "index" else spec.unit),
                        ui.delta_badge(change, f"{change:+.1f}pp in {horizon_years}y"),
                        f"{series.index[-1].year} projection",
                    ),
                    unsafe_allow_html=True,
                )


def _overview_narrative(data: dict[str, pd.Series]) -> str:
    """Compose a short, data-derived read of the economy."""
    points: list[str] = []

    if "gdp_growth" in data:
        s = data["gdp_growth"]
        latest = s.iloc[-1]
        prior = s.iloc[-2] if len(s) > 1 else latest
        direction = "faster" if latest > prior else "slower"
        points.append(
            f"The economy grew <b>{latest:.2f}%</b> in {s.index[-1].year}, "
            f"{direction} than the {prior:.2f}% recorded a year earlier."
        )

    if "cpi_yoy" in data:
        s = data["cpi_yoy"].dropna()
        latest = s.iloc[-1]
        year_ago = s[s.index <= s.index[-1] - pd.DateOffset(years=1)]
        if len(year_ago):
            delta = latest - year_ago.iloc[-1]
            verb = "eased" if delta < 0 else "firmed"
            points.append(
                f"Consumer inflation {verb} to <b>{latest:.2f}%</b>, "
                f"{abs(delta):.2f}pp {'below' if delta < 0 else 'above'} the level a year ago."
            )
        else:
            points.append(f"Consumer inflation stands at <b>{latest:.2f}%</b>.")

    if "policy_rate" in data and "cpi_yoy" in data:
        real_rate = data["policy_rate"].iloc[-1] - data["cpi_yoy"].iloc[-1]
        stance = "restrictive" if real_rate > 0 else "accommodative"
        points.append(
            f"With the repo rate at <b>{data['policy_rate'].iloc[-1]:.2f}%</b>, the real policy "
            f"rate is <b>{real_rate:+.2f}%</b> — a {stance} stance."
        )

    if "unemployment" in data:
        s = data["unemployment"]
        latest = s.iloc[-1]
        trend = latest - s.iloc[-2] if len(s) > 1 else 0
        word = "rose" if trend > 0 else "fell" if trend < 0 else "held"
        points.append(
            f"Unemployment {word} to <b>{latest:.1f}%</b> of the labour force."
        )

    if "usd_zar" in data:
        s = data["usd_zar"]
        prior = s[s.index <= s.index[-1] - pd.DateOffset(years=1)]
        if len(prior):
            move = (s.iloc[-1] / prior.iloc[-1] - 1) * 100
            verb = "weakened" if move > 0 else "strengthened"
            points.append(
                f"The rand has {verb} <b>{abs(move):.1f}%</b> over the past year "
                f"to R{s.iloc[-1]:.2f} per dollar."
            )

    if not points:
        return ""

    body = " ".join(points)
    return ui.insight("Read-through", body)


# ---------------------------------------------------------------------------
# View: Explore
# ---------------------------------------------------------------------------
def view_explore(data: dict[str, pd.Series], specs_by_key):
    st.markdown(
        ui.section("Explore", "pick any series, choose a window, examine the shape"),
        unsafe_allow_html=True,
    )

    available = [s for s in SERIES if s.key in data]
    default_index = next(
        (i for i, s in enumerate(available) if s.key == FORECAST_DEFAULT), 0
    )

    left, mid, right = st.columns([2.2, 1, 1], gap="small")
    with left:
        key = st.selectbox(
            "Series",
            options=[s.key for s in available],
            index=default_index,
            format_func=lambda k: f"{specs_by_key[k].label} · {specs_by_key[k].freq_label}",
            label_visibility="collapsed",
        )
    spec = specs_by_key[key]
    series = data[key].dropna()
    freq = infer_frequency(series.index)
    ppy = periods_per_year(freq)

    with mid:
        windows = {"5y": 5, "10y": 10, "25y": 25, "Max": 0}
        choice = st.selectbox(
            "Window", list(windows), index=2, label_visibility="collapsed"
        )
    years = windows[choice]
    view = series if years == 0 else series[series.index >= series.index[-1] - pd.DateOffset(years=years)]

    with right:
        view_type = st.selectbox(
            "Style", ["Line", "Bars", "Yearly"], label_visibility="collapsed"
        )

    # --- stat strip -------------------------------------------------------
    latest = series.iloc[-1]
    change_1 = yoy_change(series, ppy)
    full_change = series.iloc[-1] - series.iloc[0]
    cagr = None
    span_years = (series.index[-1] - series.index[0]).days / 365.25
    if span_years > 0.5:
        if series.iloc[0] > 0 and series.iloc[-1] > 0:
            cagr = ((series.iloc[-1] / series.iloc[0]) ** (1 / span_years) - 1) * 100
    vol = series.pct_change().std() * np.sqrt(ppy) * 100 if ppy > 1 else None

    stats = st.columns(5, gap="small")
    tiles = [
        ("Latest", ui.fmt(latest, spec.decimals, spec.unit), spec.unit if spec.unit != "index" else "", f"{ui.last_updated(series.index[-1], freq)}"),
        ("Change (1y)", ui.fmt_pct(change_1), "", "vs same period last year"),
        ("Change (full)", ui.fmt_pct(full_change), "", f"since {series.index[0].year}"),
        (
            "CAGR" if cagr is not None else "Range",
            f"{cagr:.1f}%" if cagr is not None else ui.fmt(series.max() - series.min(), spec.decimals),
            "",
            "compound annual growth" if cagr is not None else "max − min",
        ),
        ("Volatility", f"{vol:.1f}%" if vol else "—", "", "annualised st. dev." if vol else "annual data"),
    ]
    for col, (label, value, unit, foot) in zip(stats, tiles):
        with col:
            st.markdown(
                ui.kpi_tile(label, value, unit, "", foot), unsafe_allow_html=True
            )

    st.markdown(
        ui.chips(
            [
                (f"<b>{spec.category}</b>", ""),
                (f"<b>{spec.freq_label}</b>", ""),
                (f"<b>{len(series):,}</b> observations", ""),
                (f"from <b>{series.index[0].strftime('%b %Y')}</b>", ""),
                (provider_of(spec), ""),
                (spec.description, ""),
            ]
        ),
        unsafe_allow_html=True,
    )

    # --- chart ------------------------------------------------------------
    st.markdown(ui.panel_open(spec.label, f"showing {choice}"), unsafe_allow_html=True)
    if view_type == "Bars":
        fig = ui.bar_chart(view, unit=spec.unit, freq=freq)
    elif view_type == "Yearly":
        annual = view.groupby(view.index.year).mean()
        fig = ui.bar_chart(
            pd.Series(annual.to_numpy(), index=pd.to_datetime([f"{y}-01-01" for y in annual.index])),
            unit=spec.unit,
            freq="A",
        )
    else:
        fig = ui.trend_line(
            view,
            color=ui.ACCENT,
            unit=spec.unit,
            freq=freq,
            height=380,
            show_last=True,
        )
    fig.update_layout(dragmode="pan")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": "hover"})
    st.markdown(ui.PANEL_CLOSE, unsafe_allow_html=True)

    # --- monthly / distribution detail ------------------------------------
    left, right = st.columns([1.6, 1], gap="medium")
    with left:
        st.markdown(ui.panel_open("Period-on-period change", "how fast the series is moving"), unsafe_allow_html=True)
        if ppy > 1:
            chg = view.pct_change() * 100
            st.plotly_chart(
                ui.bar_chart(chg.dropna(), unit="%", freq=freq, height=250),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            chg = view.diff()
            st.plotly_chart(
                ui.bar_chart(chg.dropna(), unit=spec.unit, freq=freq, height=250),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        st.markdown(ui.PANEL_CLOSE, unsafe_allow_html=True)

    with right:
        st.markdown(ui.panel_open("Distribution", "spread of observed values"), unsafe_allow_html=True)
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Box(
                y=view.to_numpy(),
                name="",
                marker_color=ui.SKY,
                fillcolor="rgba(79,179,255,0.16)",
                line={"color": ui.SKY, "width": 1.6},
                boxpoints="outliers",
            )
        )
        fig.update_layout(**ui.base_layout(250))
        fig.update_yaxes(showgrid=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(ui.PANEL_CLOSE, unsafe_allow_html=True)

    # --- export -----------------------------------------------------------
    with st.expander("Download this series"):
        frame = view.rename(spec.label).to_frame()
        st.dataframe(frame.tail(40), use_container_width=True)
        st.download_button(
            "Download CSV",
            frame.to_csv().encode("utf-8"),
            file_name=f"{key}.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# View: Forecast lab
# ---------------------------------------------------------------------------
def view_forecast(data: dict[str, pd.Series], specs_by_key):
    st.markdown(
        ui.section(
            "Forecast lab",
            "one-shot time-series models, picked by rolling-origin backtest",
        ),
        unsafe_allow_html=True,
    )

    forecastable = [s for s in SERIES if s.key in data and s.forecast]
    if not forecastable:
        st.warning("No forecastable series available.")
        return

    left, mid, right = st.columns([2, 1.2, 1.2], gap="small")
    with left:
        key = st.selectbox(
            "Series to forecast",
            options=[s.key for s in forecastable],
            index=next((i for i, s in enumerate(forecastable) if s.key == FORECAST_DEFAULT), 0),
            format_func=lambda k: f"{specs_by_key[k].label} · {specs_by_key[k].freq_label}",
            label_visibility="collapsed",
        )
    spec = specs_by_key[key]
    series = data[key].dropna()
    freq = infer_frequency(series.index)
    ppy = periods_per_year(freq)

    with mid:
        max_h = {12: 36, 4: 16, 1: 10}.get(ppy, 12)
        default_h = {12: 12, 4: 8, 1: 5}.get(ppy, 6)
        horizon = st.number_input(
            f"Horizon ({ui_horizon_word(freq)}s)",
            min_value=1,
            max_value=max_h,
            value=default_h,
            step=1,
        )
    with right:
        confidence = st.select_slider(
            "Interval", options=[0.5, 0.8, 0.9, 0.95], value=0.8,
            format_func=lambda v: f"{int(v*100)}%",
        )

    with st.expander("Advanced settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            window_years = st.select_slider(
                "History window",
                options=[5, 10, 15, 20, 25, 40, 0],
                value=25,
                format_func=lambda v: "All available" if v == 0 else f"{v} years",
            )
        with c2:
            model_options = model_names(freq, thorough=True, long_series=True)
            model_choice = st.selectbox("Model", model_options, index=0)

    window_obs = (window_years * ppy) if window_years else 0

    with st.spinner("Fitting models and backtesting…"):
        try:
            result = run_forecast(
                series,
                int(horizon),
                confidence=float(confidence),
                model=model_choice,
                max_history=window_obs if window_obs else None,
            )
        except ForecastError as exc:
            st.error(f"**Cannot forecast this series.** {exc}")
            st.caption(
                "This usually means the history is too short or too irregular. "
                "Try a longer window or a different series."
            )
            return
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the app
            st.error(f"Forecast failed unexpectedly: {exc}")
            return

    # --- result headline --------------------------------------------------
    mase = result.mase
    quality, quality_note = _quality(mase, result.skill_vs_naive)

    cols = st.columns(5, gap="small")
    tiles = [
        (
            "Forecast endpoint",
            ui.fmt(result.end_value, spec.decimals, spec.unit),
            spec.unit if spec.unit != "index" else "",
            f"{ui.period_label(result.forecast.index[-1], freq)}",
        ),
        (
            "Expected change",
            ui.fmt_pct(result.change_pct),
            "",
            f"{result.change:+.{spec.decimals}f} vs today",
        ),
        (
            f"{int(confidence*100)}% range",
            f"{ui.fmt(result.lower.iloc[-1], spec.decimals)} – {ui.fmt(result.upper.iloc[-1], spec.decimals)}",
            "",
            "at the horizon",
        ),
        ("Model", result.model.replace("ETS", "ETS").replace(" (", "\n("), "", result.family),
        ("Backtest quality", quality, "", quality_note),
    ]
    for col, (label, value, unit, foot) in zip(cols, tiles):
        with col:
            st.markdown(ui.kpi_tile(label, value, unit, "", foot), unsafe_allow_html=True)

    # --- narrative --------------------------------------------------------
    st.markdown(_forecast_narrative(result, spec), unsafe_allow_html=True)

    # --- chart with optional benchmark ------------------------------------
    benchmark = None
    benchmark_name = "IMF WEO"
    if key == "cpi_yoy" and "weo_inflation" in data:
        benchmark = data["weo_inflation"]
        benchmark_name = "IMF WEO inflation"
    elif key == "gdp_growth" and "weo_growth" in data:
        benchmark = data["weo_growth"]
        benchmark_name = "IMF WEO growth"
    elif key == "unemployment" and "weo_unemployment" in data:
        benchmark = data["weo_unemployment"]
        benchmark_name = "IMF WEO unemployment"

    st.markdown(
        ui.panel_open(
            f"{spec.label} — projection",
            f"{result.horizon_label} ahead · {result.model}",
        ),
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        ui.forecast_chart(
            result,
            unit=spec.unit,
            height=440,
            benchmark=benchmark,
            benchmark_name=benchmark_name,
        ),
        use_container_width=True,
        config={"displayModeBar": "hover"},
    )
    st.markdown(ui.PANEL_CLOSE, unsafe_allow_html=True)

    # --- table + leaderboard ----------------------------------------------
    left, right = st.columns([1.15, 1], gap="medium")

    with left:
        st.markdown(ui.panel_open("Projected values", "the numbers behind the line"), unsafe_allow_html=True)
        table = pd.DataFrame(
            {
                "Period": [ui.period_label(i, freq) for i in result.forecast.index],
                "Forecast": [round(v, spec.decimals) for v in result.forecast],
                f"Lower {int(confidence*100)}%": [round(v, spec.decimals) for v in result.lower],
                f"Upper {int(confidence*100)}%": [round(v, spec.decimals) for v in result.upper],
            }
        )
        st.dataframe(table, use_container_width=True, hide_index=True, height=300)
        st.download_button(
            "Download forecast CSV",
            table.to_csv(index=False).encode("utf-8"),
            file_name=f"{key}-forecast.csv",
            mime="text/csv",
        )
        st.markdown(ui.PANEL_CLOSE, unsafe_allow_html=True)

    with right:
        st.markdown(
            ui.panel_open("Model leaderboard", "rolling-origin backtest at this horizon"),
            unsafe_allow_html=True,
        )
        display = result.table.copy()
        numeric = ["MAE", "RMSE", "MAPE", "sMAPE", "MASE"]
        for c in numeric:
            if c in display:
                display[c] = display[c].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}")
        st.dataframe(display, use_container_width=True, height=300)
        st.caption(
            "Lower MASE is better. Below 1.0 beats a naive forecast; "
            "above 1.0 means the model adds nothing over simply repeating the last value."
        )
        st.markdown(ui.PANEL_CLOSE, unsafe_allow_html=True)

    if result.warnings:
        with st.expander("Notes from the modelling run"):
            for note in result.warnings:
                st.caption(f"• {note}")


def ui_horizon_word(freq: str) -> str:
    return {"M": "month", "Q": "quarter", "A": "year"}.get(freq, "period")


def _quality(mase: float | None, skill: float | None) -> tuple[str, str]:
    """Describe forecast quality relative to the naive baseline.

    Raw MASE alone is misleading: a volatile series can produce MASE above 1.0
    for every candidate model, including the one that wins. What matters is
    whether the selected model beat the naive benchmark.
    """
    if mase is None or (isinstance(mase, float) and np.isnan(mase)):
        return "n/a", "not enough data to backtest"

    if skill is not None:
        if skill >= 20:
            return "Strong", f"beats naive by {skill:.0f}% (MASE {mase:.2f})"
        if skill >= 5:
            return "Good", f"beats naive by {skill:.0f}% (MASE {mase:.2f})"
        if skill > -5:
            return "Neutral", f"about equal to naive (MASE {mase:.2f})"
        return "Weak", f"worse than naive by {abs(skill):.0f}% (MASE {mase:.2f})"

    if mase < 1.0:
        return "Good", f"MASE {mase:.2f} — beats naive"
    return "Weak", f"MASE {mase:.2f} — no better than naive"


def _forecast_narrative(result, spec) -> str:
    direction = "rise" if result.change > 0 else "fall" if result.change < 0 else "hold steady"
    end_period = ui.period_label(result.forecast.index[-1], result.frequency)
    unit_txt = "" if spec.unit == "index" else f" {spec.unit}"
    width = result.upper.iloc[-1] - result.lower.iloc[-1]

    good_or_bad = ""
    if spec.higher_is_better is True:
        good_or_bad = " a favourable direction" if result.change > 0 else " an unfavourable direction"
    elif spec.higher_is_better is False:
        good_or_bad = " a favourable direction" if result.change < 0 else " an unfavourable direction"

    body = (
        f"The selected model projects <b>{spec.label.lower()}</b> to {direction} from "
        f"<b>{ui.fmt(result.last_actual, spec.decimals)}{unit_txt}</b> to "
        f"<b>{ui.fmt(result.end_value, spec.decimals)}{unit_txt}</b> by {end_period}"
        f"{' —' + good_or_bad + ' for this indicator' if good_or_bad else ''}. "
        f"The {int(result.confidence*100)}% interval spans "
        f"<b>{ui.fmt(result.lower.iloc[-1], spec.decimals)}</b> to "
        f"<b>{ui.fmt(result.upper.iloc[-1], spec.decimals)}</b>, a range of "
        f"{ui.fmt(width, spec.decimals)}{unit_txt}."
    )

    mase = result.mase
    skill = result.skill_vs_naive
    if mase is not None and skill is not None:
        if skill >= 5:
            confidence_note = (
                f"Backtesting supports this: the model beat a naive forecast by "
                f"<b>{skill:.0f}%</b> across rolling test windows "
                f"(MASE {mase:.2f} against the baseline's {result.naive_mase:.2f})."
            )
        elif skill > -5:
            confidence_note = (
                f"This projection is roughly as accurate as simply repeating the last "
                f"observation (MASE {mase:.2f} vs {result.naive_mase:.2f}) — treat the "
                f"direction as indicative only."
            )
        else:
            confidence_note = (
                f"Treat it with caution: every candidate model tested — including this "
                f"one — fared worse than a naive forecast on this series "
                f"(MASE {mase:.2f} vs {result.naive_mase:.2f})."
            )
        body += " " + confidence_note
    elif mase is not None:
        body += f" Backtested MASE is <b>{mase:.2f}</b>."

    warn = skill is None or skill < 5
    return ui.insight("Forecast read-out", body, warn=warn)


# ---------------------------------------------------------------------------
# View: Compare
# ---------------------------------------------------------------------------
def view_compare(data: dict[str, pd.Series], specs_by_key):
    st.markdown(
        ui.section("Compare", "put series side by side, indexed to a common start"),
        unsafe_allow_html=True,
    )

    available = [s for s in SERIES if s.key in data]
    defaults = [k for k in ["cpi_yoy", "unemployment", "policy_rate"] if k in data]

    picked = st.multiselect(
        "Series",
        options=[s.key for s in available],
        default=defaults,
        format_func=lambda k: specs_by_key[k].label,
        max_selections=6,
    )
    if len(picked) < 1:
        st.info("Pick at least one series to plot.")
        return

    mode = st.radio(
        "Scale",
        ["Indexed (start = 100)", "Raw values", "Percent change from start"],
        horizontal=True,
    )

    frames = {}
    for key in picked:
        spec = specs_by_key[key]
        s = data[key].dropna()
        if mode == "Indexed (start = 100)":
            if s.iloc[0] == 0:
                continue
            frames[spec.label] = s / s.iloc[0] * 100
        elif mode == "Percent change from start":
            if s.iloc[0] == 0:
                continue
            frames[spec.label] = (s / s.iloc[0] - 1) * 100
        else:
            frames[spec.label] = s

    frame = series_frame(frames)
    if frame.empty:
        st.warning("Could not align these series.")
        return

    unit = "%" if mode != "Raw values" else ""
    st.plotly_chart(
        ui.multi_line(frame, unit=unit, height=430),
        use_container_width=True,
        config={"displayModeBar": "hover"},
    )

    # --- correlation ------------------------------------------------------
    st.markdown(
        ui.section("Relationships", "how these series move together over time"),
        unsafe_allow_html=True,
    )
    left, right = st.columns([1, 1.4], gap="medium")

    with left:
        st.markdown(ui.panel_open("Correlation matrix", "Pearson r on overlapping periods"), unsafe_allow_html=True)
        if len(frames) >= 2:
            monthly = series_frame(frames).resample("MS").mean().interpolate(limit_direction="both")
            st.plotly_chart(
                ui.correlation_heatmap(monthly, height=380),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        st.markdown(ui.PANEL_CLOSE, unsafe_allow_html=True)

    with right:
        st.markdown(ui.panel_open("Scatter", "pick a pair to inspect the relationship"), unsafe_allow_html=True)
        if len(frames) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                xcol = st.selectbox("X axis", list(frame.columns), index=0, key="sx")
            with c2:
                ycol = st.selectbox(
                    "Y axis", list(frame.columns), index=min(1, len(frame.columns) - 1), key="sy"
                )
            monthly = frame.resample("MS").mean().interpolate(limit_direction="both").dropna()
            if xcol != ycol and len(monthly) > 2:
                r = monthly[xcol].corr(monthly[ycol])
                st.plotly_chart(
                    ui.comparison_scatter(monthly, xcol, ycol, height=310),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
                st.caption(f"Pearson r = **{r:.2f}** across {len(monthly)} overlapping months.")
            else:
                st.caption("Choose two different series.")
        st.markdown(ui.PANEL_CLOSE, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# View: Custom dashboard
# ---------------------------------------------------------------------------
def _session_owner() -> str:
    """Identify the owner of saved dashboards.

    Uses a per-session token held in Streamlit's session state. This is a
    deliberate placeholder: it gives each browser its own namespace without
    requiring accounts. It is NOT authentication — anyone reaching the app gets
    a fresh namespace, and there is no protection between users. Replace with
    real auth before exposing saved dashboards beyond a trusted audience.
    """
    if "owner_id" not in st.session_state:
        import uuid

        st.session_state["owner_id"] = f"session-{uuid.uuid4().hex[:12]}"
    return st.session_state["owner_id"]


def view_custom(data: dict[str, pd.Series], specs_by_key):
    st.markdown(
        ui.section("My dashboard", "build your own view, then save it for later"),
        unsafe_allow_html=True,
    )

    available = [s for s in SERIES if s.key in data]
    owner = _session_owner()

    # --- saved layouts ----------------------------------------------------
    saved = store.list_dashboards(owner)
    if not saved.empty:
        cols = st.columns([3, 1])
        with cols[0]:
            choice = st.selectbox(
                "Load a saved dashboard",
                options=["— new / unsaved —"] + saved["name"].tolist(),
                label_visibility="collapsed",
            )
        with cols[1]:
            if choice != "— new / unsaved —" and st.button(
                "Delete", use_container_width=True
            ):
                store.delete_dashboard(owner, choice)
                st.rerun()

        if choice != "— new / unsaved —":
            loaded = store.get_dashboard(owner, choice)
            if loaded:
                st.session_state["custom_keys"] = loaded.get("keys", [])
                st.session_state["custom_layout"] = loaded.get("layout", 3)
                st.session_state["custom_style"] = loaded.get("style", "Area line")
                st.session_state["custom_window"] = loaded.get("window", 10)

    with st.expander("Configure your dashboard", expanded=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            picked = st.multiselect(
                "Series to include",
                options=[s.key for s in available],
                default=st.session_state.get(
                    "custom_keys", [k for k in HEADLINE if k in data][:4]
                ),
                format_func=lambda k: f"{specs_by_key[k].label} ({specs_by_key[k].category})",
                max_selections=12,
            )
            st.session_state["custom_keys"] = picked
        with c2:
            styles = ["Area line", "Bars", "Sparkline"]
            layouts = [2, 3, 4]
            layout = st.selectbox(
                "Columns",
                layouts,
                index=layouts.index(st.session_state.get("custom_layout", 3)),
            )
            chart_type = st.selectbox(
                "Chart style",
                styles,
                index=styles.index(st.session_state.get("custom_style", "Area line")),
            )

        window_choice = st.select_slider(
            "History window",
            options=[2, 5, 10, 25, 0],
            value=st.session_state.get("custom_window", 10),
            format_func=lambda v: "All" if v == 0 else f"{v}y",
        )

        # --- save ---------------------------------------------------------
        save_cols = st.columns([2, 1])
        with save_cols[0]:
            save_name = st.text_input(
                "Save as",
                placeholder="name this dashboard, e.g. Inflation watch",
                label_visibility="collapsed",
            )
        with save_cols[1]:
            if st.button(
                "Save dashboard", use_container_width=True, disabled=not save_name.strip()
            ):
                store.save_dashboard(
                    owner,
                    save_name.strip(),
                    {
                        "keys": picked,
                        "layout": layout,
                        "style": chart_type,
                        "window": window_choice,
                    },
                )
                st.success(f"Saved “{save_name.strip()}”")
                st.rerun()

    if not picked:
        st.info("Select one or more series above to build your dashboard.")
        return

    st.markdown(
        ui.chips([(f"<b>{len(picked)}</b> tiles", "ok"), (f"{layout} columns", ""), (chart_type, "")]),
        unsafe_allow_html=True,
    )

    # Lay tiles out in a grid.
    rows = [picked[i : i + layout] for i in range(0, len(picked), layout)]
    for row in rows:
        cols = st.columns(layout, gap="small")
        for col, key in zip(cols, row):
            spec = specs_by_key[key]
            series = data[key].dropna()
            freq = infer_frequency(series.index)
            ppy = periods_per_year(freq)
            view = (
                series
                if window_choice == 0
                else series[series.index >= series.index[-1] - pd.DateOffset(years=window_choice)]
            )
            change = yoy_change(series, ppy)
            with col:
                st.markdown(
                    ui.kpi_tile(
                        _tile_label(spec),
                        ui.fmt(series.iloc[-1], spec.decimals, spec.unit),
                        "" if spec.unit in ("index", "ZAR", "people") else spec.unit,
                        ui.delta_badge(change, ui.fmt_pct(change) + " y/y", invert=(spec.higher_is_better is False)),
                        ui.last_updated(series.index[-1], freq),
                        spark_svg=(
                            ui.svg_sparkline(view, color=ui.ACCENT, height=26)
                            if chart_type == "Sparkline"
                            else ""
                        ),
                    ),
                    unsafe_allow_html=True,
                )
                if chart_type == "Bars":
                    fig = ui.bar_chart(view, unit=spec.unit, freq=freq, height=160)
                elif chart_type == "Sparkline":
                    continue
                else:
                    fig = ui.trend_line(
                        view, color=ui.ACCENT, unit=spec.unit, freq=freq, height=170
                    )
                fig.update_layout(margin={"l": 4, "r": 4, "t": 4, "b": 4})
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # --- combined table ---------------------------------------------------
    st.markdown(ui.section("Your numbers, together"), unsafe_allow_html=True)
    summary_rows = []
    for key in picked:
        spec = specs_by_key[key]
        series = data[key].dropna()
        freq = infer_frequency(series.index)
        ppy = periods_per_year(freq)
        change = yoy_change(series, ppy)
        summary_rows.append(
            {
                "Indicator": spec.label,
                "Category": spec.category,
                "Latest": round(float(series.iloc[-1]), spec.decimals),
                "Unit": spec.unit,
                "1y change": "—" if change is None else f"{change:+.{spec.decimals}f}",
                "As of": ui.last_updated(series.index[-1], freq),
                "Frequency": spec.freq_label,
            }
        )
    summary = pd.DataFrame(summary_rows)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.download_button(
        "Download summary CSV",
        summary.to_csv(index=False).encode("utf-8"),
        file_name="sa-insights-dashboard.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# View: Downloads — always generated from live data
# ---------------------------------------------------------------------------
def view_downloads(summary: dict):
    st.markdown(
        ui.section("Downloads", "every artefact is built from the live data when you click"),
        unsafe_allow_html=True,
    )

    st.markdown(
        ui.insight(
            "Generated on demand",
            f"These files are rendered from the warehouse at request time — "
            f"<b>{summary['observations']:,}</b> observations and "
            f"<b>{summary['forecasts']}</b> forecast runs, as of "
            f"<b>{pd.Timestamp(summary['last_ingest']).strftime('%d %b %Y %H:%M') if summary.get('last_ingest') else 'never'}</b>. "
            f"Nothing here is a stale build artefact.",
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="medium")

    with left:
        st.markdown(
            ui.panel_open(
                "Power BI model",
                "star schema CSVs, TMSL model, and 17 DAX measures",
            ),
            unsafe_allow_html=True,
        )
        try:
            payload = downloads.powerbi_bundle()
            # `summary` is the warehouse-level dict, which has no per-export row
            # counts; read them from the download layer instead of assuming keys.
            stats = downloads.summary()
            st.caption(
                f"{len(payload):,} bytes · {stats['indicators']} indicators · "
                f"{stats['powerbi_rows']:,} fact rows · {stats['forecast_rows']:,} forecast rows"
            )
            st.download_button(
                "Download Power BI bundle",
                payload,
                file_name="sa-insights-powerbi.zip",
                mime="application/zip",
                use_container_width=True,
            )
        except Exception as exc:  # noqa: BLE001 - never break the page over a download
            st.error(f"Could not build the bundle: {exc}")
        st.markdown(ui.PANEL_CLOSE, unsafe_allow_html=True)

        st.markdown(
            ui.panel_open("Observations", "the raw measurements behind every chart"),
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download observations CSV",
            downloads.observations_csv(),
            file_name="sa-insights-observations.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown(ui.PANEL_CLOSE, unsafe_allow_html=True)

    with right:
        st.markdown(
            ui.panel_open(
                "Apache Ossie semantic model",
                "validated against the official JSON Schema",
            ),
            unsafe_allow_html=True,
        )
        try:
            st.download_button(
                "Ossie 0.1.1 · YAML",
                downloads.ossie_yaml(),
                file_name="sa_insights.ossie.yaml",
                mime="text/yaml",
                use_container_width=True,
            )
            st.download_button(
                "Ossie 0.1.1 · JSON",
                downloads.ossie_json(),
                file_name="sa_insights.ossie.json",
                mime="application/json",
                use_container_width=True,
            )
            st.download_button(
                "Ossie 0.2.0 draft · YAML",
                downloads.ossie_yaml(downloads.DRAFT_VERSION),
                file_name="sa_insights.ossie-draft.yaml",
                mime="text/yaml",
                use_container_width=True,
            )
            st.caption(
                "0.1.1 is the released shape; 0.2.0 draft is the flat form that "
                "the published schema validates."
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not build the semantic model: {exc}")
        st.markdown(ui.PANEL_CLOSE, unsafe_allow_html=True)

        st.markdown(
            ui.panel_open("Forecasts", "runs with model, MASE and prediction bands"),
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download forecasts CSV",
            downloads.forecasts_csv(),
            file_name="sa-insights-forecasts.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown(ui.PANEL_CLOSE, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
def _chat_agent():
    """One Agent per Streamlit session.

    Cached on the session rather than module level: the graph is built once per
    user, so the first question pays for it and the rest are instant, without a
    shared object leaking conversation state between browsers.
    """
    if "chat_agent" not in st.session_state:
        from sa_insights.chat.agent import Agent

        st.session_state.chat_agent = Agent()
    return st.session_state.chat_agent


def _chat_session():
    from sa_insights.chat.agent import Session

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = Session()
    return st.session_state.chat_session


def _render_citations(citations: list) -> None:
    """Render the evidence behind an answer.

    Citations are the reason to trust any of this, so they are shown inline by
    default rather than hidden behind an expander — a reader who has to click to
    check has not really been given a source.
    """
    if not citations:
        return
    order = {"observation": 0, "derived": 1, "forecast": 2}
    shown = sorted(citations, key=lambda c: order.get(c.source, 3))[:8]
    parts = []
    for cite in shown:
        value = "—" if cite.value is None else f"{cite.value:,.2f}".rstrip("0").rstrip(".")
        label = cite.label or cite.series_key
        cls = cite.source if cite.source in {"forecast", "derived"} else ""
        hint = f" · {cite.detail}" if cite.detail else ""
        parts.append(
            f'<span class="cite {cls}"><b>{value}</b> · {label} · {cite.period}{hint}</span>'
        )
    st.markdown(f'<div class="chat-meta">{"".join(parts)}</div>', unsafe_allow_html=True)


def _render_exchange(question: str, answer) -> None:
    kind_class = {"clarify": " clarify", "refuse": " refuse"}.get(answer.kind, "")
    st.markdown(
        f'<div class="chat-q"><b>You</b> · {question}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="chat-a{kind_class}">{answer.text}</div>',
        unsafe_allow_html=True,
    )

    if answer.kind == "answer" and answer.citations:
        _render_citations(answer.citations)

    if answer.follow_up:
        st.markdown(
            f'<div class="chat-meta">↳ {answer.follow_up}</div>',
            unsafe_allow_html=True,
        )

    # Surface the verification result rather than leaving it invisible. If a
    # draft was rejected for containing an unverifiable figure, the user should
    # know that is what happened — hiding it looks like the bot simply failed.
    if answer.unverified_numbers:
        st.markdown(
            ui.insight(
                "Draft rejected",
                "Some figures in the generated answer did not match any value in "
                "the warehouse, so the narrative was discarded. The verified "
                "numbers are in the citations above.",
                warn=True,
            ),
            unsafe_allow_html=True,
        )


def view_chat(summary: dict):
    st.markdown(
        ui.section(
            "Ask the data",
            "grounded answers with citations — or an honest refusal",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        ui.insight(
            "Numbers come from the warehouse, not the model",
            f"Every figure is fetched from <b>{summary['observations']:,}</b> stored "
            f"observations, and each answer is checked before display: if a number "
            f"cannot be traced to a stored value, the answer is rejected rather than "
            f"shown. Ask about any of the <b>{summary['series']}</b> series — "
            f"inflation, the repo rate, GDP growth, unemployment, the rand.",
        ),
        unsafe_allow_html=True,
    )

    session = _chat_session()
    agent = _chat_agent()

    # Suggestions are real, working questions rather than decorative prompt
    # chips — each one is exercised in the test suite.
    if not session.turns:
        st.markdown(
            '<div class="chat-empty"><b>Try one of these</b><br>'
            "What is CPI inflation now? · How has the repo rate changed since 2020? "
            "· Is inflation getting better? · What about the rand?</div>",
            unsafe_allow_html=True,
        )

    for turn in session.turns:
        _render_exchange(turn.question, turn.answer)

    question = st.chat_input("Ask about the South African economy…")
    if question:
        with st.spinner("Checking the data…"):
            try:
                # The return value is redundant here: agent.ask appends the turn
                # to the session itself, and the rerun below renders it from
                # there. Storing it would be a variable nobody reads.
                agent.ask(question, session)
            except Exception as exc:  # noqa: BLE001 - a bad question must not break the page
                st.error(f"Could not answer that: {exc}")
                return
        # Rerun so the new exchange renders above the input box, keeping the
        # conversation in reading order with the newest at the bottom.
        st.rerun()

    if session.turns:
        left, right = st.columns([1, 5])
        with left:
            if st.button("Clear", use_container_width=True):
                st.session_state.pop("chat_session", None)
                st.rerun()
        with right:
            st.caption(
                f"{len(session.turns)} turn(s) in this conversation · "
                "follow-ups carry context, so “and unemployment?” resolves to "
                "the previous subject."
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # First run on a fresh volume: populate before rendering anything, so the
    # page never opens empty.
    if not store.is_initialised():
        st.markdown(ui.brand_bar("bootstrap", "loading", 0), unsafe_allow_html=True)
        with st.spinner("First run — collecting data and fitting models. This takes a minute…"):
            try:
                report = refresh.refresh_all(force=True, do_forecast=True)
                st.success(
                    f"Loaded {report.series_loaded} series and "
                    f"{report.forecasts_written} forecasts."
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Initial data load failed: {exc}")
                st.stop()
        st.rerun()

    token = data_token()
    data = load_warehouse(token)
    summary = warehouse_summary(token)
    meta = warehouse_metadata(token)

    specs_by_key = {s.key: s for s in SERIES}

    total_obs = summary["observations"]
    as_of = (
        pd.Timestamp(summary["last_ingest"]).strftime("%d %b %Y, %H:%M")
        if summary.get("last_ingest")
        else "unknown"
    )

    st.markdown(
        ui.brand_bar(f"{summary['series']}/{len(SERIES)} series", as_of, total_obs),
        unsafe_allow_html=True,
    )

    sidebar(summary, meta, set(data.keys()))

    tabs = st.tabs(
        [
            "Overview",
            "Explore",
            "Forecast lab",
            "Compare",
            "Ask",
            "My dashboard",
            "Downloads",
        ]
    )
    with tabs[0]:
        view_overview(data, specs_by_key)
    with tabs[1]:
        view_explore(data, specs_by_key)
    with tabs[2]:
        view_forecast(data, specs_by_key)
    with tabs[3]:
        view_compare(data, specs_by_key)
    with tabs[4]:
        view_chat(summary)
    with tabs[5]:
        view_custom(data, specs_by_key)
    with tabs[6]:
        view_downloads(summary)

    sources = sorted({provider_of(s) for s in SERIES})
    st.markdown(
        f'<p style="font-size:.7rem;color:#5C7089;text-align:center;margin-top:2.5rem;'
        f'padding-top:1rem;border-top:1px solid rgba(148,163,184,.16)">'
        f'SA Insights · {summary["series"]} series · {total_obs:,} observations · '
        f'{summary["forecasts"]} forecast runs · '
        f'sources: {" · ".join(sources)} · '
        f'forecasts are statistical extrapolations, not predictions of policy or events.'
        f"</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
