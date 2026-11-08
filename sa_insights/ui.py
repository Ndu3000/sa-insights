"""Visual design system: palette, CSS and Plotly chart builders.

The look is a clean editorial dashboard — deep slate canvas, soft cards, a
restrained accent set, and typography that lets the numbers lead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
INK = "#0B1220"
SURFACE = "#111C2E"
SURFACE_2 = "#16243A"
BORDER = "rgba(148,163,184,0.16)"
TEXT = "#E8EEF7"
MUTED = "#8FA3BF"
FAINT = "#5C7089"

ACCENT = "#3DDC97"       # primary green
ACCENT_DIM = "#17A673"
AMBER = "#F5B94B"
CORAL = "#FF6B6B"
SKY = "#4FB3FF"
VIOLET = "#9B8CFF"

SERIES_COLORS = [
    "#3DDC97", "#4FB3FF", "#F5B94B", "#FF6B6B",
    "#9B8CFF", "#43D9C4", "#FFA06B", "#7DD87D",
]

# South African flag cues, used sparingly for the brand mark only.
FLAG = ["#007A4D", "#FFB612", "#DE3831", "#002395", "#FFFFFF", "#000000"]


def palette(n: int) -> list[str]:
    return [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(n)]


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------
def fmt(value: float | None, spec_decimals: int = 1, unit: str = "") -> str:
    """Human, compact number formatting for KPI tiles and tooltips."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    u = unit.lower()
    if "people" in u or abs(value) >= 1e9:
        if abs(value) >= 1e9:
            return f"{value / 1e9:,.2f}bn"
        if abs(value) >= 1e6:
            return f"{value / 1e6:,.1f}m"
        if abs(value) >= 1e4:
            return f"{value / 1e3:,.0f}k"
        return f"{value:,.0f}"
    if abs(value) >= 1e6 and "us$ m" not in u:
        return f"{value / 1e6:,.2f}m"
    if abs(value) >= 1e4 and "us$ m" in u:
        return f"{value:,.0f}"
    if spec_decimals == 0:
        return f"{value:,.0f}"
    return f"{value:,.{spec_decimals}f}"


def fmt_pct(value: float | None, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:+.{decimals}f}%"


def period_label(ts: pd.Timestamp, freq: str) -> str:
    if freq == "A":
        return ts.strftime("%Y")
    if freq == "Q":
        return f"{ts.year} Q{(ts.month - 1) // 3 + 1}"
    return ts.strftime("%b %Y")


def last_updated(ts: pd.Timestamp | None, freq: str) -> str:
    if ts is None:
        return "unknown"
    return period_label(pd.Timestamp(ts), freq)


# ---------------------------------------------------------------------------
# Plotly base layout
# ---------------------------------------------------------------------------
def base_layout(height: int = 340, *, show_legend: bool = False, title: str | None = None) -> dict:
    layout = {
        "height": height,
        "margin": {"l": 8, "r": 12, "t": 8 if not title else 34, "b": 8},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, -apple-system, Segoe UI, sans-serif", "size": 12, "color": MUTED},
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": SURFACE_2,
            "bordercolor": BORDER,
            "font": {"color": TEXT, "size": 12},
        },
        "showlegend": show_legend,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": MUTED, "size": 11},
        },
        "xaxis": {
            "showgrid": False,
            "zeroline": False,
            "linecolor": BORDER,
            "tickfont": {"color": FAINT, "size": 11},
            "showspikes": False,
        },
        "yaxis": {
            "gridcolor": BORDER,
            "zeroline": False,
            "linecolor": "rgba(0,0,0,0)",
            "tickfont": {"color": FAINT, "size": 11},
            "ticksuffix": "",
        },
        "dragmode": "pan",
    }
    if title:
        layout["title"] = {
            "text": title,
            "font": {"size": 13, "color": TEXT},
            "x": 0,
            "xanchor": "left",
        }
    return layout


def svg_sparkline(
    series: pd.Series,
    color: str = ACCENT,
    *,
    width: int = 160,
    height: int = 32,
    stroke: float = 1.5,
    points: int = 60,
) -> str:
    """Inline SVG sparkline that lives *inside* a KPI card.

    Rendering this as SVG rather than a chart keeps the tile self-contained:
    it can never overflow its card, and it costs nothing to draw.
    """
    if series is None:
        return ""
    vals = pd.Series(series).dropna().tail(points).to_numpy(dtype=float)
    if len(vals) < 2:
        return ""
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    span = (hi - lo) or 1.0
    pad = stroke
    n = len(vals)
    xs = np.linspace(0.0, width, n)
    ys = height - pad - (vals - lo) / span * (height - 2 * pad)
    line = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
    area = (
        f"M{xs[0]:.2f},{height:.2f} L"
        + " L".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
        + f" L{xs[-1]:.2f},{height:.2f} Z"
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'preserveAspectRatio="none" style="display:block;margin-top:.5rem">'
        f'<path d="{area}" fill="{color}" opacity="0.11"/>'
        f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'vector-effect="non-scaling-stroke"/>'
        f"</svg>"
    )


def sparkline(series: pd.Series, color: str = ACCENT, height: int = 46) -> go.Figure:
    """Plotly sparkline, for places that need hover detail."""
    fig = go.Figure()
    if series is None or series.empty:
        fig.add_annotation(text="no data", showarrow=False, font={"color": FAINT, "size": 10})
    else:
        y = series.tail(60)
        fig.add_trace(
            go.Scatter(
                x=y.index,
                y=y.values,
                mode="lines",
                line={"color": color, "width": 1.7},
                hovertemplate="%{x|%b %Y}<br>%{y:,.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis={"visible": False, "fixedrange": True},
        yaxis={"visible": False, "fixedrange": True},
        hoverlabel={"bgcolor": SURFACE_2, "bordercolor": BORDER, "font": {"color": TEXT}},
    )
    return fig


def _alpha(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def trend_line(
    series: pd.Series,
    *,
    color: str = ACCENT,
    unit: str = "",
    decimals: int = 1,
    freq: str = "M",
    height: int = 360,
    show_last: bool = True,
    name: str = "Actual",
) -> go.Figure:
    """Single-series area line, used throughout the explore tabs."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.to_numpy(),
            mode="lines",
            name=name,
            line={"color": color, "width": 2.4, "shape": "spline", "smoothing": 0.4},
            fill="tozeroy",
            fillcolor=_alpha(color, 0.10),
            hovertemplate="%{x|" + ("%Y" if freq == "A" else "%b %Y") + "}<br>%{y:,.2f} " + unit + "<extra></extra>",
        )
    )
    if show_last and len(series):
        fig.add_trace(
            go.Scatter(
                x=[series.index[-1]],
                y=[series.iloc[-1]],
                mode="markers",
                marker={"size": 8, "color": color, "line": {"width": 2, "color": SURFACE}},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.update_layout(**base_layout(height))
    fig.update_yaxes(ticksuffix="" if unit == "%" else "")
    return fig


def multi_line(
    frame: pd.DataFrame,
    *,
    unit: str = "",
    height: int = 380,
    freq: str = "M",
) -> go.Figure:
    fig = go.Figure()
    for i, column in enumerate(frame.columns):
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column].to_numpy(),
                mode="lines",
                name=str(column),
                line={"color": palette(len(frame.columns))[i], "width": 2.1},
                hovertemplate="%{y:,.2f} " + unit + "<extra>" + str(column) + "</extra>",
            )
        )
    fig.update_layout(**base_layout(height, show_legend=len(frame.columns) > 1))
    return fig


def bar_chart(
    series: pd.Series,
    *,
    unit: str = "",
    height: int = 380,
    diverging: bool = True,
    freq: str = "A",
) -> go.Figure:
    colors = []
    for v in series.to_numpy():
        if not diverging:
            colors.append(ACCENT)
        else:
            colors.append(ACCENT if v >= 0 else CORAL)
    fig = go.Figure(
        go.Bar(
            x=series.index,
            y=series.to_numpy(),
            marker={"color": colors, "line": {"width": 0}},
            hovertemplate="%{x|%Y}<br>%{y:,.2f} " + unit + "<extra></extra>",
        )
    )
    fig.update_layout(
        **base_layout(height),
        bargap=0.35,
    )
    if diverging:
        fig.update_yaxes(zeroline=True, zerolinecolor=BORDER, zerolinewidth=1)
    return fig


def forecast_chart(
    result,
    *,
    unit: str = "",
    height: int = 430,
    show_history_points: bool = False,
    benchmark: pd.Series | None = None,
    benchmark_name: str = "IMF WEO",
) -> go.Figure:
    """History + projection with a graded confidence band."""
    fig = go.Figure()

    hist = result.history
    freq = result.frequency
    date_fmt = "%Y" if freq == "A" else "%b %Y"

    # Confidence band (upper then lower, filled between).
    band_x = list(result.forecast.index)
    fig.add_trace(
        go.Scatter(
            x=band_x + band_x[::-1],
            y=list(result.upper) + list(result.lower)[::-1],
            fill="toself",
            fillcolor=_alpha(AMBER, 0.14),
            line={"width": 0},
            name=f"{int(result.confidence * 100)}% interval",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist.to_numpy(),
            mode="lines" + ("+markers" if show_history_points and freq != "M" else ""),
            name="Actual",
            line={"color": SKY, "width": 2.3},
            marker={"size": 3.5, "color": SKY},
            hovertemplate="%{x|" + date_fmt + "}<br><b>%{y:,.2f}</b> " + unit + "<extra>Actual</extra>",
        )
    )

    # Bridge the last actual point into the forecast so the line is continuous.
    bridge_x = [hist.index[-1]] + band_x
    bridge_y = [hist.iloc[-1]] + list(result.forecast)
    fig.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_y,
            mode="lines",
            name=f"Forecast · {result.model}",
            line={"color": AMBER, "width": 2.4, "dash": "dot"},
            hovertemplate="%{x|" + date_fmt + "}<br><b>%{y:,.2f}</b> " + unit + "<extra>Forecast</extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=band_x,
            y=list(result.forecast),
            mode="markers",
            marker={"size": 5, "color": AMBER, "symbol": "circle"},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Optional published benchmark (e.g. IMF WEO), aligned on the forecast dates.
    if benchmark is not None and not benchmark.empty:
        aligned = benchmark.reindex(band_x)
        if aligned.notna().any():
            bx = [p for p, v in aligned.items() if pd.notna(v)]
            by = [v for v in aligned if pd.notna(v)]
            fig.add_trace(
                go.Scatter(
                    x=bx,
                    y=by,
                    mode="lines+markers",
                    name=benchmark_name,
                    line={"color": VIOLET, "width": 2.0, "dash": "dashdot"},
                    marker={"size": 5},
                    hovertemplate="%{x|" + date_fmt + "}<br>%{y:,.2f} " + unit + "<extra>" + benchmark_name + "</extra>",
                )
            )

    # Marker where history ends.
    fig.add_vline(
        x=hist.index[-1],
        line={"color": BORDER, "width": 1, "dash": "dash"},
    )
    fig.add_annotation(
        x=hist.index[-1],
        y=1.0,
        yref="paper",
        text="forecast →",
        showarrow=False,
        font={"color": FAINT, "size": 10},
        xanchor="left",
        yanchor="bottom",
    )

    fig.update_layout(**base_layout(height, show_legend=True))
    fig.update_xaxes(range=[hist.index[max(0, len(hist) - 120)], band_x[-1]])
    return fig


def gauge(value: float, *, title: str, low: float, high: float, unit: str = "%", height: int = 170) -> go.Figure:
    """Compact radial gauge for a single bounded indicator."""
    if high <= low:
        high = low + 1
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={"suffix": unit, "font": {"size": 26, "color": TEXT}},
            gauge={
                "axis": {"range": [low, high], "tickcolor": FAINT, "tickfont": {"size": 9, "color": FAINT}},
                "bar": {"color": ACCENT, "thickness": 0.62},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [low, low + (high - low) * 0.5], "color": _alpha(SKY, 0.10)},
                    {"range": [low + (high - low) * 0.5, high], "color": _alpha(AMBER, 0.10)},
                ],
            },
            title={"text": title, "font": {"size": 11, "color": MUTED}},
        )
    )
    fig.update_layout(
        height=height,
        margin={"l": 12, "r": 12, "t": 26, "b": 4},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"},
    )
    return fig


def comparison_scatter(
    frame: pd.DataFrame, x: str, y: str, *, size_by: str | None = None, height: int = 420
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame[x],
            y=frame[y],
            mode="markers",
            marker={
                "size": 9,
                "color": SKY,
                "opacity": 0.75,
                "line": {"width": 1, "color": SURFACE},
            },
            hovertemplate=f"{x}: %{{x:,.2f}}<br>{y}: %{{y:,.2f}}<extra></extra>",
        )
    )
    if len(frame) > 2:
        # Simple least-squares fit line for context.
        import numpy as np

        xs = frame[x].to_numpy(dtype=float)
        ys = frame[y].to_numpy(dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.sum() > 2:
            slope, intercept = np.polyfit(xs[mask], ys[mask], 1)
            grid = np.linspace(xs[mask].min(), xs[mask].max(), 50)
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=slope * grid + intercept,
                    mode="lines",
                    line={"color": AMBER, "width": 1.5, "dash": "dash"},
                    name="trend",
                    hoverinfo="skip",
                )
            )
    fig.update_layout(**base_layout(height, show_legend=False))
    fig.update_xaxes(title={"text": x, "font": {"size": 11, "color": FAINT}})
    fig.update_yaxes(title={"text": y, "font": {"size": 11, "color": FAINT}})
    return fig


def correlation_heatmap(frame: pd.DataFrame, height: int = 460) -> go.Figure:
    corr = frame.corr(numeric_only=True)
    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(),
            x=list(corr.columns),
            y=list(corr.index),
            colorscale=[
                [0.0, CORAL],
                [0.5, SURFACE_2],
                [1.0, ACCENT],
            ],
            zmin=-1,
            zmax=1,
            hovertemplate="%{y} ↔ %{x}<br>r = %{z:.2f}<extra></extra>",
            colorbar={
                "thickness": 8,
                "outlinewidth": 0,
                "tickfont": {"size": 10, "color": FAINT},
            },
        )
    )
    fig.update_layout(**base_layout(height, show_legend=False))
    fig.update_xaxes(tickangle=-40, tickfont={"size": 10})
    fig.update_yaxes(tickfont={"size": 10})
    return fig


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --ink:#0B1220; --surface:#111C2E; --surface2:#16243A;
  --border:rgba(148,163,184,.16); --text:#E8EEF7; --muted:#8FA3BF; --faint:#5C7089;
  --accent:#3DDC97; --amber:#F5B94B; --coral:#FF6B6B; --sky:#4FB3FF;
}

html, body, [class*="css"], .stApp {
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  -webkit-font-smoothing:antialiased;
}
.stApp { background:var(--ink); }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding:1.1rem 2.2rem 3rem; max-width:1500px; }

/* ---------- Brand bar ---------- */
.brand {
  display:flex; align-items:center; gap:.85rem;
  padding:.2rem 0 1.15rem; border-bottom:1px solid var(--border); margin-bottom:1.4rem;
}
.brand-mark {
  width:38px; height:38px; border-radius:11px; flex:0 0 auto;
  background:linear-gradient(140deg,#007A4D 0%,#3DDC97 45%,#4FB3FF 100%);
  display:grid; place-items:center;
  box-shadow:0 6px 22px rgba(61,220,151,.25);
  font-weight:800; font-size:.95rem; color:#04180F; letter-spacing:-.02em;
}
.brand-text h1 {
  font-size:1.06rem; font-weight:700; color:var(--text);
  margin:0; letter-spacing:-.015em; line-height:1.2;
}
.brand-text p { font-size:.765rem; color:var(--faint); margin:.14rem 0 0; }
.brand-spacer { flex:1; }
.brand-meta {
  font-size:.72rem; color:var(--faint); text-align:right; line-height:1.5;
}
.brand-meta b { color:var(--muted); font-weight:600; }

/* ---------- KPI tiles ---------- */
.kpi {
  background:linear-gradient(165deg,var(--surface2) 0%,var(--surface) 78%);
  border:1px solid var(--border); border-radius:15px;
  padding:.95rem 1.05rem .55rem; height:100%;
  transition:border-color .18s ease, transform .18s ease;
}
.kpi:hover { border-color:rgba(61,220,151,.34); transform:translateY(-1px); }
.kpi-label {
  font-size:.655rem; font-weight:600; letter-spacing:.045em;
  text-transform:uppercase; color:var(--faint); margin-bottom:.32rem;
  line-height:1.28; min-height:1.72em; max-height:1.72em;
  overflow:hidden; overflow-wrap:normal; word-break:normal;
}
.kpi-value {
  font-size:1.52rem; font-weight:750; color:var(--text);
  letter-spacing:-.03em; line-height:1.08; font-variant-numeric:tabular-nums;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.kpi-unit { font-size:.66rem; font-weight:500; color:var(--faint); margin-left:.16rem; }
.kpi-delta {
  display:inline-block; max-width:100%;
  font-size:.715rem; font-weight:600; margin-top:.3rem;
  padding:.12rem .42rem; border-radius:6px;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.kpi-delta.up   { color:#3DDC97; background:rgba(61,220,151,.12); }
.kpi-delta.down { color:#FF6B6B; background:rgba(255,107,107,.12); }
.kpi-delta.flat { color:#8FA3BF; background:rgba(143,163,191,.12); }
.kpi-foot {
  font-size:.655rem; color:var(--faint); margin-top:.34rem;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}

/* ---------- Section headings ---------- */
.section-head {
  display:flex; align-items:baseline; gap:.6rem;
  margin:1.75rem 0 .85rem; padding-bottom:.42rem;
  border-bottom:1px solid var(--border);
}
.section-head h2 {
  font-size:.95rem; font-weight:650; color:var(--text); margin:0;
  letter-spacing:-.01em;
}
.section-head span { font-size:.75rem; color:var(--faint); }

/* ---------- Panels ---------- */
.panel {
  background:var(--surface); border:1px solid var(--border);
  border-radius:14px; padding:1rem 1.1rem .7rem; margin-bottom:.9rem;
}
.panel-title {
  font-size:.8rem; font-weight:650; color:var(--text); margin:0 0 .1rem;
}
.panel-sub { font-size:.71rem; color:var(--faint); margin:0 0 .55rem; }

/* ---------- Insight callout ---------- */
.insight {
  background:linear-gradient(135deg,rgba(61,220,151,.09),rgba(79,179,255,.05));
  border:1px solid rgba(61,220,151,.26);
  border-left:3px solid var(--accent);
  border-radius:11px; padding:.8rem 1rem; margin:.55rem 0 .9rem;
}
.insight-title {
  font-size:.7rem; font-weight:700; letter-spacing:.08em;
  text-transform:uppercase; color:var(--accent); margin-bottom:.28rem;
}
.insight-body { font-size:.845rem; color:var(--text); line-height:1.55; }
.insight-body b { color:#3DDC97; font-weight:650; }

.insight.warn {
  background:linear-gradient(135deg,rgba(245,185,75,.10),rgba(255,107,107,.05));
  border-color:rgba(245,185,75,.30); border-left-color:var(--amber);
}
.insight.warn .insight-title { color:var(--amber); }
.insight.warn .insight-body b { color:var(--amber); }

/* ---------- Data provenance chips ---------- */
.chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.3rem 0 .8rem; }
.chip {
  font-size:.685rem; font-weight:550; color:var(--muted);
  background:rgba(148,163,184,.09); border:1px solid var(--border);
  border-radius:999px; padding:.2rem .62rem; white-space:nowrap;
}
.chip b { color:var(--text); font-weight:650; }
.chip.ok    { color:#3DDC97; border-color:rgba(61,220,151,.30); background:rgba(61,220,151,.09); }
.chip.stale { color:#F5B94B; border-color:rgba(245,185,75,.30); background:rgba(245,185,75,.09); }
.chip.fail  { color:#FF6B6B; border-color:rgba(255,107,107,.30); background:rgba(255,107,107,.09); }

/* ---------- Streamlit widget polish ---------- */
.stTabs [data-baseweb="tab-list"] {
  gap:.15rem; border-bottom:1px solid var(--border); background:transparent;
}
.stTabs [data-baseweb="tab"] {
  height:auto; padding:.62rem .95rem; background:transparent;
  font-size:.815rem; font-weight:550; color:var(--faint);
  border-radius:9px 9px 0 0;
}
.stTabs [data-baseweb="tab"]:hover { color:var(--muted); background:rgba(148,163,184,.05); }
.stTabs [aria-selected="true"] {
  color:var(--accent) !important; background:rgba(61,220,151,.07) !important;
  box-shadow:inset 0 -2px 0 var(--accent) !important;
}
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] div,
.stTabs [aria-selected="true"] span { color:var(--accent) !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none !important; }

div[data-testid="stSidebar"] {
  background:#0D1626; border-right:1px solid var(--border);
}
div[data-testid="stSidebar"] .block-container { padding-top:1.2rem; }

div[data-testid="stMetricValue"] {
  font-variant-numeric:tabular-nums; color:var(--text);
}
.stButton>button {
  background:rgba(61,220,151,.10); color:#3DDC97;
  border:1px solid rgba(61,220,151,.30); border-radius:9px;
  font-size:.8rem; font-weight:600; padding:.42rem 1rem;
  transition:all .16s ease;
}
.stButton>button:hover {
  background:rgba(61,220,151,.18); border-color:rgba(61,220,151,.5); color:#3DDC97;
}
div[data-baseweb="select"]>div, div[data-baseweb="input"]>div {
  background:var(--surface2); border-color:var(--border); border-radius:9px;
}
.stDataFrame { border:1px solid var(--border); border-radius:11px; overflow:hidden; }
.stAlert { border-radius:11px; font-size:.82rem; }
hr { border-color:var(--border); }
.stSlider [data-baseweb="slider"] { padding-top:.2rem; }
[data-testid="stExpander"] {
  background:var(--surface); border:1px solid var(--border); border-radius:11px;
}
.stDownloadButton>button {
  background:rgba(79,179,255,.10); color:#4FB3FF;
  border:1px solid rgba(79,179,255,.30); border-radius:9px;
  font-size:.78rem; font-weight:600;
}
.stDownloadButton>button:hover { background:rgba(79,179,255,.18); color:#4FB3FF; }

/* ---------- Chat ---------- */
.chat-q {
  background:rgba(79,179,255,.10); border:1px solid rgba(79,179,255,.24);
  border-radius:11px; padding:.62rem .9rem; margin:.2rem 0 .6rem;
  font-size:.86rem; color:var(--text); line-height:1.5;
}
.chat-q b { color:#4FB3FF; font-weight:650; }
.chat-a {
  background:var(--surface); border:1px solid var(--border);
  border-left:3px solid var(--accent);
  border-radius:11px; padding:.7rem .95rem; margin:0 0 .5rem;
  font-size:.87rem; color:var(--text); line-height:1.6;
}
.chat-a.clarify { border-left-color:var(--amber); }
.chat-a.refuse  { border-left-color:var(--faint); color:var(--muted); }
.chat-a b { color:#3DDC97; font-weight:650; }
.chat-meta {
  font-size:.68rem; color:var(--faint); margin:.15rem 0 .85rem;
  display:flex; flex-wrap:wrap; gap:.4rem; align-items:center;
}
.cite {
  font-size:.685rem; color:var(--muted);
  background:rgba(148,163,184,.08); border:1px solid var(--border);
  border-radius:7px; padding:.16rem .5rem; white-space:nowrap;
  font-variant-numeric:tabular-nums;
}
.cite b { color:var(--text); font-weight:650; }
.cite.forecast { color:#4FB3FF; border-color:rgba(79,179,255,.28); background:rgba(79,179,255,.08); }
.cite.derived  { color:#F5B94B; border-color:rgba(245,185,75,.28); background:rgba(245,185,75,.08); }
.chat-empty {
  border:1px dashed var(--border); border-radius:12px;
  padding:1.4rem 1.1rem; text-align:center; color:var(--faint);
  font-size:.82rem; line-height:1.6;
}
.chat-empty b { color:var(--muted); font-weight:600; }
</style>
"""


def brand_bar(sources: str, as_of: str, observations: int) -> str:
    return f"""
<div class="brand">
  <div class="brand-mark">SA</div>
  <div class="brand-text">
    <h1>SA Insights</h1>
    <p>South African statistics · live data · forecast lab</p>
  </div>
  <div class="brand-spacer"></div>
  <div class="brand-meta">
    <b>{observations:,}</b> observations<br>
    {as_of} · {sources}
  </div>
</div>
"""


def kpi_tile(
    label_html: str,
    value: str,
    unit: str,
    delta_html: str,
    foot: str,
    spark_svg: str = "",
) -> str:
    unit_html = f'<span class="kpi-unit">{unit}</span>' if unit else ""
    return f"""
<div class="kpi">
  <div class="kpi-label">{label_html}</div>
  <div class="kpi-value">{value}{unit_html}</div>
  {delta_html}
  <div class="kpi-foot">{foot}</div>
  {spark_svg}
</div>
"""


def delta_badge(change: float | None, text: str, *, invert: bool = False) -> str:
    """Colour a change badge by whether the movement is good or bad."""
    if change is None or (isinstance(change, float) and pd.isna(change)):
        return '<span class="kpi-delta flat">—</span>'
    if abs(change) < 1e-9:
        return f'<span class="kpi-delta flat">■ {text}</span>'
    rising = change > 0
    good = (not rising) if invert else rising
    cls = "up" if good else "down"
    arrow = "▲" if rising else "▼"
    return f'<span class="kpi-delta {cls}">{arrow} {text}</span>'


def insight(title: str, body: str, *, warn: bool = False) -> str:
    cls = "insight warn" if warn else "insight"
    return f"""
<div class="{cls}">
  <div class="insight-title">{title}</div>
  <div class="insight-body">{body}</div>
</div>
"""


def section(title: str, sub: str = "") -> str:
    sub_html = f"<span>{sub}</span>" if sub else ""
    return f'<div class="section-head"><h2>{title}</h2>{sub_html}</div>'


def chips(items: list[tuple[str, str]]) -> str:
    """items = [(text, css_class)] where css_class is '', 'ok', 'stale' or 'fail'."""
    inner = "".join(f'<span class="chip {cls}">{text}</span>' for text, cls in items)
    return f'<div class="chips">{inner}</div>'


def panel_open(title: str, sub: str = "") -> str:
    sub_html = f'<p class="panel-sub">{sub}</p>' if sub else ""
    return f'<div class="panel"><p class="panel-title">{title}</p>{sub_html}'


PANEL_CLOSE = "</div>"
