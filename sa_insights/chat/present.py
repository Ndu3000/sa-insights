"""Presentation logic shared by every chat front end.

The Streamlit tab and the React/CopilotKit UI must tell a reader the same thing
about an answer: which numbers back it, which of them are observations versus
forecasts or derived changes, and whether the answer actually passed
verification. If each front end worked that out for itself they would drift, and
the one place a reader needs consistency is the evidence trail.

This module is deliberately free of Streamlit, FastAPI and HTML. It turns an
``Answer`` into plain data; each front end only decides how to draw it.

Verification states
-------------------
The agent never ships an unverified figure, but "never ships a bad number" is not
the same as "tells the reader what happened". The states below name what actually
occurred so the interface does not have to imply more than the code guarantees.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any, Iterable, Literal

# Order in which evidence is shown inside one series: what was measured first,
# then what was derived from it, then what is merely predicted.
SOURCE_ORDER: tuple[str, ...] = ("observation", "derived", "forecast")

SOURCE_LABEL: dict[str, str] = {
    "observation": "Observed",
    "derived": "Derived",
    "forecast": "Forecast",
}

SOURCE_HELP: dict[str, str] = {
    "observation": "A value published by the source and stored in the warehouse.",
    "derived": "Computed from stored observations, for example a year-on-year change.",
    "forecast": "A model projection. Not a published figure.",
}

# Questions that stand on their own. Each is exercised in the test suite, so a
# chip can never lead to a refusal: a suggestion that fails is worse than none.
SUGGESTIONS: tuple[str, ...] = (
    "What is CPI inflation now?",
    "How has the repo rate changed since 2020?",
    "Is inflation getting better?",
    "What is the unemployment rate?",
)

# Offered after an answer. These lean on conversation context ("and ..."), which
# is the behaviour the follow-up resolver exists for, so they are tested with a
# preceding turn rather than on a cold session.
FOLLOW_UPS: tuple[str, ...] = (
    "What about the rand?",
    "And the repo rate?",
    "How has that changed since 2020?",
)


def suggestions(has_history: bool) -> list[str]:
    """Chips to show: standalone questions first, context-dependent ones after."""
    return list(FOLLOW_UPS if has_history else SUGGESTIONS)


# --------------------------------------------------------------------------- #
# Citations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CitationItem:
    source: str
    source_label: str
    period: str
    value: float | None
    value_text: str
    detail: str


@dataclass(frozen=True)
class CitationGroup:
    series_key: str
    label: str
    items: tuple[CitationItem, ...]


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from a Citation object or a plain dict (the JSON form)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def format_value(value: float | None) -> str:
    """A value as a reader would write it: thousands separators, no trailing zeros."""
    if value is None:
        return "—"
    text = f"{value:,.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def group_citations(citations: Iterable[Any], max_series: int = 6) -> list[CitationGroup]:
    """Group citations by series, evidence kinds in a stable order.

    Groups keep the order in which series first appear, which is retrieval rank:
    the series most relevant to the question stays on top. Within a group the
    order is observed, derived, forecast.

    The group label prefers a human label over a series key. Forecast citations
    carry only the key, so taking the label from the first non-key citation avoids
    headlining a group with ``cpi_yoy`` when the observation says "CPI inflation".
    """
    by_series: dict[str, dict[str, Any]] = {}
    for cite in citations:
        key = _get(cite, "series_key") or ""
        bucket = by_series.setdefault(key, {"label": "", "items": []})
        label = _get(cite, "label") or ""
        if label and label != key and not bucket["label"]:
            bucket["label"] = label

        source = _get(cite, "source") or "observation"
        value = _get(cite, "value")
        bucket["items"].append(
            CitationItem(
                source=source,
                source_label=SOURCE_LABEL.get(source, source.title()),
                period=str(_get(cite, "period") or ""),
                value=value,
                value_text=format_value(value),
                detail=str(_get(cite, "detail") or ""),
            )
        )

    rank = {name: i for i, name in enumerate(SOURCE_ORDER)}
    groups: list[CitationGroup] = []
    for key, bucket in by_series.items():
        items = sorted(bucket["items"], key=lambda item: rank.get(item.source, len(rank)))
        groups.append(
            CitationGroup(
                series_key=key,
                label=bucket["label"] or key,
                items=tuple(items),
            )
        )
    return groups[:max_series]


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
VerificationState = Literal["verified", "rejected", "needs_input", "unanswered"]


@dataclass(frozen=True)
class Verification:
    state: VerificationState
    headline: str
    detail: str
    figures: int


def verification(answer: Any) -> Verification:
    """Say plainly what the verification gate did with this answer.

    ``figures`` counts cited values, not numbers parsed from the prose: the gate
    checks the prose against the evidence, and the citations are that evidence, so
    the count is what the reader can actually inspect.
    """
    kind = _get(answer, "kind")
    unverified = list(_get(answer, "unverified_numbers") or [])
    citations = list(_get(answer, "citations") or [])
    figures = sum(1 for c in citations if _get(c, "value") is not None)

    if kind == "clarify":
        return Verification(
            state="needs_input",
            headline="No figures shown yet",
            detail="The question could mean several things, so nothing was looked up until it is narrowed.",
            figures=0,
        )

    if kind != "answer":
        return Verification(
            state="unanswered",
            headline="Not answered",
            detail="Nothing in the warehouse covers this, so no figures were given.",
            figures=0,
        )

    if unverified:
        return Verification(
            state="rejected",
            headline="Draft rejected",
            detail=(
                f"The generated text contained {len(unverified)} figure"
                f"{'' if len(unverified) == 1 else 's'} that match no stored value, "
                "so it was discarded. The cited values below are the verified ones."
            ),
            figures=figures,
        )

    return Verification(
        state="verified",
        headline="Verified",
        detail=(
            f"Every figure in this answer matches a stored value ({figures} cited)."
            if figures
            else "Every figure in this answer matches a stored value."
        ),
        figures=figures,
    )


# --------------------------------------------------------------------------- #
# Serialisation for the HTTP API
# --------------------------------------------------------------------------- #
def group_to_dict(group: CitationGroup) -> dict[str, Any]:
    return {
        "series_key": group.series_key,
        "label": group.label,
        "items": [
            {
                "source": item.source,
                "source_label": item.source_label,
                "period": item.period,
                "value": item.value,
                "value_text": item.value_text,
                "detail": item.detail,
            }
            for item in group.items
        ],
    }


def verification_to_dict(result: Verification) -> dict[str, Any]:
    return {
        "state": result.state,
        "headline": result.headline,
        "detail": result.detail,
        "figures": result.figures,
    }


# --------------------------------------------------------------------------- #
# HTML for the Streamlit tab
# --------------------------------------------------------------------------- #
def citations_html(groups: Iterable[CitationGroup]) -> str:
    """Grouped evidence as HTML. Every interpolated string is escaped.

    Labels and details originate in the catalogue and the model, and Streamlit
    renders this with ``unsafe_allow_html``, so escaping here is what stops a
    stray ``<`` in a label from breaking the page.
    """
    rows: list[str] = []
    for group in groups:
        chips = []
        for item in group.items:
            period = f" · {html.escape(item.period)}" if item.period else ""
            detail = f" · {html.escape(item.detail)}" if item.detail else ""
            title = html.escape(SOURCE_HELP.get(item.source, ""))
            chips.append(
                f'<span class="cite {html.escape(item.source)}" title="{title}">'
                f'<i>{html.escape(item.source_label)}</i> '
                f"<b>{html.escape(item.value_text)}</b>{period}{detail}</span>"
            )
        rows.append(
            '<div class="cite-group">'
            f'<span class="cite-label">{html.escape(group.label)}</span>'
            f'{"".join(chips)}</div>'
        )
    return f'<div class="cite-block">{"".join(rows)}</div>' if rows else ""


def verification_html(result: Verification) -> str:
    return (
        f'<div class="verify {html.escape(result.state)}">'
        f"<b>{html.escape(result.headline)}</b> {html.escape(result.detail)}</div>"
    )
