"""Typed answer contracts with a deterministic refusal gate.

Adapted from the ``agentic_typed_rag_pydanticai`` template in
Shubhamsaboo/awesome-llm-apps (Apache-2.0). The template's core idea is that an
answer is a *validated object*, not a string: it carries exact citations, a
confidence score, and an explicit ``answered`` flag. If retrieval is too weak it
refuses **before** calling the language model.

Why that pattern and not the graph-with-citations template
----------------------------------------------------------
The other candidate (``knowledge_graph_rag_citations``) answers with ``[1][2]``
markers into source text. That works for prose. It is the wrong shape here,
because the failure mode that matters is not "cited the wrong document" but
"stated a number nobody computed". So citations here point at **tool results and
data periods**, not at text spans, and the gate is arithmetic rather than
cosine similarity.

The gate
--------
Two independent refusal conditions, both checkable without a model:

1. **No evidence.** The question matched nothing in the semantic graph and no
   tool returned data. Refuse rather than improvise.
2. **Unverified figures.** The model produced numbers that do not appear in any
   tool result. This is the dangerous case: a fluent answer containing an
   invented statistic. ``verify_numbers`` is what makes the guarantee real.

A refusal is a first-class success state, not an error. Saying "I don't have
that" costs the user nothing; a confident wrong number about the repo rate
costs them money.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

# Numbers in prose, including negative, decimal and thousands-separated forms.
# Deliberately excludes a leading +/- sign used for range notation ("5-6%").
_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")

# ISO dates are masked before scanning. Without this, "2026-09-01" is read as
# three separate numbers and the "09" is reported as a fabricated figure — a
# false positive that rejected every correct answer mentioning a period.
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# "2020 to 2026" is a range of years, not two measurements.
_YEAR_PAIR = re.compile(r"\b\d{4}\s*(?:-|–|to|and)\s*\d{4}\b")

AnswerKind = Literal["answer", "clarify", "refuse"]


@dataclass
class Citation:
    """A checkable claim: this value, for this series, at this period.

    ``source`` distinguishes a stored observation from a model projection, so
    a reader can tell "CPI was 4.5% in September" from "CPI is forecast to be
    4.8% by September next year".
    """

    series_key: str
    label: str
    period: str
    value: float | None
    source: Literal["observation", "forecast", "derived"] = "observation"
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "series_key": self.series_key,
            "label": self.label,
            "period": self.period,
            "value": self.value,
            "source": self.source,
            "detail": self.detail,
        }


@dataclass
class Evidence:
    """Everything the tools returned, before any prose was written."""

    series: dict[str, dict] = field(default_factory=dict)
    forecasts: dict[str, dict] = field(default_factory=dict)
    readings: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def numbers(self) -> set[float]:
        """Every figure the tools produced, for the verification check.

        Rounded variants are included because an answer will legitimately say
        "4.5" where the stored value is 4.5012317. Comparing only exact values
        would reject correct answers.
        """
        found: set[float] = set()

        def add(value: Any) -> None:
            if isinstance(value, bool) or value is None:
                return
            if isinstance(value, (int, float)):
                number = float(value)
                found.add(round(number, 6))
                found.add(round(number, 2))
                found.add(round(number, 1))
                found.add(round(abs(number), 2))
                found.add(float(round(number)))
                # Percentage and basis-point framings of the same figure.
                for scale in (100.0, 1.0):
                    found.add(round(number * scale, 2))

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    # Periods and keys are not measurements.
                    if key in {"period", "from_period", "to_period", "series_key", "key"}:
                        continue
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)
            else:
                add(node)

        walk(self.series)
        walk(self.forecasts)
        walk(self.readings)
        return found

    def is_empty(self) -> bool:
        return not (self.series or self.forecasts or self.readings)

    def citations(self) -> list[Citation]:
        out: list[Citation] = []
        for key, snap in self.series.items():
            latest = snap.get("latest") or {}
            if latest.get("value") is not None:
                out.append(
                    Citation(
                        series_key=key,
                        label=snap.get("label", key),
                        period=str(latest.get("period", "")),
                        value=float(latest["value"]),
                        source="observation",
                        detail=snap.get("unit") or "",
                    )
                )
            yoy = snap.get("year_on_year") or {}
            if yoy.get("absolute") is not None:
                out.append(
                    Citation(
                        series_key=key,
                        label=snap.get("label", key),
                        period=f"{yoy.get('from_period')} to {yoy.get('to_period')}",
                        value=float(yoy["absolute"]),
                        source="derived",
                        detail="year-on-year change",
                    )
                )
        for key, fc in self.forecasts.items():
            for point in (fc.get("path") or [])[:1]:
                if point.get("forecast") is not None:
                    out.append(
                        Citation(
                            series_key=key,
                            label=key,
                            period=str(point.get("period", "")),
                            value=float(point["forecast"]),
                            source="forecast",
                            detail=f"{fc.get('model')} (horizon {fc.get('horizon')})",
                        )
                    )
        return out


@dataclass
class Answer:
    """The only thing the API will return. Never a bare string."""

    kind: AnswerKind
    text: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    follow_up: str | None = None
    unverified_numbers: list[float] = field(default_factory=list)
    evidence_series: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "text": self.text,
            "citations": [c.as_dict() for c in self.citations],
            "confidence": round(self.confidence, 3),
            "follow_up": self.follow_up,
            "unverified_numbers": self.unverified_numbers,
            "evidence_series": self.evidence_series,
        }


def _parse(text: str) -> list[float]:
    # Dates and year ranges are removed first: they are labels for the data, not
    # measurements, and their components otherwise read as invented figures.
    cleaned = _ISO_DATE.sub(" ", text)
    cleaned = _YEAR_PAIR.sub(" ", cleaned)

    found: list[float] = []
    for match in _NUMBER.finditer(cleaned):
        raw = match.group(0).replace(",", "")
        try:
            found.append(float(raw))
        except ValueError:
            continue
    return found


def verify_numbers(text: str, evidence: Evidence) -> list[float]:
    """Numbers in the prose that no tool produced.

    An empty list means every figure is traceable. This is the check that turns
    "the model usually doesn't hallucinate" into "the model cannot hallucinate
    here without the answer being rejected".

    Some numbers are legitimately unremarkable and are skipped: years (2026),
    small ordinals used as list markers, and the horizon counts a tool itself
    returned. Years are matched explicitly because otherwise every answer
    mentioning 2026 would fail verification.
    """
    allowed = evidence.numbers()
    suspect: list[float] = []

    for number in _parse(text):
        if 1940 <= number <= 2100 and number == int(number):
            continue  # a calendar year, not a measurement
        if number in (0.0, 1.0) and abs(number) == int(abs(number)):
            continue  # "1 of 3", "0.0%" placeholders
        if round(number, 2) in allowed or round(number, 1) in allowed:
            continue
        if round(abs(number), 2) in allowed or float(round(number)) in allowed:
            continue
        suspect.append(number)

    return suspect


def gate(
    *,
    question: str,
    evidence: Evidence,
    graph_hits: list[str],
    model_answer: str | None,
) -> Answer:
    """Decide whether to answer, ask for clarification, or refuse.

    Runs the checks in increasing cost: evidence presence is free, number
    verification is cheap, and only if both pass is the model's prose returned.
    """
    if evidence.is_empty() and not graph_hits:
        return Answer(
            kind="refuse",
            text=(
                "I could not find anything in the SA Insights data covering that. "
                "The dashboard tracks 33 South African macroeconomic series — "
                "things like inflation, the repo rate, GDP growth, unemployment "
                "and the rand. Ask about one of those and I can give you the "
                "numbers and their sources."
            ),
            confidence=0.0,
        )

    # A theme or category seed means the graph understood the *topic* but no
    # specific series was named. Answering the whole theme produces a wall of
    # figures that may not address what was asked, so ask first — this is the
    # one case where clarification genuinely helps.
    topic_only = bool(graph_hits) and all(
        hit.startswith(("theme:", "category:")) for hit in graph_hits
    )
    if topic_only and not evidence.is_empty():
        return Answer(
            kind="clarify",
            text=(
                "That spans a few indicators, so I want to check what you are "
                "after before giving you numbers."
            ),
            follow_up=_clarify_hint(graph_hits),
            confidence=0.3,
            evidence_series=list(evidence.series)[:5],
        )

    if evidence.is_empty():
        # The graph recognised the topic but no numbers were fetched. Asking a
        # clarifying question is better than a refusal here: the user is close.
        return Answer(
            kind="clarify",
            text=(
                "I know which indicators that touches, but I need a bit more "
                "detail before I can give you figures."
            ),
            follow_up=_clarify_hint(graph_hits),
            confidence=0.2,
            evidence_series=graph_hits[:5],
        )

    if model_answer is None:
        return Answer(
            kind="refuse",
            text="Something went wrong assembling an answer. The data is fine; the reply was not.",
            confidence=0.0,
            evidence_series=list(evidence.series)[:5],
        )

    suspect = verify_numbers(model_answer, evidence)
    citations = evidence.citations()

    if suspect:
        # Do not ship prose containing invented figures. The text is replaced
        # rather than edited: rewriting around the bad numbers would leave the
        # surrounding narrative just as unverifiable.
        return Answer(
            kind="answer",
            text=(
                "I found the data but could not produce an answer I can stand "
                "behind — some figures in the draft did not match any value the "
                "tools returned, so I have dropped the narrative. The underlying "
                "numbers are in the citations below."
            ),
            citations=citations,
            confidence=0.35,
            unverified_numbers=sorted(set(suspect))[:10],
            evidence_series=list(evidence.series)[:5],
        )

    # Confidence blends evidence breadth with citation count, capped below 1.0:
    # a statistical extrapolation should never read as certain.
    breadth = min(len(evidence.series) + len(evidence.forecasts), 4) / 4.0
    confidence = 0.55 + 0.35 * breadth

    return Answer(
        kind="answer",
        text=model_answer,
        citations=citations,
        confidence=min(confidence, 0.95),
        evidence_series=list(evidence.series)[:5],
    )


def _clarify_hint(graph_hits: list[str]) -> str:
    labels = [
        hit.split(":", 1)[-1].replace("_", " ") for hit in graph_hits[:3]
    ]
    if not labels:
        return "Which indicator do you mean?"
    if len(labels) == 1:
        return f"Do you mean {labels[0]}? If so, over what period?"
    joined = ", ".join(labels[:-1]) + f" or {labels[-1]}"
    return f"Which one — {joined}? And over what period?"
