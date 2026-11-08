"""Optional language model for phrasing — Gemini via its REST API.

What the model is allowed to do
-------------------------------
Rewrite already-verified facts into fluent prose. Nothing else. It cannot
introduce a figure (``answers.verify_numbers`` rejects any number absent from
the tool results), and it cannot change which series are discussed. If it fails,
returns nonsense, or is not configured at all, the deterministic composer in
``agent.py`` answers instead — the bot is useful with no model whatsoever.

Why REST and not the SDK
------------------------
One endpoint, one JSON body, no dependency. ``google-generativeai`` would add a
package and its own release cadence to fetch the same bytes, and this runs inside
a Modal image where every dependency is build time.

Why not put the key in code
---------------------------
It is read from ``GEMINI_API_KEY`` in the environment, injected from a Modal
secret. A key in a source file is a key in a git history, and this repository is
public.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from . import answers

ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)

# Model choice, verified rather than assumed.
#
# `gemini-2.5-flash-lite` appears in the /models listing but every
# generateContent call to it returns 404 "This model ... is not found" — a
# genuine API inconsistency, not a bad key or a wrong version. Both v1beta and v1
# reject it. `gemini-flash-lite-latest` works on v1beta (and is the cheaper,
# faster tier this task wants, since it is rewriting facts rather than reasoning).
# `gemini-2.5-flash` also works on both versions and is the fallback.
#
# Override with GEMINI_MODEL without a rebuild.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
FALLBACK_MODEL = "gemini-2.5-flash"

_SYSTEM = """You rewrite verified economic facts into clear, natural prose.

Rules, in order of importance:
1. Use ONLY the figures given in FACTS. Never introduce, recompute, round
   differently, or invent a number. No exceptions.
2. Never write a number with more than two decimal places. Write "4.46%", not
   "4.464328113591922%". Round to one decimal place for percentages unless the
   fact already has two.
3. Answer the USER QUESTION directly in the first sentence. Lead with the series
   the question is about, not with whichever fact appears first.
4. Be brief: two or three sentences. No preamble, no headings, no bullet lists,
   no restating the question.
5. Where a change is given, say whether it is good or bad news only if the facts
   state higher_is_better. Say "worse" or "better", not "bad news".
6. If a fact is a projection (a future period), call it a forecast.
7. If the facts do not answer the question, say so in one sentence.
8. South African context: "the rand" means ZAR per USD; "the repo rate" is the
   policy rate; inflation figures are year-on-year unless stated otherwise.

ANSWERING HISTORICAL QUESTIONS
"change over window" IS the historical comparison. When asked "what was X like
10 years ago" or "how has X changed since 2015", answer from that field: the
"from" value is the historical reading and the "latest" value is today. The
"from" period is the earliest data available, which may be a year or two later
than the year asked about - that is expected and is not a reason to refuse. Only
refuse if the series is absent from FACTS entirely.

ANSWERING FOLLOW-UP QUESTIONS
The user may refer to earlier turns with pronouns such as "they", "them", "that",
"those numbers", "the figures". FACTS always describes the series actually being
discussed, so answer about those series directly. Never say you do not know what
"they" refers to - the facts define it.

Write plain text. No markdown, no citations - the interface renders sources
separately."""


def available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def _facts_block(question: str, evidence: answers.Evidence) -> str:
    """Serialise the tool results into a compact, unambiguous fact list.

    Trimmed hard on purpose. Sending every observation of every series invites
    the model to comment on series nobody asked about, which is exactly the
    behaviour the deterministic composer was just fixed for.

    Numbers are pre-rounded before they reach the model. Left raw, it copied
    ``4.464328113591922%`` verbatim into prose — the value passed verification
    because the tool did produce it, so the guarantee held while the answer
    still looked broken. Rounding here fixes the presentation at the source.
    """

    def num(value: object, digits: int = 2) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    lines: list[str] = []

    for key, snap in evidence.series.items():
        label = snap.get("label", key)
        unit = snap.get("unit") or ""
        latest = snap.get("latest") or {}
        yoy = snap.get("year_on_year") or {}
        since = snap.get("change_since_first_point") or {}
        higher = snap.get("higher_is_better")

        if latest.get("value") is None:
            continue

        # Changes carry an explicit unit. Without it the model wrote "the repo
        # rate has risen by 0.75" with no unit at all — the value was right and
        # the sentence was still wrong.
        is_rate = "%" in (unit or "")
        delta_unit = " percentage points" if is_rate else (f" {unit}" if unit else "")

        parts = [
            f"- {label} ({key}): latest {num(latest.get('value'))}{unit} "
            f"at {latest.get('period')}"
        ]
        if higher is not None:
            parts.append(f"higher_is_better={higher}")
        if yoy.get("absolute") is not None:
            parts.append(
                f"change over 12 months: {float(yoy['absolute']):+.2f}{delta_unit} "
                f"(from {num(yoy.get('from_value'))}{unit} at {yoy.get('from_period')})"
            )
        if since.get("absolute") is not None:
            parts.append(
                f"change over window ({since.get('from_period')} to "
                f"{latest.get('period')}): {float(since['absolute']):+.2f}{delta_unit} "
                f"(historical reading {num(since.get('from_value'))}{unit}, "
                f"today {num(latest.get('value'))}{unit})"
            )
        forecast = evidence.forecasts.get(key)
        if forecast and forecast.get("path"):
            point = forecast["path"][-1]
            parts.append(
                f"forecast: {num(point.get('forecast'))}{unit} by {point.get('period')} "
                f"model {forecast.get('model')}"
            )
        lines.append("; ".join(parts))

    if evidence.readings:
        for row in evidence.readings:
            if row.get("latest_value") is None:
                continue
            lines.append(
                f"- {row['label']} ({row['key']}): {num(row['latest_value'])}"
                f"{row.get('unit') or ''} at {row.get('latest_period')}"
            )

    return "\n".join(lines) if lines else "(no facts)"


def phrase(
    question: str,
    evidence: answers.Evidence,
    hints: list[str],
    *,
    timeout: float = 20.0,
) -> str | None:
    """Rewrite the verified facts as prose. Returns None on any failure.

    Never raises: a language-model outage must degrade to the deterministic
    composer, not fail the request. The caller treats None as "use the facts
    verbatim", which is always a valid answer.
    """
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or evidence.is_empty():
        return None

    prompt = (
        f"USER QUESTION: {question}\n\n"
        f"FACTS (the only numbers you may use):\n{_facts_block(question, evidence)}\n\n"
        f"Related indicators mentioned: {', '.join(hints[:4]) or 'none'}"
    )

    body = {
        "systemInstruction": {"parts": [{"text": _SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.25,  # low: this is transcription, not creativity
            "maxOutputTokens": 320,
            "topP": 0.9,
        },
    }

    url = ENDPOINT.format(model=DEFAULT_MODEL)
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            # Header, not a query parameter: a key in a URL ends up in logs.
            "x-goog-api-key": key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # 404 means this model name is not served. Google's /models listing and
        # the generateContent endpoint disagree for at least one model, so retry
        # once on the known-good fallback rather than degrading to plain prose
        # for every future request because of a naming mismatch.
        if exc.code == 404 and DEFAULT_MODEL != FALLBACK_MODEL:
            print(f"gemini: {DEFAULT_MODEL} returned 404; retrying {FALLBACK_MODEL}")
            return _retry_with_fallback(body, key, timeout)
        print(f"gemini: request failed ({exc}); using deterministic composer")
        return None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"gemini: request failed ({exc}); using deterministic composer")
        return None
    except json.JSONDecodeError:
        print("gemini: response was not JSON; using deterministic composer")
        return None

    return _extract_text(payload)


def _retry_with_fallback(body: dict, key: str, timeout: float) -> str | None:
    """One retry against FALLBACK_MODEL. Never recurses further."""
    request = urllib.request.Request(
        ENDPOINT.format(model=FALLBACK_MODEL),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _extract_text(json.loads(response.read().decode("utf-8")))
    except Exception as exc:  # noqa: BLE001 - fallback failure is not fatal
        print(f"gemini: fallback also failed ({exc}); using deterministic composer")
        return None


def _extract_text(payload: dict) -> str | None:
    """Pull the text out of a Gemini response, tolerating safety blocks.

    A blocked or empty candidate is a normal outcome, not an error — returning
    None lets the deterministic composer take over rather than surfacing an
    exception for something the user cannot act on.
    """
    candidates = payload.get("candidates") or []
    if not candidates:
        feedback = payload.get("promptFeedback") or {}
        print(f"gemini: no candidates (blockReason={feedback.get('blockReason')})")
        return None

    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts).strip()

    if not text:
        return None
    return " ".join(text.split())
