"""The agent: question → retrieval → tools → verified answer.

Design note on the language model
---------------------------------
The model is an *optional* final step, not the mechanism. Everything that
matters — which series are relevant, what the values are, whether the answer is
defensible — is decided by deterministic code. That ordering means:

- With no API key the chatbot still answers correctly, in plainer prose. Useful
  on a laptop, usable in a demo, and it makes the retrieval quality visible
  without a model papering over it.
- With a model the prose improves, but the numbers cannot change: ``answers.gate``
  rejects any figure no tool produced.

Two things the agent must get right that a naive RAG loop does not:

**Clarification.** "How has the economy improved?" is genuinely ambiguous —
since when, and improved on whose terms? The agent asks rather than guessing,
but only when the ambiguity is real. If the question names a series and a
period, it just answers.

**Follow-ups.** "And unemployment?" after a question about inflation must not
reset the context. The session carries the series discussed and the window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import store
from . import answers, graph as graph_mod, tools

# Phrases that signal a follow-up rather than a fresh question. When one of
# these opens the message, the previous turn's series are carried forward.
_FOLLOW_UP = re.compile(
    r"^\s*(and|what about|how about|same for|and what|then|also)\b", re.IGNORECASE
)

# Ambiguity markers: the user asked for a judgement without naming a yardstick.
_VAGUE = re.compile(
    r"\b(improved?|improving|better|worse|worsening|doing|going|performing|health|healthy|"
    r"strong|weak|good|bad|okay|ok)\b",
    re.IGNORECASE,
)

# A window the user named explicitly, so we do not have to ask for one.
#
# The trailing `(?:\s+ago)?` matters: "CPI like 10 years ago compared to now"
# puts the number *after* the preposition, and the original pattern only matched
# "since 2010" style phrasing. Missing it silently fell back to a 5-year default
# window and answered a different question than the one asked.
_EXPLICIT_WINDOW = re.compile(
    r"\b(?:since|from|over|past|last|in|like|about|around)?\s*"
    r"(\d{4}|\d+\s*(?:year|month|quarter)s?)\s*(?:ago)?\b",
    re.IGNORECASE,
)

# Does the question ask about movement over time? Only then does the composer
# spend words on a window change. "What is CPI now?" wants one sentence; adding
# "over the window it is down 0.5 pp" to that is noise, and doing it for four
# series at once is how the answer became an unreadable wall.
_CHANGE_SEEKING = re.compile(
    r"\b(chang\w*|compar\w*|vs\.?|versus|trend\w*|ago|since|over|improv\w*|"
    r"worsen\w*|better|worse|better|rose|rise|risen|fell|fall|fallen|increas\w*|decreas\w*|"
    r"mov\w*|direction)\b",
    re.IGNORECASE,
)

# How many series to fetch. Lowered from 4: four series x three sentences each
# is twelve sentences, which is a report rather than an answer. Two is enough
# to compare, and the primary series always gets the detail.
_MAX_RETRIEVED = 3

# Minimum lexical score for a question to count as being about this data at all.
# Calibrated against the real graph: a genuine question lands well above 10
# ("rand" = 33, "repo rate" = 32), while an off-topic one that collides on a
# single shared token lands around 3.
MIN_SEED_SCORE = 8.0

# How many observations of one series to put in front of the reader or model.
# 60 covers five years of monthly data, or sixty years of annual data, and stays
# well inside a readable prompt.
MAX_CONTEXT_POINTS = 60


@dataclass
class Turn:
    """One exchange, retained so follow-ups resolve."""

    question: str
    answer: answers.Answer
    series: list[str] = field(default_factory=list)


@dataclass
class Session:
    """Conversation state. In-memory by design.

    Persisting chat history would need a store and a privacy answer. The
    conversation is short-lived and the dashboard already has no auth, so
    keeping this in process memory is the honest scope.
    """

    turns: list[Turn] = field(default_factory=list)

    @property
    def last_series(self) -> list[str]:
        return self.turns[-1].series if self.turns else []

    def remember(self, turn: Turn) -> None:
        self.turns.append(turn)
        # Bounded: an unbounded transcript becomes a prompt nobody reads and a
        # memory leak in a long-lived container.
        if len(self.turns) > 12:
            self.turns = self.turns[-12:]


class Agent:
    """Answers questions about the warehouse, or explains why it cannot."""

    def __init__(
        self,
        *,
        llm: Callable[[str, answers.Evidence, list[str]], str | None] | None = None,
    ) -> None:
        self._llm = llm
        self._graph: graph_mod.Graph | None = None

    # ------------------------------------------------------------------ #
    # graph lifecycle
    # ------------------------------------------------------------------ #
    def graph(self) -> graph_mod.Graph:
        """Build the graph once per process.

        Rebuilding on every question would re-read 33 rows and re-derive the
        semantic model per request; the underlying metadata changes at most once
        a day, so a process-lifetime cache is both correct and cheap.
        """
        if self._graph is None:
            rows = store.indicator_metadata().to_dict("records")
            try:
                from .. import semantic

                model = semantic.build_model()
            except Exception:  # noqa: BLE001 - the graph is useful without metrics
                model = None
            self._graph = graph_mod.build_graph(rows, model)
        return self._graph

    def refresh_graph(self) -> None:
        """Drop the cached graph — call after a data refresh."""
        self._graph = None

    # ------------------------------------------------------------------ #
    # main entry point
    # ------------------------------------------------------------------ #
    def ask(self, question: str, session: Session | None = None) -> answers.Answer:
        session = session or Session()
        text = (question or "").strip()
        if not text:
            return answers.Answer(
                kind="clarify",
                text="What would you like to know about the South African economy?",
                follow_up="For example: what is CPI inflation now, or how has the repo rate changed since 2020?",
            )

        g = self.graph()

        # Follow-ups inherit the previous turn's series before scoring, so
        # "and unemployment?" is read as a new series in the same conversation
        # rather than as a question about the word "unemployment" in isolation.
        is_follow_up = bool(_FOLLOW_UP.match(text)) and bool(session.last_series)
        seeds = self._seeds(text, g)
        if is_follow_up:
            # A follow-up names its own subject: "and unemployment?" must resolve
            # to unemployment, not to whatever the previous answer was about.
            # Only when the question names nothing do we fall back to context.
            if seeds:
                seeds = [k for k in seeds if not k.startswith(("theme:", "category:"))] or seeds
            else:
                seeds = list(session.last_series)

        expansion = g.expand(seeds, hops=2) if seeds else {}
        ranked = self._rank_series(expansion, seeds, g, text)

        evidence = self._gather(text, ranked, seeds)

        # Ask for clarification only when the question is genuinely open: vague
        # wording, no explicit window, and a theme-level subject rather than a
        # named series.
        #
        # This is a direct return, not a pass through `gate`: gate answers a
        # question when it has evidence, so routing clarification through it
        # produced the "something went wrong" branch instead of a follow-up.
        if self._needs_clarification(text, evidence, ranked, seeds, is_follow_up):
            clarify = answers.Answer(
                kind="clarify",
                text=(
                    "That could mean a few different things, so I want to check "
                    "before giving you numbers."
                ),
                follow_up=answers._clarify_hint(ranked or seeds),
                confidence=0.3,
                evidence_series=ranked[:4],
            )
            session.remember(Turn(text, clarify, self._series_keys(evidence)))
            return clarify

        model_text = None
        if self._llm is not None and not evidence.is_empty():
            try:
                model_text = self._llm(text, evidence, ranked)
            except Exception as exc:  # noqa: BLE001 - a model outage must not fail the answer
                evidence.notes.append(f"language model unavailable: {exc}")

        if model_text is None and not evidence.is_empty():
            # Deterministic path. Not a fallback hack: this is the answer, and
            # on a warehouse of 33 series it is often better than prose because
            # it states exactly what moved and by how much.
            model_text = self._compose(text, evidence)

        answer = answers.gate(
            question=text,
            evidence=evidence,
            graph_hits=ranked or seeds,
            model_answer=model_text,
        )
        session.remember(Turn(text, answer, self._series_keys(evidence)))
        return answer

    # ------------------------------------------------------------------ #
    # retrieval
    # ------------------------------------------------------------------ #
    def _seeds(self, text: str, g: graph_mod.Graph) -> list[str]:
        if not _vocabulary_grounded(text, g):
            return []
        scored = g.score(text)
        if not scored:
            return []
        ordered = sorted(scored.items(), key=lambda kv: -kv[1])
        top_score = ordered[0][1]

        # Keep seeds that are clearly dominant, not every weak match — a noisy
        # seed set pollutes the expansion and drags in unrelated series.
        return [key for key, value in ordered[:5] if value >= top_score * 0.45]

    def _rank_series(
        self,
        expansion: dict[str, float],
        seeds: list[str],
        g: graph_mod.Graph,
        question: str = "",
    ) -> list[str]:
        """Order candidate series so the one actually asked about comes first.

        Graph expansion scores *relatedness*, not directness: asking about CPI
        reached the repo rate and bond yield too, and bond yield happened to
        score highest, so the answer led with the wrong series. What matters for
        the lead sentence is which series the question *names*, so candidates are
        re-ranked by direct vocabulary overlap with the question — the same
        measure used for grounding, applied here for ordering.
        """
        ranked = [key for key, _ in sorted(expansion.items(), key=lambda kv: -kv[1])]
        series = [key for key in ranked if g.nodes[key].kind == "series"]
        named = [key for key in seeds if g.nodes.get(key) and g.nodes[key].kind == "series"]
        candidates = list(dict.fromkeys(named + series))

        if not question:
            return candidates[:_MAX_RETRIEVED]

        query_terms = {t for t in graph_mod._tokens(question) if t.isalpha()}

        def directness(key: str) -> tuple[int, int, float]:
            node = g.nodes[key]
            vocab = set(graph_mod._tokens(node.label))
            for synonym in node.synonyms:
                vocab |= set(graph_mod._tokens(synonym))
            # Whole key also counts, so "usd_zar" matches "usd zar" phrasing.
            vocab |= set(graph_mod._tokens(key.replace("_", " ")))
            overlap = len(query_terms & vocab)
            # Prefer observed data over projections. The IMF/WEO series carry
            # labels like "IMF: unemployment" and their latest point is a *2031
            # forecast*, so leading with them states a projection as though it
            # were the current reading — the single most misleading thing this
            # composer could do.
            projected = 1 if _is_projection(node) else 0
            return (projected, -overlap, -expansion.get(key, 0.0))

        ordered = sorted(candidates, key=directness)

        # Drop secondary series that share nothing with the question. "What is
        # CPI now?" was pulling in the CPI *index* level as "for context", which
        # says the same thing twice in different units and reads as noise.
        relevant = [k for k in ordered if -directness(k)[1] > 0]
        if relevant:
            # Keep at most one zero-overlap series, and only if nothing else was
            # named — otherwise the answer drifts off the question asked.
            tail = [k for k in ordered if -directness(k)[1] == 0][:1]
            ordered = relevant + (tail if len(relevant) < 2 else [])

        return ordered[:_MAX_RETRIEVED]

    # ------------------------------------------------------------------ #
    # tool calls
    # ------------------------------------------------------------------ #
    def _gather(self, text: str, ranked: list[str], seeds: list[str]) -> answers.Evidence:
        evidence = answers.Evidence()
        window = _window_from(text)

        for key in ranked:
            try:
                # When the user named a window, honour it but still cap the
                # resolution: "since 1980" on a monthly series is 550 points,
                # which is more context than any reader or model needs.
                #
                # The default count is per frequency, not a flat 60. Sixty
                # *annual* observations reached back to 1991 and produced
                # "unemployment is up 9.25 pp since 1991" for a question asking
                # whether it is getting better now — technically true, useless as
                # an answer. Ten years is the comparison people actually mean.
                snapshot = tools.get_series(
                    key,
                    since=window,
                    periods=None if window else _default_periods(key),
                    max_points=MAX_CONTEXT_POINTS,
                )
            except tools.ToolError as exc:
                evidence.notes.append(f"{key}: {exc}")
                continue
            evidence.series[key] = snapshot.to_dict()
            try:
                evidence.forecasts[key] = tools.get_forecast(key)
            except tools.ToolError:
                # Not every series is forecast; that is normal, not a failure.
                pass

        if not evidence.series and seeds:
            # The question was grounded in the graph but no series data came
            # back. Falling back to headline readings keeps a broad "how is the
            # economy doing" useful.
            #
            # Gated on `seeds`: without that, "what is the meaning of life"
            # matched nothing, fell back to readings, and cheerfully answered
            # with GDP figures. An ungrounded question must reach the refusal
            # path, not the fallback.
            try:
                evidence.readings = tools.latest_readings()
            except tools.ToolError as exc:
                evidence.notes.append(f"readings: {exc}")
        return evidence

    @staticmethod
    def _series_keys(evidence: answers.Evidence) -> list[str]:
        return list(evidence.series)[:4]

    # ------------------------------------------------------------------ #
    # clarification policy
    # ------------------------------------------------------------------ #
    def _needs_clarification(
        self,
        text: str,
        evidence: answers.Evidence,
        ranked: list[str],
        seeds: list[str],
        is_follow_up: bool,
    ) -> bool:
        """Ask only when the question is genuinely underspecified.

        Clarifying when the answer is already determinable is the most annoying
        failure mode a data chatbot has, so the bar is deliberately high. All
        three must hold:

        1. vague evaluative wording ("improved", "doing")
        2. no window named by the user
        3. the *seed* was a theme, not a series

        ``seeds`` rather than ``ranked`` is what makes (3) work: after expansion
        every theme has resolved to concrete series keys, so testing the ranked
        list asked "is a theme at the top of the series?" — which is never true.
        "Is inflation getting better?" seeds CPI directly and answers; "How has
        the economy improved?" seeds the economy theme and asks.
        """
        if is_follow_up or not ranked:
            return False
        if _EXPLICIT_WINDOW.search(text):
            return False
        if not _VAGUE.search(text):
            return False
        return bool(seeds) and seeds[0].startswith(("theme:", "category:"))

    # ------------------------------------------------------------------ #
    # deterministic prose
    # ------------------------------------------------------------------ #
    def _compose(self, question: str, evidence: answers.Evidence) -> str:
        """Write the answer from tool output alone, with no model involved.

        Every clause is derived from a field the tools returned, which is why
        this output can never contain an unverified number.

        Written to answer the question asked, not to describe every series that
        was retrieved. The earlier version emitted latest + window change +
        forecast for *each* of four series, which produced twelve sentences of
        undifferentiated figures — technically correct, unreadable, and it
        buried the one series the user actually asked about.
        """
        if not evidence.series:
            return self._compose_readings(evidence)

        keys = list(evidence.series)
        primary = keys[0]
        others = keys[1:]
        wants_change = bool(_CHANGE_SEEKING.search(question))
        window = _window_from(question)

        parts: list[str] = []
        parts.append(self._sentence_primary(question, primary, evidence, wants_change, window))

        # Secondary series get one clause each, and only when their movement
        # contrasts with the primary — the reason to mention a second series is
        # comparison, not completeness.
        if others:
            parts.append(self._sentence_others(primary, others, evidence, wants_change))

        return " ".join(p for p in parts if p)

    def _sentence_primary(
        self,
        question: str,
        key: str,
        evidence: answers.Evidence,
        wants_change: bool,
        window: str | None,
    ) -> str:
        snap = evidence.series[key]
        label = snap.get("label", key)
        unit = snap.get("unit") or ""
        latest = snap.get("latest") or {}
        yoy = snap.get("year_on_year") or {}
        since = snap.get("change_since_first_point") or {}
        higher = snap.get("higher_is_better")
        as_rate = _is_rate(unit, label)

        if latest.get("value") is None:
            return ""

        # Lead with the value and when it is from — the question is almost always
        # "what is it", and the period is what makes the number checkable.
        # A future-dated period is labelled a projection: saying "unemployment
        # is 31.9% as of 2031" would read as an observation of something that
        # has not happened.
        is_projection = _is_future(latest.get("period"))
        qualifier = " (IMF projection)" if is_projection else ""
        text = (
            f"{label} is {_fmt(latest['value'], unit)} as of {latest.get('period')}"
            f"{qualifier}"
        )

        # A window change is the point of a comparison question, so it leads.
        if wants_change and since.get("from_period") and since.get("absolute") is not None:
            span = _span_phrase(window or since.get("from_period"))
            moved = "up" if since["absolute"] > 0 else "down"
            amount = _fmt(abs(since["absolute"]), "pp" if as_rate else (unit or ""))
            verdict = _verdict(since["absolute"], higher)
            text += (
                f" — {moved} {amount} {span}"
                f"{f', which is {verdict}' if verdict else ''}"
                f" (from {_fmt(since['from_value'], unit)} in {since['from_period']})"
            )
        elif yoy.get("absolute") is not None:
            moved = "up" if yoy["absolute"] > 0 else "down"
            amount = _fmt(abs(yoy["absolute"]), "pp" if as_rate else (unit or ""))
            verdict = _verdict(yoy["absolute"], higher)
            text += (
                f", {moved} {amount} year on year"
                f"{f' ({verdict})' if verdict else ''}"
            )

        text += "."

        # One forecast sentence, and only when asked for change or a forecast.
        forecast = evidence.forecasts.get(key)
        if forecast and forecast.get("path") and wants_change:
            point = forecast["path"][-1]
            text += (
                f" The {forecast.get('model')} forecast puts it at "
                f"{_fmt(point.get('forecast'), unit)} by {point.get('period')}."
            )

        return text

    def _sentence_others(
        self,
        primary: str,
        others: list[str],
        evidence: answers.Evidence,
        wants_change: bool,
    ) -> str:
        """One contrast sentence about the other series, or nothing.

        Deliberately terse. The secondary series exists to give context to the
        primary, so it contributes a value and a direction — not its own
        forecast, window history and units.

        Series that restate the primary in another unit are dropped: asking
        "what is CPI inflation?" was answered with "for context: CPI (all items)
        is 205.4 index", which is the same subject twice and reads as noise.
        """
        clauses: list[str] = []
        primary_terms = _subject_terms(primary)

        for key in others[:2]:
            if _subject_terms(key) & primary_terms:
                continue
            snap = evidence.series[key]
            label = _clean_label(snap.get("label", key))
            unit = snap.get("unit") or ""
            latest = snap.get("latest") or {}
            if latest.get("value") is None:
                continue

            move = ""
            if wants_change:
                yoy = snap.get("year_on_year") or {}
                if yoy.get("absolute") is not None:
                    moved = "up" if yoy["absolute"] > 0 else "down"
                    amount = _change_amount(abs(yoy["absolute"]), unit)
                    move = f", {moved} {amount} y/y"
            clauses.append(f"{label} is {_fmt(latest['value'], unit)}{move}")

        if not clauses:
            return ""
        if len(clauses) == 1:
            return f"For context, {clauses[0]}."
        return f"For context: {'; '.join(clauses)}."

    def _compose_readings(self, evidence: answers.Evidence) -> str:
        """Fallback when no series was resolved but headline readings exist."""
        parts = []
        for row in evidence.readings:
            if row.get("latest_value") is None:
                continue
            parts.append(
                f"{row['label']} {_fmt(row['latest_value'], row.get('unit') or '')} "
                f"({row.get('latest_period')})"
            )
        return "Latest readings: " + "; ".join(parts) + "." if parts else ""


def _span_phrase(from_period: str | None) -> str:
    """Turn a start period into 'since 2016' / 'over the last 10 years'."""
    if not from_period:
        return "over the window"
    year = str(from_period)[:4]
    if year.isdigit():
        return f"since {year}"
    return "over the window"


def _verdict(absolute: float, higher_is_better: bool | None) -> str | None:
    """A plain-language judgement, only where the metadata supports one."""
    if higher_is_better is None or abs(absolute) < 1e-9:
        return None
    return "an improvement" if (absolute > 0) == higher_is_better else "a deterioration"


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def _vocabulary_grounded(text: str, g: graph_mod.Graph) -> bool:
    """Is this question about the vocabulary this data actually uses?

    Tested against each node's **label and synonyms** — the curated vocabulary —
    rather than its description or a score threshold. That distinction is the
    whole trick:

    - Descriptions are prose. Matching them incidentally let "who won the 2010
      world cup" look like a GDP question, because a description happened to
      contain "won" and 2010 is inside a data range.
    - Labels and synonyms are deliberate. If a question shares a content word
      with them, or matches a theme synonym, it is genuinely about this data.

    This also avoids the opposite failure: "is inflation getting better?" scored
    *below* any sensible threshold because "inflation" is common across the
    corpus, yet it is obviously a question about CPI. Vocabulary grounding gets
    that right without a tuned constant.
    """
    query_terms = {t for t in graph_mod._tokens(text) if t.isalpha()}
    if not query_terms:
        return False

    for node in g.nodes.values():
        vocabulary = set(graph_mod._tokens(node.label))
        for synonym in node.synonyms:
            vocabulary |= set(graph_mod._tokens(synonym))
        if query_terms & vocabulary:
            return True

    return False


def _matched_words(text: str, key: str, g: graph_mod.Graph) -> int:
    """Distinct *topical* word matches between the question and a node.

    Excluded on purpose:

    - **Numbers.** A year like 2010 appears in a series' date range, so counting
      it as a match let "who won the 2010 world cup" be read as a question about
      GDP growth.
    - **Verb-ish filler.** "won" in "who won the world cup" collided with the
      word "won" inside a description, which is a grammatical accident rather
      than a topic. Only nouns and adjectives carry subject matter here.

    What remains is the vocabulary that actually identifies a subject.
    """
    node = g.nodes.get(key)
    if node is None:
        return 0
    query_terms = {
        t for t in graph_mod._tokens(text) if t.isalpha() and t not in _NON_TOPICAL
    }
    node_terms = set(graph_mod._tokens(node.search_text()))
    return len(query_terms & node_terms)


# Words that appear inside descriptions for grammatical reasons and therefore
# imply nothing about what the user is asking about. Kept short deliberately:
# every entry is a word that produced a false positive in testing.
_NON_TOPICAL = frozenset(
    """won win lose lost beat game match play played cup world team champion
    season final score against player""".split()
)


def _names_a_series(text: str, key: str, g: graph_mod.Graph) -> bool:
    """True when the question contains the series key or label outright."""
    node = g.nodes.get(key)
    if node is None:
        return False
    haystack = text.lower().replace("_", " ")
    if key.replace("_", " ") in haystack or key.replace("_", "") in haystack.replace(" ", ""):
        return True
    label = node.label.lower()
    return bool(label) and label in haystack


def _fmt(value: Any, unit: str = "") -> str:
    if value is None:
        return "n/a"
    number = float(value)
    # Magnitudes that span rates (4.5) and rand levels (16.68) need different
    # precision; a flat format renders either 4.5000 or 17.
    if abs(number) >= 1000:
        text = f"{number:,.0f}"
    elif abs(number) >= 100:
        text = f"{number:.1f}"
    else:
        text = f"{number:.2f}".rstrip("0").rstrip(".")
    return f"{text}{unit}" if unit in {"%", " index"} else f"{text} {unit}".strip()


def _default_periods(key: str) -> int:
    """How many observations to fetch when the user named no window.

    Ten years' worth, not a flat sixty points. Sixty observations is five years
    for a monthly series but sixty years for an annual one, so a single constant
    gave wildly different horizons depending on the series — and for annual data
    turned "is it getting better?" into a comparison with 1991.
    """
    try:
        rows = tools.run_select(
            "SELECT frequency FROM indicators WHERE series_key = ?", [key]
        )
        freq = (rows[0].get("frequency") or "M").upper() if rows else "M"
    except Exception:  # noqa: BLE001 - a lookup failure must not break the answer
        freq = "M"

    return {"M": 120, "Q": 40, "A": 10}.get(freq, 60)


def _subject_terms(key: str) -> set[str]:
    """Words identifying a series' *subject*, for de-duplication.

    Used to drop a secondary series that restates the primary in another unit —
    "CPI inflation" and "CPI (all items)" are the same subject, so offering the
    index level as "context" says the same thing twice in different units.
    """
    words = set(graph_mod._tokens(key.replace("_", " ")))
    # Drop the measure qualifier so inflation_yoy and inflation_index collide.
    generic = {"yoy", "index", "annual", "avg", "average", "rate", "real", "nominal", "usd"}
    return words - generic


def _clean_label(label: str) -> str:
    """Strip source prefixes and parenthetical units from a display label.

    "IMF: unemployment" reads as an authority claim inside a sentence, and
    "Leading indicator is 119.6 index (2019=100), up 2.24 index (2019=100) y/y"
    says the unit twice. The source is recorded in the citation, where it
    belongs.
    """
    text = label
    for prefix in ("IMF: ", "WEO: "):
        if text.startswith(prefix):
            text = text[len(prefix):]
    # Remove a trailing "(...)" qualifier, and any "(...)" inside the value part.
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    return text or label


def _change_amount(absolute: float, unit: str) -> str:
    """Format a change magnitude with the right unit suffix.

    A rate moving 1.1 points is "1.1 pp", not "1.1 %" — the second reads as a
    relative move and is the mistake that made CPI look like it tripled.
    """
    is_rate = any(w in (unit or "").lower() for w in ("%", "percent"))
    if is_rate:
        return f"{absolute:.2f} pp"
    # Index units like "index (2019=100)" are noisy mid-sentence; the number and
    # direction carry the meaning.
    if "index" in (unit or "").lower():
        return f"{absolute:.2f}"
    suffix = f" {unit}" if unit and unit not in {"%"} else ""
    return f"{absolute:.2f}{suffix}"


def _is_future(period: str | None) -> bool:
    """Is this period in the future, meaning the value is a projection?"""
    if not period:
        return False
    from datetime import date

    year = str(period)[:4]
    return year.isdigit() and int(year) > date.today().year


def _is_projection(node: graph_mod.Node) -> bool:
    """Is this series a published projection rather than an observation?

    The IMF/WEO series end in 2031 and are labelled "IMF: <thing>". Their
    "latest" value is therefore a forecast, and reporting it as the current
    reading is worse than useless. Detected from the label rather than a
    hand-maintained key list, so a new WEO-driven series is handled correctly.
    """
    return node.label.lower().startswith(("imf:", "weo")) or node.key.startswith(("weo_", "imf_"))


def _is_rate(unit: str, label: str) -> bool:
    text = f"{unit} {label}".lower()
    return any(word in text for word in ("rate", "%", "percent", "inflation", "yoy"))


def _window_from(text: str) -> str | None:
    """Translate a named window into a start date, or None to use a default."""
    from datetime import date

    match = _EXPLICIT_WINDOW.search(text)
    if not match:
        return None
    raw = match.group(1).lower().strip()
    today = date.today()

    if raw.isdigit() and len(raw) == 4:
        return f"{raw}-01-01"

    amount = int(re.match(r"(\d+)", raw).group(1))
    if "year" in raw:
        return f"{today.year - amount}-{today.month:02d}-01"
    if "month" in raw:
        month = today.month - amount
        year = today.year + (month - 1) // 12
        return f"{year}-{(month - 1) % 12 + 1:02d}-01"
    if "quarter" in raw:
        months = amount * 3
        month = today.month - months
        year = today.year + (month - 1) // 12
        return f"{year}-{(month - 1) % 12 + 1:02d}-01"
    return None
