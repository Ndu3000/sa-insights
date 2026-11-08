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
_EXPLICIT_WINDOW = re.compile(
    r"\b(?:since|from|over|past|last|in)\s+(\d{4}|\d+\s*(?:year|month|quarter)s?)\b",
    re.IGNORECASE,
)

_MAX_RETRIEVED = 4

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
        ranked = self._rank_series(expansion, seeds, g)

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
    ) -> list[str]:
        ranked = [key for key, _ in sorted(expansion.items(), key=lambda kv: -kv[1])]
        series = [key for key in ranked if g.nodes[key].kind == "series"]
        # Seeds that are themselves series come first; a user who named a series
        # means that series, not its neighbours.
        named = [key for key in seeds if g.nodes.get(key) and g.nodes[key].kind == "series"]
        ordered = list(dict.fromkeys(named + series))
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
                snapshot = tools.get_series(
                    key,
                    since=window,
                    periods=None if window else 60,
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
        """
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

            head = f"{label} is {_fmt(latest['value'], unit)} ({latest.get('period')})"
            if yoy.get("absolute") is not None:
                head += f", {_verb(yoy['absolute'], higher, _is_rate(unit, label))}"
            lines.append(head + ".")

            if since.get("absolute") is not None and since.get("from_period") != latest.get("period"):
                direction = "up" if since["absolute"] > 0 else "down"
                # A rate moving from 4.44 to 1.11 is a change of 3.32 percentage
                # points, not 3.32% — printing it with the series' own unit made
                # every rate change read as a relative move.
                amount = abs(since["absolute"])
                unit_suffix = "pp" if _is_rate(unit, label) else (unit or "")
                change = _fmt(amount, unit_suffix)
                lines.append(
                    f"Over the window, it is {direction} {change} from "
                    f"{_fmt(since['from_value'], unit)} in {since['from_period']}."
                )

            forecast = evidence.forecasts.get(key)
            if forecast and forecast.get("path"):
                point = forecast["path"][0]
                horizon = forecast.get("horizon")
                model = forecast.get("model")
                lines.append(
                    f"The current forecast ({model}) puts it at "
                    f"{_fmt(point.get('forecast'), unit)} by {point.get('period')} "
                    f"over a {horizon}-period horizon."
                )

        if not lines and evidence.readings:
            parts = []
            for row in evidence.readings:
                if row.get("latest_value") is None:
                    continue
                parts.append(
                    f"{row['label']} {_fmt(row['latest_value'], row.get('unit') or '')} "
                    f"({row.get('latest_period')})"
                )
            if parts:
                lines.append("Latest readings: " + "; ".join(parts) + ".")

        return " ".join(lines) if lines else ""


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


def _is_rate(unit: str, label: str) -> bool:
    text = f"{unit} {label}".lower()
    return any(word in text for word in ("rate", "%", "percent", "inflation", "yoy"))


def _verb(absolute: float, higher_is_better: bool | None, is_rate: bool) -> str:
    """"up 1.1 percentage points, which is worse" — direction plus judgement."""
    unit = "pp" if is_rate else ""
    magnitude = f"{abs(absolute):.2f}{(' ' + unit) if unit else ''}".strip()
    moved = "up" if absolute > 0 else "down"
    if higher_is_better is None:
        return f"{moved} {magnitude} year on year"
    improved = (absolute > 0) == higher_is_better
    return f"{moved} {magnitude} year on year ({'an improvement' if improved else 'a deterioration'})"


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
