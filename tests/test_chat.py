"""Tests for the chat layer.

These assert the *policies*, not the prose. Exact wording will change as the
deterministic composer improves; what must not change is that a question is
answered when it is answerable, clarified when it is genuinely open, refused
when it is off-topic, and that no figure escapes verification.

The number-verification tests are the important ones: they are the difference
between "the model usually behaves" and "an invented statistic cannot reach a
user".
"""

from __future__ import annotations

import pytest

from sa_insights.chat import answers, tools
from sa_insights.chat.agent import Agent, Session


# --------------------------------------------------------------------------- #
# Number verification — the core reliability guarantee
# --------------------------------------------------------------------------- #
@pytest.fixture()
def cpi_evidence() -> answers.Evidence:
    evidence = answers.Evidence()
    evidence.series["cpi_yoy"] = {
        "key": "cpi_yoy",
        "label": "CPI inflation (YoY)",
        "unit": "%",
        "frequency": "M",
        "latest": {"period": "2026-09-01", "value": 4.5},
        "year_on_year": {
            "absolute": 1.1,
            "percent": None,
            "from_period": "2025-09-01",
            "from_value": 3.4,
            "to_period": "2026-09-01",
            "to_value": 4.5,
        },
    }
    return evidence


def test_real_figures_pass(cpi_evidence: answers.Evidence) -> None:
    text = "CPI inflation is 4.5%, up 1.1 percentage points from 3.4% a year earlier."
    assert answers.verify_numbers(text, cpi_evidence) == []


def test_invented_figure_is_caught(cpi_evidence: answers.Evidence) -> None:
    text = "CPI inflation is 7.9%, a sharp acceleration."
    assert answers.verify_numbers(text, cpi_evidence) == [7.9]


def test_iso_dates_are_not_read_as_numbers(cpi_evidence: answers.Evidence) -> None:
    """A regression guard.

    "2026-09-01" was being tokenised as 2026, 09 and 01; the 09 was then reported
    as an invented figure, which rejected every correct answer that named a
    period. Dates are labels, not measurements.
    """
    text = "In 2026-09-01 the reading was 4.5%."
    assert answers.verify_numbers(text, cpi_evidence) == []


def test_year_ranges_are_not_read_as_numbers(cpi_evidence: answers.Evidence) -> None:
    text = "Between 2020 and 2026, CPI inflation reached 4.5%."
    assert answers.verify_numbers(text, cpi_evidence) == []


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
def test_gate_refuses_without_evidence() -> None:
    answer = answers.gate(
        question="who won the world cup",
        evidence=answers.Evidence(),
        graph_hits=[],
        model_answer=None,
    )
    assert answer.kind == "refuse"
    assert answer.confidence == 0.0


def test_gate_replaces_prose_containing_bad_numbers(cpi_evidence: answers.Evidence) -> None:
    answer = answers.gate(
        question="what is cpi",
        evidence=cpi_evidence,
        graph_hits=["cpi_yoy"],
        model_answer="CPI is 9.9%, soaring.",
    )
    # The bad prose must not survive, but the verified figures still reach the
    # user through citations.
    assert "9.9" not in answer.text
    assert answer.unverified_numbers == [9.9]
    assert answer.citations


def test_gate_returns_clean_answer_when_numbers_check_out(cpi_evidence: answers.Evidence) -> None:
    answer = answers.gate(
        question="what is cpi",
        evidence=cpi_evidence,
        graph_hits=["cpi_yoy"],
        model_answer="CPI inflation is 4.5%.",
    )
    assert answer.kind == "answer"
    assert answer.unverified_numbers == []
    assert 0.5 < answer.confidence <= 0.95


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def test_guard_rejects_writes() -> None:
    for sql in ("DELETE FROM observations", "UPDATE indicators SET label='x'", "DROP TABLE forecasts"):
        with pytest.raises(tools.ToolError):
            tools._guard(sql)


def test_guard_rejects_multiple_statements() -> None:
    with pytest.raises(tools.ToolError):
        tools._guard("SELECT 1; SELECT 2")


def test_guard_allows_select_and_cte() -> None:
    tools._guard("SELECT series_key FROM indicators")
    tools._guard("WITH x AS (SELECT 1) SELECT * FROM x")


def test_rate_series_omit_percent_change() -> None:
    """CPI 3.4% -> 4.5% is 1.1 points, not +32%.

    The percentage is arithmetically correct and completely misleading; a reader
    would conclude inflation tripled in level.
    """
    snapshot = tools.get_series("cpi_yoy", periods=13)
    if snapshot.since_start is not None:
        assert snapshot.since_start.percent is None


def test_level_series_keep_percent_change() -> None:
    snapshot = tools.get_series("usd_zar", periods=13)
    if snapshot.since_start is not None and snapshot.since_start.percent is not None:
        assert isinstance(snapshot.since_start.percent, float)


# --------------------------------------------------------------------------- #
# Agent policies
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def agent() -> Agent:
    return Agent()


@pytest.mark.parametrize(
    "question",
    [
        "What is CPI inflation now?",
        "How has the repo rate changed since 2020?",
        "what about the rand",
        "are interest rates high?",
    ],
)
def test_grounded_questions_are_answered(agent: Agent, question: str) -> None:
    answer = agent.ask(question, Session())
    assert answer.kind == "answer", f"{question!r} -> {answer.kind}: {answer.text}"
    assert answer.citations, f"{question!r} produced no citations"


@pytest.mark.parametrize(
    "question",
    [
        "Who won the 2010 world cup?",
        "tell me a joke",
        "what is the meaning of life",
        "what is the weather tomorrow",
    ],
)
def test_off_topic_questions_are_refused(agent: Agent, question: str) -> None:
    """Guard against the specific failure that shipped briefly.

    "who won the 2010 world cup" matched a GDP series because 2010 sits inside
    its data range, and the bot answered with growth figures.
    """
    answer = agent.ask(question, Session())
    assert answer.kind == "refuse", f"{question!r} was answered: {answer.text}"


def test_vague_theme_question_asks_for_clarification(agent: Agent) -> None:
    answer = agent.ask("How has the economy improved?", Session())
    assert answer.kind == "clarify"
    assert answer.follow_up


def test_named_series_with_vague_wording_is_answered(agent: Agent) -> None:
    """"is inflation getting better?" names its subject, so asking would annoy."""
    answer = agent.ask("is inflation getting better?", Session())
    assert answer.kind == "answer"


def test_follow_up_resolves_its_own_subject(agent: Agent) -> None:
    session = Session()
    agent.ask("What is CPI inflation now?", session)
    follow_up = agent.ask("And unemployment?", session)
    assert follow_up.kind == "answer"
    assert any("unemploy" in key for key in follow_up.evidence_series), follow_up.evidence_series


def test_answers_never_carry_unverified_numbers(agent: Agent) -> None:
    for question in ("What is CPI inflation now?", "what about the rand", "How has the repo rate changed since 2020?"):
        answer = agent.ask(question, Session())
        assert answer.unverified_numbers == [], f"{question!r} -> {answer.unverified_numbers}"


# --------------------------------------------------------------------------- #
# Retrieval regressions
# --------------------------------------------------------------------------- #
def test_a_named_series_survives_a_time_window(agent: Agent) -> None:
    """A reported bug, as a test.

    "what was unemployment like 10 years ago?" answered with the **long bond
    yield**, because bond_yield's description reads "Daily average yield on
    government bonds of 10 years and longer" — so the tokens "10" and "years"
    matched a series nobody asked about, and scored 54 to unemployment's 13.

    Time expressions and bare digits are now excluded from retrieval.
    """
    answer = agent.ask("what was unemployment like 10 years ago?", Session())
    assert answer.kind == "answer"
    assert any("unemployment" in key for key in answer.evidence_series), answer.evidence_series
    assert "bond yield" not in answer.text.lower()


def test_projection_does_not_outrank_observed_data(agent: Agent) -> None:
    """IMF series end in 2031, so leading with one states a forecast as fact."""
    answer = agent.ask("what is unemployment?", Session())
    assert answer.evidence_series, "no evidence"
    assert answer.evidence_series[0] == "unemployment", answer.evidence_series


def test_comparative_follow_up_keeps_both_subjects(agent: Agent) -> None:
    """"compare both" must compare, not silently answer about one series."""
    session = Session()
    agent.ask("What is CPI inflation now?", session)
    answer = agent.ask("and unemployment? compare both to today", session)
    joined = " ".join(answer.evidence_series)
    assert "unemployment" in joined, answer.evidence_series
    assert "cpi" in joined, f"previous subject dropped: {answer.evidence_series}"


def test_time_words_are_not_topics() -> None:
    """Guard the tokeniser directly, so the class of bug cannot return."""
    from sa_insights.chat.graph import _tokens

    tokens = _tokens("what was unemployment like 10 years ago")
    assert "unemployment" in tokens
    for noise in ("10", "years", "ago", "what", "was", "like"):
        assert noise not in tokens, f"{noise!r} should not be a retrieval token"


def test_pronoun_follow_up_borrows_the_previous_subject(agent: Agent) -> None:
    """A reported bug, as a test.

    "what do they confirm?" contains no vocabulary at all — every token is a
    pronoun or a verb — so grounding failed and the bot refused, despite the
    answer being on screen from the turn before. Resolving the reference from
    conversation state is what makes this conversational rather than keyword
    search.
    """
    session = Session()
    agent.ask("what were unemployment and inflation like 10 years ago?", session)
    answer = agent.ask("what do they confirm?", session)
    assert answer.kind == "answer", answer.text
    assert any("unemploy" in k for k in answer.evidence_series), answer.evidence_series


def test_named_subject_is_never_overridden_by_a_pronoun(agent: Agent) -> None:
    """The inverse of the bug above, and the reason the rule needs a guard.

    "and unemployment? compare both to today" names unemployment, so it must
    answer about unemployment. Treating the word "both" as a reference signal let
    the pronoun branch win and answered about the previous series instead.
    """
    session = Session()
    agent.ask("what is cpi now", session)
    answer = agent.ask("and unemployment? compare both to today", session)
    assert any("unemployment" in k for k in answer.evidence_series), answer.evidence_series
    assert answer.evidence_series[0].startswith(("unemployment", "weo_unemployment"))


def test_window_is_labelled_honestly_when_data_is_shorter(agent: Agent) -> None:
    """Never claim a comparison the data does not support.

    Unemployment starts in 2017, but asking about "10 years ago" produced "up
    5.36 pp since 2016" — a year the series does not cover. Arithmetically
    correct, unverifiable in practice.
    """
    answer = agent.ask("what was unemployment like 10 years ago?", Session())
    if "since 2017" in answer.text:
        assert "not 2016" in answer.text, answer.text
