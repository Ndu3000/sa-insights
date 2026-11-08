"""Evaluate the chatbot and log results to MLflow.

Why evaluate at all
-------------------
Every bug fixed so far was found by a human reading a screenshot — "unemployment
returned the bond yield", "10 years ago" silently became 5. The same class of
regression will return the next time retrieval changes, and no unit test catches
it because the failure is *which series got chosen*, not an exception.

This turns each reported bug into a scored, repeatable case. The dataset is
deliberately the real failures rather than synthetic ones: a suite that only
tests questions the system already gets right measures nothing.

Why not ``mlflow.genai.evaluate`` with the built-in LLM scorers
---------------------------------------------------------------
MLflow 3 ships ``Correctness``, ``RetrievalGroundedness`` and friends, but they
call a judge model per row. For this system that is both expensive and the wrong
instrument: the guarantees here are *deterministic and already checked in code* —
``answers.verify_numbers`` proves every figure came from a tool, and the expected
series is known exactly. A judge model would be slower, cost money per run, and
be less reliable than an assertion.

So the scorers below are deterministic, and MLflow is used for what it is
genuinely good at: storing runs, comparing them over time, and giving each
regression a name and a history. ``mlflow.genai`` is the right tool for
open-ended prose quality later; it is the wrong tool for "did it pick the right
series".

Usage:
    uv run python scripts/eval_chat.py                  # run and log
    uv run python scripts/eval_chat.py --no-log          # local only

Requires the warehouse (SA_DB_PATH) and, for the phrasing-under-test path, a
GEMINI_API_KEY. Without a key it evaluates the deterministic composer, which is
the fallback users get — still meaningful.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sa_insights.chat import llm  # noqa: E402
from sa_insights.chat.agent import Agent, Session  # noqa: E402


@dataclass
class Case:
    """One evaluated question, with the behaviour it must exhibit.

    ``must_mention`` / ``must_not_mention`` are series keys. Naming the series is
    the real assertion: a fluent answer about the wrong indicator is the failure
    mode this suite exists for.
    """

    question: str
    expect_kind: str  # answer | clarify | refuse
    must_mention: list[str] = field(default_factory=list)
    must_not_mention: list[str] = field(default_factory=list)
    needs_context: bool = False  # run as a follow-up to `context_question`
    context_question: str | None = None
    bug: str = ""  # the regression this case locks down


# --------------------------------------------------------------------------- #
# The suite: every case is a bug that actually shipped
# --------------------------------------------------------------------------- #
CASES: list[Case] = [
    Case(
        question="What is CPI inflation now?",
        expect_kind="answer",
        must_mention=["cpi_yoy"],
        bug="baseline",
    ),
    Case(
        question="how has the repo rate changed since 2020?",
        expect_kind="answer",
        must_mention=["policy_rate"],
        bug="baseline",
    ),
    Case(
        question="what was unemployment like 10 years ago?",
        expect_kind="answer",
        must_mention=["unemployment"],
        must_not_mention=["bond_yield"],
        bug="'10 years' matched bond_yield's description ('of 10 years and longer')",
    ),
    Case(
        question="what were unemployment and inflation like 10 years ago?",
        expect_kind="answer",
        must_mention=["unemployment"],
        must_not_mention=["bond_yield"],
        bug="window question answered with a series nobody named",
    ),
    Case(
        question="How has the economy improved?",
        expect_kind="clarify",
        bug="vague theme must ask rather than dump figures",
    ),
    Case(
        question="Who won the 2010 world cup?",
        expect_kind="refuse",
        bug="off-topic matched a series via the year 2010 in its data range",
    ),
    Case(
        question="what is the meaning of life",
        expect_kind="refuse",
        bug="headline fallback fired on an ungrounded question",
    ),
    Case(
        question="what do they confirm?",
        expect_kind="answer",
        must_mention=["unemployment"],
        needs_context=True,
        context_question="what were unemployment and inflation like 10 years ago?",
        bug="pronoun-only follow-up refused despite the answer being on screen",
    ),
    Case(
        question="and unemployment? compare both to today",
        expect_kind="answer",
        must_mention=["unemployment"],
        needs_context=True,
        context_question="what is cpi now",
        bug="comparative follow-up answered about one series only",
    ),
    Case(
        question="what about the rand",
        expect_kind="answer",
        must_mention=["usd_zar"],
        bug="baseline",
    ),
]


@dataclass
class Result:
    case: Case
    kind: str
    text: str
    evidence: list[str]
    latency_ms: float
    unverified: list[float]

    @property
    def kind_ok(self) -> bool:
        return self.kind == self.case.expect_kind

    @property
    def mentions_ok(self) -> bool:
        joined = " ".join(self.evidence).lower() + " " + self.text.lower()
        for key in self.case.must_mention:
            if key.lower() not in joined:
                return False
        return True

    @property
    def no_false_series(self) -> bool:
        joined = " ".join(self.evidence).lower() + " " + self.text.lower()
        return all(key.lower() not in joined for key in self.case.must_not_mention)

    @property
    def verified(self) -> bool:
        return not self.unverified

    @property
    def passed(self) -> bool:
        return self.kind_ok and self.mentions_ok and self.no_false_series and self.verified


def run_case(agent: Agent, case: Case) -> Result:
    session = Session()
    if case.needs_context and case.context_question:
        agent.ask(case.context_question, session)

    started = time.perf_counter()
    answer = agent.ask(case.question, session)
    elapsed = (time.perf_counter() - started) * 1000

    return Result(
        case=case,
        kind=answer.kind,
        text=answer.text,
        evidence=list(answer.evidence_series),
        latency_ms=elapsed,
        unverified=answer.unverified_numbers,
    )


def summarise(results: list[Result]) -> dict[str, float]:
    total = len(results)
    if not total:
        return {}
    return {
        "pass_rate": sum(r.passed for r in results) / total,
        "kind_accuracy": sum(r.kind_ok for r in results) / total,
        "series_accuracy": sum(r.mentions_ok for r in results) / total,
        "no_false_series_rate": sum(r.no_false_series for r in results) / total,
        "verified_rate": sum(r.verified for r in results) / total,
        "avg_latency_ms": sum(r.latency_ms for r in results) / total,
    }


def log_to_mlflow(results: list[Result], metrics: dict[str, float]) -> str | None:
    """Log the run. Returns the MLflow run id, or None if logging was skipped.

    Never raises: an evaluation must still produce its console report when
    tracking is unavailable, otherwise the tooling becomes the thing that fails.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        return None

    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "sa-insights-chat-eval"))
        with mlflow.start_run(run_name=f"chat-eval-{time.strftime('%Y%m%d-%H%M')}") as run:
            mlflow.log_params(
                {
                    "cases": len(results),
                    "model": llm.DEFAULT_MODEL if llm.available() else "deterministic",
                    "judge": "none (deterministic scorers)",
                }
            )
            mlflow.log_metrics(metrics)

            # One row per case so a failure is inspectable, not just a count.
            import json

            rows = [
                {
                    "question": r.case.question,
                    "bug": r.case.bug,
                    "passed": r.passed,
                    "expected_kind": r.case.expect_kind,
                    "actual_kind": r.kind,
                    "evidence": r.evidence,
                    "unverified": r.unverified,
                    "latency_ms": round(r.latency_ms, 1),
                    "text": r.text[:400],
                }
                for r in results
            ]
            path = Path("chat_eval_results.json")
            path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(path))
            path.unlink(missing_ok=True)
            return run.info.run_id
    except Exception as exc:  # noqa: BLE001 - tooling must not fail the eval
        print(f"mlflow: logging skipped ({exc})")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-log", action="store_true", help="skip MLflow logging")
    args = parser.parse_args()

    agent = Agent(llm=llm.phrase) if llm.available() else Agent()
    print(f"phrasing: {'gemini ' + llm.DEFAULT_MODEL if llm.available() else 'deterministic'}")

    results = [run_case(agent, case) for case in CASES]
    metrics = summarise(results)

    print(f"\n{'=' * 78}\n{'result':6} {'kind':9} {'question':50} bug")
    print("=" * 78)
    for r in results:
        flag = "PASS" if r.passed else "FAIL"
        print(f"{flag:6} {r.kind:9} {r.case.question[:50]:50} {r.case.bug[:40]}")
        if not r.passed:
            if not r.kind_ok:
                print(f"       expected kind={r.case.expect_kind}, got {r.kind}")
            if not r.mentions_ok:
                print(f"       missing {r.case.must_mention}; evidence={r.evidence}")
            if not r.no_false_series:
                print(f"       should not mention {r.case.must_not_mention}")
            if not r.verified:
                print(f"       unverified numbers {r.unverified}")

    print("=" * 78)
    for key, value in metrics.items():
        print(f"  {key:24} {value:.3f}" if value <= 1 else f"  {key:24} {value:.0f}")

    if not args.no_log:
        run_id = log_to_mlflow(results, metrics)
        if run_id:
            print(f"\nmlflow run: {run_id}")

    return 0 if metrics.get("pass_rate", 0) >= 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
