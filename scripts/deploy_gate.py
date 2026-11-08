"""Deploy gate: refuse to ship a chatbot that fails its own regression suite.

Why a gate rather than a scheduled job
--------------------------------------
A nightly eval reports a regression twelve hours after it shipped. For a system
whose failures are "answered about the wrong series" — wrong but plausible — the
cost of that window is real: someone reads a confident number about the bond
yield when they asked about unemployment.

This runs the suite **before** the deploy and aborts on a bad score, so the
previous version keeps serving. The suite is cheap (seconds, no judge model), so
there is no reason to defer it.

The rule
--------
Fail on any of:

* pass rate below the threshold (default 0.9)
* any case where the answer names a series it must not (a wrong-subject answer
  is worse than a refusal — it is confidently wrong)
* any unverified number (the core guarantee, so never negotiable)

Everything logs to MLflow whether it passes or fails, because a blocked deploy is
exactly the run you want to inspect afterwards.

Usage:
    uv run python scripts/deploy_gate.py                 # gate a deploy
    uv run python scripts/deploy_gate.py --dry-run       # report, never block
    uv run python scripts/deploy_gate.py --threshold 1.0 # strict
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sa_insights.chat import llm  # noqa: E402
from sa_insights.chat.agent import Agent  # noqa: E402
from scripts.eval_chat import CASES, log_to_mlflow, run_case, summarise  # noqa: E402


def evaluate() -> tuple[list, dict]:
    agent = Agent(llm=llm.phrase) if llm.available() else Agent()
    results = [run_case(agent, case) for case in CASES]
    return results, summarise(results)


def blocking_problems(results: list, metrics: dict, threshold: float) -> list[str]:
    """The reasons this build must not ship. Empty means safe to deploy."""
    problems: list[str] = []

    pass_rate = metrics.get("pass_rate", 0.0)
    if pass_rate < threshold:
        failed = [r.case.question for r in results if not r.passed]
        problems.append(
            f"pass rate {pass_rate:.0%} is below the {threshold:.0%} threshold; "
            f"failing: {', '.join(failed[:4])}"
        )

    # A wrong-subject answer is not a near miss. It is a confident, citable
    # statement about an indicator the user never asked about, so it blocks even
    # when the overall pass rate is fine.
    wrong_subject = [
        r.case.question for r in results if not r.no_false_series
    ]
    if wrong_subject:
        problems.append(
            "answers named a forbidden series: " + ", ".join(wrong_subject[:4])
        )

    unverified = [r.case.question for r in results if not r.verified]
    if unverified:
        problems.append(
            "answers contained unverified numbers: " + ", ".join(unverified[:4])
        )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true", help="report but never block")
    args = parser.parse_args()

    phrasing = f"gemini {llm.DEFAULT_MODEL}" if llm.available() else "deterministic"
    print(f"deploy gate: evaluating {len(CASES)} cases (phrasing: {phrasing})")

    results, metrics = evaluate()

    for result in results:
        if not result.passed:
            print(f"  FAIL {result.case.question}")
            print(f"       bug: {result.case.bug}")
            if not result.kind_ok:
                print(f"       kind: expected {result.case.expect_kind}, got {result.kind}")
            if not result.mentions_ok:
                print(f"       missing series {result.case.must_mention}; got {result.evidence}")
            if not result.no_false_series:
                print(f"       mentioned forbidden {result.case.must_not_mention}")
            if not result.verified:
                print(f"       unverified numbers {result.unverified}")

    print(
        f"deploy gate: pass {metrics.get('pass_rate', 0):.0%} "
        f"| series {metrics.get('series_accuracy', 0):.0%} "
        f"| verified {metrics.get('verified_rate', 0):.0%} "
        f"| {metrics.get('avg_latency_ms', 0):.0f}ms avg"
    )

    # Log regardless of outcome: a blocked deploy is the run worth reading.
    run_id = log_to_mlflow(results, metrics)
    if run_id:
        print(f"deploy gate: mlflow run {run_id}")

    problems = blocking_problems(results, metrics, args.threshold)
    if not problems:
        print("deploy gate: PASSED")
        return 0

    for problem in problems:
        print(f"deploy gate: BLOCKED — {problem}", file=sys.stderr)

    if args.dry_run:
        print("deploy gate: --dry-run, not blocking")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
