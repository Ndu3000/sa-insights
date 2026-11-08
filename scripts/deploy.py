"""Deploy the Modal app, but only if the chatbot passes its regression suite.

One command instead of two, because a gate that has to be remembered is a gate
that gets skipped on the day it matters. The failure mode this guards against is
specific and has already happened twice: a retrieval change that looks harmless
and silently starts answering about the wrong series.

    uv run python scripts/deploy.py              # gate, then deploy
    uv run python scripts/deploy.py --skip-gate  # explicit override, logged

The override exists because a broken *evaluator* must not become a reason work
cannot ship. It is deliberately spelled out rather than a default, and it prints
a warning so the decision is visible in the terminal record.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(label: str, command: list[str]) -> int:
    print(f"\n=== {label} ===")
    return subprocess.run(command, cwd=ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="deploy without running the chatbot regression suite",
    )
    args = parser.parse_args()

    if args.skip_gate:
        print("WARNING: skipping the regression gate; deploying untested chat behaviour")
    else:
        if run("regression gate", ["uv", "run", "python", "scripts/deploy_gate.py"]) != 0:
            print(
                "\nDeploy aborted: the chatbot failed its regression suite.\n"
                "Fix the failure, or pass --skip-gate to override deliberately.",
                file=sys.stderr,
            )
            return 1

    code = run("modal deploy", ["uv", "run", "modal", "deploy", "modal_app.py"])
    if code == 0:
        print("\nDeployed. Verify with:")
        print("  curl -s https://ndu3000--sa-insights-chat-api.us-east.modal.run/health")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
