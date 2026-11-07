#!/usr/bin/env python
"""Fetch every catalogued series into the local cache.

This is the entry point CI uses to refresh data before rebuilding the site. It
is deliberately separate from the exporter scripts so a failed fetch surfaces
as its own step rather than as a confusing downstream error.

Exit code is non-zero only if *nothing* could be loaded — individual series
failures are reported and tolerated, because a single unavailable upstream
indicator should not block a deployment.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sa_insights.catalog import SERIES  # noqa: E402
from sa_insights.sources import load_all  # noqa: E402


def main() -> int:
    def progress(spec, status, index, total):
        marker = {
            "live": "live",
            "cache": "cached",
            "stale": "STALE",
            "failed": "FAILED",
        }.get(status["state"], status["state"])
        print(f"[{index:2d}/{total}] {spec.key:22s} {marker:6s}", end="")
        if status.get("error"):
            print(f"  {status['error'][:90]}", end="")
        print()

    data, statuses = load_all(SERIES, force=True, progress=progress)

    failed = [s for s in statuses if s["state"] == "failed"]
    loaded = len(data)
    total_obs = sum(len(s) for s in data.values())

    print()
    print(f"series loaded : {loaded}/{len(SERIES)}")
    print(f"observations  : {total_obs:,}")

    if failed:
        print(f"failed        : {len(failed)}", file=sys.stderr)
        for item in failed:
            print(f"  - {item.get('label', item['key'])}: {item.get('error')}", file=sys.stderr)

    if loaded == 0:
        print("nothing loaded — refusing to continue", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
