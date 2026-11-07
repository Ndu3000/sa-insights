#!/usr/bin/env python
"""Report how stale each series is, and fail if a critical one has gone bad.

This exists so CI catches a silently broken upstream before it publishes a
dashboard full of year-old numbers. Thresholds are per-frequency because an
annual statistic is legitimately a year behind while a policy rate is not.

Usage:
    python scripts/check_freshness.py            # report, exit 1 on breach
    python scripts/check_freshness.py --warn     # report only, never fail
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from sa_insights.catalog import SERIES, short_label  # noqa: E402
from sa_insights.sources import read_cache  # noqa: E402

# Allowed staleness in days, by frequency. Generous enough for publication
# lag, tight enough that a frozen feed trips it.
MAX_LAG_DAYS = {"M": 150, "Q": 400, "A": 800}

# Indicators the dashboard leads with. A breach here is treated as more serious
# because these are the numbers a reader sees first.
CRITICAL = {"cpi_yoy", "policy_rate", "usd_zar", "lending_rate", "bond_yield"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warn", action="store_true", help="report without failing the build"
    )
    args = parser.parse_args()

    now = pd.Timestamp(datetime.now(timezone.utc).date())
    rows = []

    for spec in SERIES:
        series = read_cache(spec.key)
        if series is None or series.empty:
            rows.append((None, spec, "missing"))
            continue
        lag = (now - series.index[-1]).days
        rows.append((lag, spec, str(series.index[-1].date())))

    # Negative lag means the series extends into the future — that is an IMF
    # forecast, not staleness.
    breaches = []
    for lag, spec, last in rows:
        if lag is None:
            breaches.append((spec, "missing from cache", None))
            continue
        if lag < 0:
            continue
        limit = MAX_LAG_DAYS.get(spec.frequency, 800)
        if lag > limit:
            breaches.append((spec, f"{lag} days (limit {limit})", last))

    rows.sort(key=lambda r: (r[0] is None, -(r[0] or 0)))

    print(f"{'lag':>7}  {'series':24s} {'freq':4s} {'last':12s} status")
    print("-" * 68)
    for lag, spec, last in rows:
        if lag is None:
            print(f"{'--':>7}  {spec.key:24s} {spec.frequency:4s} {'--':12s} MISSING")
            continue
        if lag < 0:
            status = "forecast"
        else:
            limit = MAX_LAG_DAYS.get(spec.frequency, 800)
            status = "OK" if lag <= limit else f"STALE (limit {limit})"
        print(f"{lag:7d}  {spec.key:24s} {spec.frequency:4s} {last:12s} {status}")

    observed = [lag for lag, _, _ in rows if lag is not None and lag >= 0]
    print()
    if observed:
        print(f"median lag: {int(pd.Series(observed).median())} days")
    print(f"breaches  : {len(breaches)}")

    critical_breaches = [b for b in breaches if b[0].key in CRITICAL]
    if breaches:
        print()
        for spec, reason, _ in breaches:
            flag = "CRITICAL" if spec.key in CRITICAL else "warning "
            print(f"  {flag} {short_label(spec):24s} {reason}", file=sys.stderr)

    if args.warn:
        return 0

    # Only critical indicators fail the build. A stale annual series is a fact
    # of life, not a reason to block a deploy.
    return 1 if critical_breaches else 0


if __name__ == "__main__":
    raise SystemExit(main())
