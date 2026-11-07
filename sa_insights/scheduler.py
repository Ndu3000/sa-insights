"""Background scheduler used by the ``worker`` container.

APScheduler rather than cron because the schedule lives in version control with
the rest of the code, and a failed run surfaces in the same logs as everything
else. Runs are serialised: a long foundation-model refresh must not overlap with
the next tick.

Usage:
    python -m sa_insights.scheduler                 # run forever
    python -m sa_insights.scheduler --once          # single pass, then exit
    python -m sa_insights.scheduler --cron "0 6 * * *"
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

from . import refresh, store, tracking

LOG = logging.getLogger("sa-insights.scheduler")

DEFAULT_CRON = os.getenv("REFRESH_CRON", "15 5 * * *")  # 05:15 UTC daily


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        stream=sys.stdout,
    )


def run_once(force: bool = True) -> refresh.RefreshReport:
    LOG.info("refresh starting (tracking=%s)", tracking.status()["enabled"])
    report = refresh.refresh_all(force=force)

    LOG.info(
        "refresh done in %.1fs | series %d/%d | observations %d | forecasts %d",
        report.duration_seconds,
        report.series_loaded,
        report.series_total,
        report.observations,
        report.forecasts_written,
    )
    if report.failed:
        LOG.warning("%d series failed to load", len(report.failed))
        for item in report.failed:
            LOG.warning("  %s: %s", item.get("key"), item.get("error"))
    if report.forecasts_failed:
        LOG.warning("%d series failed to forecast", len(report.forecasts_failed))
        for item in report.forecasts_failed:
            LOG.warning("  %s: %s", item.get("key"), item.get("error"))

    summary = store.data_summary()
    LOG.info(
        "store: %d series, %d observations, %d forecast runs",
        summary["series"],
        summary["observations"],
        summary["forecasts"],
    )

    _publish()

    return report


def _publish() -> None:
    """Publish the static site after a successful refresh.

    GitHub Actions is unavailable on this account, so the worker is the only
    thing that runs on a schedule. Publishing here is what makes the public
    Pages site actually refresh daily.

    Gated on SA_PUBLISH_REPO so local runs never touch a git remote: with no
    checkout mounted, this is a cheap no-op.
    """
    from . import publish

    if not os.getenv("SA_PUBLISH_REPO"):
        LOG.info("publish: skipped (SA_PUBLISH_REPO not set)")
        return

    LOG.info("publish: rendering static site")
    try:
        changed = publish.publish()
    except Exception as exc:  # noqa: BLE001 - a publish failure must not kill the worker
        LOG.error("publish failed: %s", exc)
        return

    LOG.info("publish: %s", "site updated and pushed" if changed else "no changes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run one refresh and exit")
    parser.add_argument("--cron", default=DEFAULT_CRON, help="cron expression")
    parser.add_argument(
        "--no-force", action="store_true", help="allow the local cache to be used"
    )
    args = parser.parse_args()

    configure_logging()

    if args.once:
        run_once(force=not args.no_force)
        return 0

    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler(timezone="UTC")
    trigger = CronTrigger.from_crontab(args.cron, timezone="UTC")
    scheduler.add_job(
        run_once,
        trigger=trigger,
        id="refresh",
        name="sa-insights refresh",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
        kwargs={"force": not args.no_force},
    )

    LOG.info("scheduler started; cron=%r (UTC)", args.cron)

    # Bootstrap immediately if the store is empty, so a fresh deployment does not
    # wait until tomorrow's tick to show data.
    try:
        if not store.is_initialised():
            LOG.info("store empty — performing initial refresh")
            run_once(force=True)
    except Exception:  # noqa: BLE001 - a failed bootstrap must not kill the scheduler
        LOG.exception("initial refresh failed; will retry on schedule")

    def _shutdown(signum, _frame):
        LOG.info("signal %s received; shutting down", signum)
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        LOG.info("scheduler stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
