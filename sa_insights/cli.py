"""Command-line entry points.

These exist so the pipeline is scriptable outside Streamlit and without the
scheduler — useful in CI, in a container one-off, and for debugging a single
stage.

    python -m sa_insights.cli collect
    python -m sa_insights.cli forecast
    python -m sa_insights.cli export --out dist
    python -m sa_insights.cli doctor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import downloads, refresh, store, tracking


def collect_main() -> int:
    data, _statuses, report = refresh.collect(force=True)
    print(json.dumps(report.as_dict(), indent=2, default=str))
    return 0 if report.series_loaded else 1


def forecast_main() -> int:
    written, failures = refresh.forecast_all(
        progress=lambda spec, i, n, result, err: print(
            f"[{i:2d}/{n}] {spec.key:22s} "
            + (f"{result.model} MASE={result.mase:.2f}" if result else f"FAILED {err}")
        )
    )
    print(f"\nforecasts written: {written} | failed: {len(failures)}")
    return 0 if written else 1


def export_main(out: Path | str = "dist") -> int:
    """Write every artefact to a directory, as a publish step would."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    (out / "sa-insights-powerbi.zip").write_bytes(downloads.powerbi_bundle())
    (out / "sa_insights.ossie.yaml").write_bytes(downloads.ossie_yaml())
    (out / "sa_insights.ossie-draft.yaml").write_bytes(
        downloads.ossie_yaml(downloads.DRAFT_VERSION)
    )
    (out / "observations.csv").write_bytes(downloads.observations_csv())
    (out / "forecasts.csv").write_bytes(downloads.forecasts_csv())

    for path in sorted(out.iterdir()):
        print(f"{path.name:34s} {path.stat().st_size:>9,} bytes")
    return 0


def doctor_main() -> int:
    """Report the state of every dependency so setup problems are obvious."""
    from .catalog import SERIES
    from .forecast import FM_ENABLED

    summary = store.data_summary()
    tstatus = tracking.status()

    print("sa-insights doctor")
    print(f"  database        : {store.DB_PATH}")
    print(f"  initialised     : {store.is_initialised()}")
    print(f"  series stored   : {summary['series']} / {len(SERIES)} in catalogue")
    print(f"  observations    : {summary['observations']:,}")
    print(f"  forecast runs   : {summary['forecasts']}")
    print(f"  last ingest     : {summary['last_ingest']}")
    print(f"  fm enabled      : {FM_ENABLED}")

    try:
        from . import chronos_fm

        info = chronos_fm.describe()
        print(f"  chronos         : {'ready' if info['available'] else info['reason']}")
    except Exception as exc:  # noqa: BLE001
        print(f"  chronos         : unavailable ({exc})")

    print(f"  mlflow tracking : {tstatus['enabled']} {tstatus['reason'] or ''}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["collect", "forecast", "export", "doctor", "refresh"]
    )
    parser.add_argument("--out", default="dist")
    args = parser.parse_args()

    if args.command == "collect":
        return collect_main()
    if args.command == "forecast":
        return forecast_main()
    if args.command == "export":
        return export_main(args.out)
    if args.command == "doctor":
        return doctor_main()
    if args.command == "refresh":
        report = refresh.refresh_all(force=True)
        print(json.dumps(report.as_dict(), indent=2, default=str))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
