#!/usr/bin/env python
"""Regenerate the Apache Ossie semantic model from the collected data.

Usage:
    python scripts/build_semantic_model.py            # write semantic/*
    python scripts/build_semantic_model.py --check     # validate only

The model is written in two document shapes:

* ``sa_insights.ossie.*``       - version 0.1.1, the released shape that wraps
  the model in a ``semantic_model`` array.
* ``sa_insights.ossie-draft.*`` - version 0.2.0.dev0, the flat draft shape that
  the official Apache Ossie JSON Schema validates.

Re-run this whenever the series catalogue or cached data changes, then commit
the regenerated files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sa_insights.semantic import (  # noqa: E402
    DRAFT_VERSION,
    OUT_DIR,
    OSSIE_VERSION,
    SCHEMA_PATH,
    build_model,
    validate,
    validate_schema,
    write_model,
)


def _summary(document: dict, label: str, problems: list[str]) -> bool:
    model = document["semantic_model"][0] if "semantic_model" in document else document
    datasets = model["datasets"]
    facts = [d for d in datasets if d["name"].startswith("fct_")]
    dims = [d for d in datasets if d["name"].startswith("dim_")]
    metrics = len(model.get("metrics", [])) + sum(
        len(d.get("metrics", [])) for d in datasets
    )

    print(f"\n{label}  (version {document['version']})")
    print(f"  datasets      : {len(datasets)} ({len(facts)} fact, {len(dims)} dimension)")
    print(f"  relationships : {len(model.get('relationships', []))}")
    print(f"  metrics       : {metrics}")
    if problems:
        print(f"  problems      : {len(problems)}")
        for problem in problems[:12]:
            print(f"    - {problem}", file=sys.stderr)
        if len(problems) > 12:
            print(f"    ... and {len(problems) - 12} more", file=sys.stderr)
        return False
    print("  validation    : OK")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="validate without writing files"
    )
    args = parser.parse_args()

    ok = True

    released = build_model(version=OSSIE_VERSION)
    ok &= _summary(released, "Released shape", validate(released))

    draft = build_model(version=DRAFT_VERSION)
    ok &= _summary(draft, "Draft shape", validate_schema(draft, SCHEMA_PATH))

    if not SCHEMA_PATH.exists():
        print(
            f"\nnote: {SCHEMA_PATH.name} not found - official schema validation skipped.\n"
            "      fetch it with:\n"
            "      curl -sLo semantic/ossie-schema.json "
            "https://raw.githubusercontent.com/apache/ossie/main/core-spec/ossie-schema.json",
            file=sys.stderr,
        )

    if not ok:
        print("\nValidation failed.", file=sys.stderr)
        return 1

    if not args.check:
        for path in write_model(OUT_DIR):
            print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
