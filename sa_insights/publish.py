#!/usr/bin/env python
"""Publish the static site from the running container's warehouse.

Why this exists
---------------
GitHub Actions is blocked on this account ("your account is locked due to a
billing issue"), so the scheduled workflow cannot run. The Docker worker already
refreshes data on a cron, so this script makes that same worker responsible for
publishing to GitHub Pages — using branch-based Pages, which is already enabled.

Shape
-----
1. Take a consistent snapshot of the live DuckDB warehouse into a temp copy.
   DuckDB is single-writer and the worker is actively using it, so we copy the
   file rather than reading it in place. This avoids a lock clash mid-publish
   and gives us a point-in-time view.
2. Point the export tools at that snapshot and render every artefact.
3. Commit the result to ``docs/`` and push, if anything changed.

Environment
-----------
    SA_PUBLISH_REPO    path to a git clone whose docs/ should be updated
    SA_PUBLISH_BRANCH  branch to push to (default: main)
    GIT_AUTHOR_NAME / GIT_AUTHOR_EMAIL   commit identity
    SA_PUBLISH_DRY_RUN "1" to render without committing

Usage:
    python -m sa_insights.publish
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import store

REPO = Path(os.getenv("SA_PUBLISH_REPO", "/publish"))
BRANCH = os.getenv("SA_PUBLISH_BRANCH", "main")
DRY_RUN = os.getenv("SA_PUBLISH_DRY_RUN", "").lower() in ("1", "true", "yes")

AUTHOR_NAME = os.getenv("GIT_AUTHOR_NAME", "sa-insights-bot")
AUTHOR_EMAIL = os.getenv("GIT_AUTHOR_EMAIL", "bot@sa-insights.local")

# A container cannot answer an interactive credential prompt, so the push must
# authenticate with a token embedded in the remote URL. Read from the env only —
# never written to .git/config, which would persist it on the mounted volume.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()


def _push_url(repo: Path) -> str | None:
    """Return an authenticated https URL for origin, or None if unavailable.

    Derives the URL from the existing remote so the repo's own configuration
    stays authoritative about which repository to push to.
    """
    if not GITHUB_TOKEN:
        return None

    remote = _run(["git", "remote", "get-url", "origin"], cwd=repo)
    if remote.returncode != 0:
        return None
    url = remote.stdout.strip()
    if not url.startswith("https://"):
        return None

    scheme, _, rest = url.partition("://")
    # Strip any credentials already present so we do not compound them.
    host_path = rest.split("@", 1)[-1]
    return f"{scheme}://x-access-token:{GITHUB_TOKEN}@{host_path}"


def _run(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a command and return the completed process, without raising."""
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )


def snapshot_warehouse() -> Path:
    """Copy the live database so exports read a stable point-in-time view.

    Reading the live file directly would race the worker: DuckDB permits one
    writer process, and a concurrent connection raises IOException. Copying is
    cheap (a few MB) and removes the race entirely.
    """
    if not store.DB_PATH.exists():
        raise RuntimeError(f"no warehouse at {store.DB_PATH} — run a refresh first")

    target_dir = Path(tempfile.mkdtemp(prefix="sa-publish-"))
    target = target_dir / "sa_insights.duckdb"

    # DuckDB may have a WAL alongside the main file; copy both so the snapshot
    # reflects committed-but-not-checkpointed writes.
    shutil.copy2(store.DB_PATH, target)
    wal = store.DB_PATH.with_suffix(".duckdb.wal")
    if wal.exists():
        shutil.copy2(wal, target.with_suffix(".duckdb.wal"))

    print(f"snapshot: {store.DB_PATH} -> {target} ({target.stat().st_size:,} bytes)")
    return target


def render(snapshot: Path, repo: Path) -> list[Path]:
    """Render every artefact into the checkout's docs/ directory.

    Runs the existing export scripts as subprocesses with env overrides rather
    than importing them, so each keeps its own CLI contract and a failure in one
    is reported clearly instead of half-executing the others.
    """
    docs = repo / "docs"
    semantic = repo / "semantic"
    env = {
        **os.environ,
        "SA_DB_PATH": str(snapshot),
        "SA_DOCS_OUT": str(docs),
        "SA_FORECASTS_OUT": str(docs / "data" / "forecasts.json"),
        # Without this the semantic model would be written into the image's own
        # (read-only for uid 1000) tree instead of the checkout being published.
        "SA_SEMANTIC_OUT": str(semantic),
        # Reuse the worker's stored runs; never refit during a publish.
        "SA_INSIGHTS_FM": os.getenv("SA_INSIGHTS_FM", "1"),
    }

    steps = [
        ("forecasts", ["scripts/export_forecasts.py"]),
        ("semantic", ["scripts/build_semantic_model.py"]),
        ("powerbi", ["scripts/export_powerbi.py"]),
        ("pages", ["scripts/export_pages.py"]),
    ]

    produced: list[Path] = []
    for name, script in steps:
        # sys.executable, not "python": a bare "python" resolves to whatever is
        # first on PATH, which is a different interpreter from the one running
        # this process (and lacks the project's dependencies).
        cmd = [sys.executable, *script]
        print(f"render: {name}")
        result = subprocess.run(cmd, cwd=str(repo), env=env, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout[-2000:], file=sys.stderr)
            print(result.stderr[-2000:], file=sys.stderr)
            raise RuntimeError(f"{name} export failed (exit {result.returncode})")
        if result.stdout.strip():
            print("  " + result.stdout.strip().replace("\n", "\n  ")[:800])
        produced.append(docs)

    return produced


def commit_and_push(repo: Path, semantic_out: Path) -> bool:
    """Commit changed artefacts and push. Returns True when something changed."""
    # A bind-mounted checkout is owned by the host user, not the container's uid
    # 1000, so git refuses to operate ("dubious ownership"). Marking the one
    # directory safe is scoped and explicit, unlike disabling the check globally
    # for every repo the container might ever see.
    _run(["git", "config", "--global", "--add", "safe.directory", str(repo)])

    add = _run(["git", "add", "-A", "docs", "semantic"], cwd=repo)
    if add.returncode != 0:
        raise RuntimeError(f"git add failed: {add.stderr.strip()}")

    staged = _run(["git", "diff", "--cached", "--quiet"], cwd=repo)
    # `diff --quiet` exits 0 when nothing is staged, 1 when there are changes.
    if staged.returncode == 0:
        print("no changes to publish")
        return False

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    message = f"Refresh published data ({stamp})"
    commit = _run(
        [
            "git",
            "-c",
            f"user.name={AUTHOR_NAME}",
            "-c",
            f"user.email={AUTHOR_EMAIL}",
            "commit",
            "-m",
            message,
        ],
        cwd=repo,
    )
    if commit.returncode != 0:
        raise RuntimeError(f"git commit failed: {commit.stderr.strip()}")
    print(f"committed: {message}")

    if DRY_RUN:
        print("dry run: skipping push")
        return True

    # Push through the authenticated URL rather than relying on a credential
    # helper, which does not exist in a slim container.
    url = _push_url(repo)
    if url is None:
        print(
            "warning: GITHUB_TOKEN not set or origin is not https; relying on "
            "whatever credentials the checkout already has",
            file=sys.stderr,
        )
        push = _run(["git", "push", "origin", BRANCH], cwd=repo)
    else:
        push = _run(["git", "push", url, f"HEAD:{BRANCH}"], cwd=repo)

    if push.returncode != 0:
        # Never echo the URL: it contains the token.
        raise RuntimeError(f"git push failed: {push.stderr.strip()}")
    print(f"pushed to origin/{BRANCH}")
    return True


def publish() -> bool:
    """Full publish cycle. Returns True when the site changed."""
    if not REPO.exists():
        raise RuntimeError(
            f"publish repo not found at {REPO} — mount a checkout there, or set "
            "SA_PUBLISH_REPO"
        )

    snapshot = snapshot_warehouse()
    try:
        render(snapshot, REPO)
        return commit_and_push(REPO, REPO / "semantic")
    finally:
        shutil.rmtree(snapshot.parent, ignore_errors=True)


def main() -> int:
    try:
        publish()
    except Exception as exc:  # noqa: BLE001 - surface clearly, exit non-zero
        print(f"publish failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
