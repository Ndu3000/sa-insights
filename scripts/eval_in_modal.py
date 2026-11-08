"""Run the chat evaluation inside Modal, with the Gemini key.

Needed because the phrasing path is where the regressions appeared: the
deterministic composer gets every case right, while the model has refused
questions whose facts were complete. Evaluating only the deterministic path would
measure the wrong thing.

    uv run modal run scripts/eval_in_modal.py

Prints the same report as ``scripts/eval_chat.py``. The key stays on the platform.
"""

from __future__ import annotations

import subprocess

import modal

app = modal.App("chat-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands("git clone --branch main https://github.com/Ndu3000/sa-insights.git /repo")
    .workdir("/repo")
    .run_commands(
        "pip install --quiet uv",
        "UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy "
        "uv sync --frozen --no-install-project --no-dev --extra fm --extra chat",
    )
    .env({"PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"})
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("gemini-api-key")],
    volumes={"/data": modal.Volume.from_name("sa-insights-data")},
    timeout=1800,
)
def run_eval() -> dict:
    import os

    os.environ["DATA_DIR"] = "/data"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

    result = subprocess.run(
        ["python", "scripts/eval_chat.py", "--no-log"],
        cwd="/repo",
        capture_output=True,
        text=True,
    )
    return {
        "rc": result.returncode,
        "stdout": result.stdout[-5000:],
        "stderr": result.stderr[-2000:],
    }


@app.local_entrypoint()
def main() -> None:
    outcome = run_eval.remote()
    print(outcome["stdout"])
    if outcome["rc"] != 0:
        print("STDERR:", outcome["stderr"])
