"""List the Gemini models this key can call.

One-off diagnostic. Needed because the model name in ``llm.py`` is a guess until
confirmed against the live API — ``gemini-2.5-flash-lite`` returned 404, and a
404 from this endpoint means "no such model", not "bad key". Guessing again would
be how a working integration looks broken.

Usage:
    uv run modal run scripts/probe_gemini.py

Runs inside Modal so it can read the ``gemini-api-key`` secret; the key never
leaves the platform and is never printed.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import modal

app = modal.App("gemini-probe")


@app.function(secrets=[modal.Secret.from_name("gemini-api-key")], timeout=60)
def list_models() -> dict:
    import os

    key = os.environ.get("GEMINI_API_KEY", "")
    out: dict = {"key_present": bool(key), "key_length": len(key)}
    if not key:
        return out

    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models",
        headers={"x-goog-api-key": key},
    )
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        out["error"] = f"HTTP {exc.code}: {exc.reason}"
        return out
    except Exception as exc:  # noqa: BLE001 - a probe reports, never raises
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    models = payload.get("models", [])
    usable = [
        m["name"].split("/")[-1]
        for m in models
        if "generateContent" in (m.get("supportedGenerationMethods") or [])
    ]
    out["total"] = len(models)
    out["usable"] = len(usable)
    out["flash"] = sorted(n for n in usable if "flash" in n)
    out["sample"] = sorted(usable)[:15]
    return out


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(list_models.remote(), indent=2))
