"""Try a real generateContent call and report exactly what fails.

``scripts/probe_gemini.py`` proved the model exists, so a 404 on generateContent
is not a missing model. This distinguishes the remaining possibilities: API
version (v1beta vs v1), key format, or an auth header the endpoint rejects.

Usage:
    uv run modal run scripts/probe_gemini_call.py

Prints status codes and short error bodies only — never the key.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import modal

app = modal.App("gemini-call-probe")

VERSIONS = ("v1beta", "v1")
MODELS = ("gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-flash-lite-latest")


def _attempt(url: str, key: str, *, header: bool) -> dict:
    body = json.dumps(
        {
            "contents": [{"role": "user", "parts": [{"text": "Reply with the single word: ok"}]}],
            "generationConfig": {"maxOutputTokens": 16},
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    target = url
    if header:
        headers["x-goog-api-key"] = key
    else:
        target = f"{url}?key={key}"

    request = urllib.request.Request(target, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
        text = ""
        cands = payload.get("candidates") or []
        if cands:
            parts = (cands[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts)[:40]
        return {"status": 200, "text": text}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:180]
        return {"status": exc.code, "detail": detail}
    except Exception as exc:  # noqa: BLE001 - probe reports, never raises
        return {"status": None, "detail": f"{type(exc).__name__}: {exc}"[:180]}


@app.function(secrets=[modal.Secret.from_name("gemini-api-key")], timeout=120)
def probe() -> list[dict]:
    import os

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return [{"error": "no key in environment"}]

    results = []
    for version in VERSIONS:
        for model in MODELS:
            url = (
                f"https://generativelanguage.googleapis.com/{version}/models/"
                f"{model}:generateContent"
            )
            for use_header in (True, False):
                outcome = _attempt(url, key, header=use_header)
                results.append(
                    {
                        "version": version,
                        "model": model,
                        "auth": "header" if use_header else "query",
                        **outcome,
                    }
                )
    return results


@app.local_entrypoint()
def main() -> None:
    for row in probe.remote():
        print(json.dumps(row))
