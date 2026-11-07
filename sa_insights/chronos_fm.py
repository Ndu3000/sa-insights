"""Optional Apache Chronos-Bolt adapter — a time-series foundation model.

Chronos-Bolt is a T5 encoder-decoder pretrained on roughly 100 billion time
series observations. It forecasts zero-shot (no fitting on our data) and emits
quantiles directly, which maps cleanly onto this project's
``(point, lower, upper)`` fitter contract.

Why this model
--------------
* Zero-shot accuracy: Chronos-Bolt beats models that were *trained* on the
  benchmark datasets, and beats the original Chronos-Large while being about
  600x faster.
* Native quantiles: direct multi-step quantile output means we get prediction
  intervals without simulation, keeping CPU cost low.
* Size: ``bolt-small`` is 48M parameters (~191 MB fp32). ``bolt-base`` is 205M
  (~820 MB); the extra accuracy is not worth the CPU latency here. ``bolt-tiny``
  (9M) is too weak to be useful.

Design notes
------------
* Everything is **lazy**. Importing this module must never load torch, so the
  dashboard keeps starting instantly and works without the optional deps.
* Every entry point degrades gracefully: if torch or chronos-forecasting is
  missing, ``available()`` returns False and the classical model pool is used
  unchanged.
* Inference is single-threaded-friendly and capped to a context window that the
  model actually attends over, keeping CPU latency predictable.
"""

from __future__ import annotations

import numpy as np

MODEL_ID = "amazon/chronos-bolt-small"

# Chronos-Bolt patches its context internally. Beyond this many observations the
# marginal accuracy gain is negligible, so we cap for predictable latency.
MAX_CONTEXT = 512

# Cached singleton pipeline, plus the reason loading failed (if it did).
_PIPELINE = None
_LOAD_FAILED: str | None = None


def available() -> bool:
    """True when the optional inference dependencies can be imported."""
    return _load() is not None


def unavailable_reason() -> str | None:
    """Human-readable reason the model is unavailable, or None if it is ready."""
    _load()
    return _LOAD_FAILED


def _load():
    """Build the pipeline once. Returns None when unavailable."""
    global _PIPELINE, _LOAD_FAILED

    if _PIPELINE is not None:
        return _PIPELINE
    if _LOAD_FAILED is not None:
        return None

    try:
        import torch
        from chronos import BaseChronosPipeline
    except Exception as exc:  # noqa: BLE001 - optional dependency
        _LOAD_FAILED = f"missing dependency ({type(exc).__name__}: {exc})"
        return None

    try:
        # float32 rather than bfloat16: bfloat16 is emulated on older CPUs and
        # can be markedly slower than plain float32 in that situation.
        _PIPELINE = BaseChronosPipeline.from_pretrained(
            MODEL_ID,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    except Exception as exc:  # noqa: BLE001 - weights may be unavailable offline
        _LOAD_FAILED = f"could not load {MODEL_ID} ({type(exc).__name__}: {exc})"
        return None

    return _PIPELINE


def _to_numpy(obj) -> np.ndarray:
    """Detach a tensor (or leave an array/list alone) and convert to numpy."""
    if hasattr(obj, "cpu"):
        obj = obj.cpu()
    if hasattr(obj, "numpy"):
        obj = obj.numpy()
    return np.asarray(obj, dtype=float)


def _extract_bands(arr: np.ndarray, horizon: int, n_levels: int):
    """Pull (low, median, high) out of Chronos' quantile tensor.

    Chronos-Bolt returns shape ``(batch, horizon, n_quantiles)``, but older
    builds and some code paths give ``(batch, n_quantiles, horizon)``. Rather
    than assume, we locate the quantile axis by its length. This function
    raises on anything unexpected — it never pads, resizes or fabricates values,
    because a silently invented forecast is worse than a visible failure.
    """
    if arr.ndim == 3:
        arr = arr[0]  # drop the single-series batch axis

    if arr.ndim != 2:
        raise ValueError(f"unexpected quantile tensor shape {arr.shape}")

    if arr.shape[-1] == n_levels:
        pass  # already (horizon, quantiles)
    elif arr.shape[0] == n_levels:
        arr = arr.T
    else:
        raise ValueError(
            f"cannot locate quantile axis in shape {arr.shape} for {n_levels} levels"
        )

    if arr.shape[0] != horizon:
        raise ValueError(
            f"quantile tensor has {arr.shape[0]} steps but horizon is {horizon}"
        )

    return arr[:, 0], arr[:, 1], arr[:, 2]


# Chronos-Bolt was trained on discrete quantile levels from 0.10 to 0.90 only.
# Asking for anything narrower silently degrades to the nearest trained level,
# so we clamp explicitly and report the achievable confidence rather than
# pretending a 95% interval exists.
TRAINED_QUANTILE_MIN = 0.10
TRAINED_QUANTILE_MAX = 0.90

# Widest interval this model can honestly produce.
MAX_CONFIDENCE = TRAINED_QUANTILE_MAX - TRAINED_QUANTILE_MIN  # 0.80


def _quantile_levels(pipe, alpha: float) -> list[float]:
    """Quantile levels for the requested interval, clamped to the trained range.

    Floating point makes this subtler than it looks: ``1 - 0.8`` is
    ``0.19999999999999996``, so ``alpha / 2`` lands on ``0.09999999999999998``
    — fractionally below the trained minimum — and Chronos warns and snaps it.
    Rounding first, then clamping, keeps requests inside the supported range.
    """
    lower = round(alpha / 2.0, 6)
    upper = round(1.0 - alpha / 2.0, 6)
    lower = max(TRAINED_QUANTILE_MIN, min(lower, TRAINED_QUANTILE_MAX))
    upper = max(TRAINED_QUANTILE_MIN, min(upper, TRAINED_QUANTILE_MAX))
    if lower >= upper:
        lower, upper = TRAINED_QUANTILE_MIN, TRAINED_QUANTILE_MAX
    return [lower, 0.5, upper]


def effective_confidence(requested: float) -> float:
    """The confidence the model can actually deliver, given its trained range."""
    return min(requested, MAX_CONFIDENCE)


def make_fitter(model_id: str = MODEL_ID):
    """Return a fitter matching forecast.py's contract.

    Signature: ``fitter(y, h, alpha, m, bands=True) -> (point, lower, upper)``
    where ``alpha`` is the two-sided tail mass (so ``alpha=0.2`` is an 80%
    interval). ``m`` is the seasonal period and is unused here — Chronos infers
    seasonal structure from the pretrained prior rather than taking it as input.
    """

    def fitter(y: np.ndarray, h: int, alpha: float, m: int, bands: bool = True):
        import torch

        pipe = _load()
        if pipe is None:
            raise RuntimeError(f"Chronos-Bolt unavailable: {_LOAD_FAILED}")

        context = np.asarray(y, dtype=np.float32)
        context = context[np.isfinite(context)]
        if context.size < 4:
            raise ValueError("not enough finite history for the foundation model")
        context = context[-MAX_CONTEXT:]

        levels = _quantile_levels(pipe, alpha)
        tensor = torch.tensor(context, dtype=torch.float32)

        # Preferred path: ask for exactly the quantiles we need.
        try:
            result = pipe.predict_quantiles(
                inputs=tensor,
                prediction_length=h,
                quantile_levels=levels,
            )
            quantiles = result[0] if isinstance(result, tuple) else result
            low, point, high = _extract_bands(
                _to_numpy(quantiles), h, len(levels)
            )
        except Exception as exc:  # noqa: BLE001 - fall back to the full matrix
            if pipe is None:
                raise
            forecast = pipe.predict(tensor, prediction_length=h)
            arr = _to_numpy(
                forecast[0] if isinstance(forecast, (list, tuple)) else forecast
            )
            if arr.ndim == 3:
                arr = arr[0]

            cached_levels = getattr(pipe, "quantile_levels", None)
            if cached_levels is None:
                if arr.shape[-1] > 1 and arr.shape[0] == h:
                    cached_levels = np.linspace(0.01, 0.99, arr.shape[-1])
                else:
                    raise ValueError(
                        f"cannot infer quantile levels ({exc}); shape {arr.shape}"
                    ) from exc
            cached_levels = np.asarray(cached_levels, dtype=float)

            if arr.shape[-1] != cached_levels.size and arr.shape[0] == cached_levels.size:
                arr = arr.T

            def snap(target: float) -> int:
                return int(np.argmin(np.abs(cached_levels - target)))

            low = arr[:, snap(levels[0])]
            point = arr[:, snap(0.5)]
            high = arr[:, snap(levels[2])]

        point = np.asarray(point, dtype=float)
        if not bands:
            return point, None, None

        # Enforce a valid, ordered band that contains the point forecast.
        low = np.minimum(np.asarray(low, dtype=float), point)
        high = np.maximum(np.asarray(high, dtype=float), point)
        return point, low, high

    fitter.__name__ = f"chronos[{model_id}]"
    return fitter


def forecast_once(y: np.ndarray, horizon: int, alpha: float = 0.2):
    """Convenience wrapper for ad-hoc checks and scripts."""
    fitter = make_fitter()
    return fitter(np.asarray(y, dtype=float), horizon, alpha, 1, True)


def describe() -> dict:
    """Status record for diagnostics, surfacing problems without raising."""
    ready = available()
    return {
        "model_id": MODEL_ID,
        "available": ready,
        "reason": None if ready else _LOAD_FAILED,
        "max_context": MAX_CONTEXT,
    }
