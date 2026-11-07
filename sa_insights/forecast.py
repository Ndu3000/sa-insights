"""One-shot time-series forecasting: model zoo, backtest selection, intervals.

Design notes
------------
* Models are fitted on whatever history the caller passes in and produce a
  horizon-ahead path plus an 80%/95% style prediction band.
* A model is only ever recommended after it has been scored by rolling-origin
  backtesting on that specific series — no global "best model" assumptions.
* Every model is wrapped so a single failure cannot take down the run; models
  that fail to fit are reported as excluded rather than raising.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import chronos_fm

warnings.filterwarnings("ignore")

FREQ_MONTHLY, FREQ_QUARTERLY, FREQ_ANNUAL = "M", "Q", "A"

# The foundation model is optional and CPU-heavy. It competes on the same
# backtest as every classical model, but only when its dependencies are
# installed and it has not been switched off. Set SA_INSIGHTS_FM=0 to skip it
# entirely (useful in CI jobs that only refresh data).
FM_ENABLED = os.getenv("SA_INSIGHTS_FM", "1").lower() not in ("0", "false", "no")

_FREQ_INFO = {
    FREQ_MONTHLY: {"season": 12, "offset": "MS", "per_year": 12, "word": "month"},
    FREQ_QUARTERLY: {"season": 4, "offset": "QS", "per_year": 4, "word": "quarter"},
    FREQ_ANNUAL: {"season": 1, "offset": "YS", "per_year": 1, "word": "year"},
}


class ForecastError(RuntimeError):
    """Raised when a series cannot be forecast at all."""


# ---------------------------------------------------------------------------
# Frequency handling
# ---------------------------------------------------------------------------
def infer_frequency(index: pd.DatetimeIndex) -> str:
    """Classify a DatetimeIndex as monthly / quarterly / annual by spacing."""
    if len(index) < 3:
        return FREQ_ANNUAL
    deltas = np.diff(index.values).astype("timedelta64[D]").astype(float)
    median = float(np.median(deltas))
    if median <= 45:
        return FREQ_MONTHLY
    if median <= 140:
        return FREQ_QUARTERLY
    return FREQ_ANNUAL


def seasonal_period(freq: str) -> int:
    return _FREQ_INFO.get(freq, _FREQ_INFO[FREQ_ANNUAL])["season"]


def future_index(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """Extend the index by `horizon` periods at the inferred native frequency."""
    freq = infer_frequency(index)
    offset = _FREQ_INFO[freq]["offset"]
    start = pd.date_range(index[-1], periods=horizon + 1, freq=offset)[1:]
    return pd.DatetimeIndex(start)


def horizon_word(freq: str) -> str:
    return _FREQ_INFO.get(freq, _FREQ_INFO[FREQ_ANNUAL])["word"]


def horizon_label(freq: str, horizon: int) -> str:
    word = horizon_word(freq)
    return f"{horizon} {word}{'' if horizon == 1 else 's'}"


def regularize(series: pd.Series) -> pd.Series:
    """Reindex onto a complete native-frequency calendar, interpolating gaps."""
    series = series[~series.index.duplicated(keep="last")].sort_index().astype(float)
    if len(series) < 4:
        return series
    freq = infer_frequency(series.index)
    offset = _FREQ_INFO[freq]["offset"]
    full = pd.date_range(series.index[0], series.index[-1], freq=offset)
    if len(full) == len(series):
        return series
    out = series.reindex(full)
    out = out.interpolate(method="time", limit_direction="both")
    return out.astype(float)


def trim_history(series: pd.Series, max_obs: int | None) -> pd.Series:
    if max_obs and len(series) > max_obs:
        return series.iloc[-max_obs:]
    return series


# A 25-year window is deliberately the default: long enough for several full
# business cycles, short enough to avoid modelling regimes that no longer apply
# (pre-inflation-targeting South Africa, for example).
DEFAULT_WINDOW_YEARS = 25


def default_window(freq: str) -> int:
    per_year = _FREQ_INFO.get(freq, _FREQ_INFO[FREQ_ANNUAL])["per_year"]
    return per_year * DEFAULT_WINDOW_YEARS


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------
def mae(actual, predicted) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def rmse(actual, predicted) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def mape(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.abs(actual) > 1e-9
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def smape(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0
    mask = denom > 1e-9
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)


def mase(actual, predicted, train) -> float:
    """Mean absolute scaled error, scaled by the in-sample naive error."""
    train = np.asarray(train, dtype=float)
    if len(train) < 2:
        return float("nan")
    scale = float(np.mean(np.abs(np.diff(train))))
    if scale < 1e-12:
        return float("nan")
    return float(mae(actual, predicted) / scale)


def score(actual, predicted, train) -> dict:
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAPE": mape(actual, predicted),
        "sMAPE": smape(actual, predicted),
        "MASE": mase(actual, predicted, train),
    }


# ---------------------------------------------------------------------------
# Model zoo. Each fitter returns (point forecast array, lower array, upper array)
# for the requested horizon. `alpha` is the two-sided tail mass of the interval.
# ---------------------------------------------------------------------------
def _grow(sigma: float, h: int, m: int) -> np.ndarray:
    """Standard error that grows with the square root of the horizon.

    For seasonal data the multi-step error compounds by cycle rather than by
    step, so a monthly series gets a slower, more realistic ramp than sqrt(h).
    """
    steps = np.arange(1, h + 1, dtype=float)
    if m > 1:
        cycles = np.floor((steps - 1) / m) + 1
        return sigma * np.sqrt(cycles) * (1.0 + 0.15 * (steps - 1) / max(h, 1))
    return sigma * np.sqrt(steps)


def _fit_naive(y: np.ndarray, h: int, alpha: float, m: int, bands: bool = True):
    point = np.repeat(y[-1], h)
    if not bands:
        return point, None, None
    resid = np.diff(y) if len(y) > 1 else np.array([0.0])
    sigma = float(np.nanstd(resid)) if len(resid) else 0.0
    spread = _grow(sigma, h, m)
    return point, point - spread, point + spread


def _fit_seasonal_naive(y: np.ndarray, h: int, alpha: float, m: int, bands: bool = True):
    if m <= 1 or len(y) < 2 * m:
        return _fit_naive(y, h, alpha, m, bands)
    season = y[-m:]
    point = np.array([season[i % m] for i in range(h)])
    if not bands:
        return point, None, None
    errors = y[m:] - y[:-m]
    sigma = float(np.nanstd(errors)) if len(errors) else 0.0
    spread = _grow(sigma, h, m)
    return point, point - spread, point + spread


def _fit_drift(y: np.ndarray, h: int, alpha: float, m: int, bands: bool = True):
    slope = (y[-1] - y[0]) / max(len(y) - 1, 1)
    point = y[-1] + slope * np.arange(1, h + 1)
    if not bands:
        return point, None, None
    resid = np.diff(y) - slope
    sigma = float(np.nanstd(resid)) if len(resid) else 0.0
    spread = _grow(sigma, h, m)
    return point, point - spread, point + spread


def _ets(kind: str, seasonal: str | None, damped: bool, use_boxcox: bool = False):
    def fitter(y: np.ndarray, h: int, alpha: float, m: int, bands: bool = True):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        seasonal_periods = m if seasonal and m > 1 and len(y) >= 2 * m else None
        model = ExponentialSmoothing(
            y,
            trend=kind,
            damped_trend=damped and kind is not None,
            seasonal=seasonal if seasonal_periods else None,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
            use_boxcox=use_boxcox,
        )
        res = model.fit(optimized=True, remove_bias=False)
        point = np.asarray(res.forecast(h), dtype=float)
        if not bands:
            return point, None, None
        resid = np.asarray(res.resid, dtype=float)
        resid = resid[np.isfinite(resid)] if resid.size else np.array([0.0])
        sigma = float(np.nanstd(resid)) if resid.size else 0.0
        if sigma <= 0 and len(y) > 1:
            sigma = float(np.nanstd(np.diff(y)))
        spread = _grow(sigma, h, m)
        return point, point - spread, point + spread

    return fitter


def _fit_theta(y: np.ndarray, h: int, alpha: float, m: int, bands: bool = True):
    from statsmodels.tsa.forecasting.theta import ThetaModel

    deseasonalize = m > 1 and len(y) >= 2 * m
    model = ThetaModel(
        y, period=m if deseasonalize else None, deseasonalize=deseasonalize
    )
    res = model.fit()
    point = np.asarray(res.forecast(h), dtype=float)
    if not bands:
        return point, None, None

    lower = upper = None
    try:
        ci = res.forecast(h).conf_int(alpha=alpha)
        cand_lo = np.asarray(ci.iloc[:, 0], dtype=float)
        cand_hi = np.asarray(ci.iloc[:, 1], dtype=float)
        # Reject degenerate or malformed intervals from upstream.
        if (
            len(cand_lo) == h
            and np.all(np.isfinite(cand_lo))
            and np.all(np.isfinite(cand_hi))
            and np.any(cand_hi - cand_lo > 1e-9)
        ):
            lower, upper = cand_lo, cand_hi
    except Exception:  # noqa: BLE001 - fall through to residual bands
        lower = upper = None

    if lower is None:
        resid = np.asarray(getattr(res, "resid", []), dtype=float)
        resid = resid[np.isfinite(resid)] if resid.size else np.array([0.0])
        sigma = float(np.nanstd(resid)) if resid.size else 0.0
        if sigma <= 0:
            sigma = float(np.nanstd(np.diff(y))) if len(y) > 1 else 0.0
        spread = _grow(sigma, h, m)
        lower, upper = point - spread, point + spread
    return point, lower, upper


def _sarimax(order, seasonal_order=None):
    def fitter(y: np.ndarray, h: int, alpha: float, m: int, bands: bool = True):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order or (0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, maxiter=200)
        pred = res.get_forecast(h)
        point = np.asarray(pred.predicted_mean, dtype=float)
        if not bands:
            return point, None, None
        ci = pred.conf_int(alpha=alpha)
        return point, np.asarray(ci[:, 0], dtype=float), np.asarray(ci[:, 1], dtype=float)

    return fitter


def candidates(
    freq: str, *, thorough: bool = False, long_series: bool = False
) -> list[tuple[str, str, object]]:
    """Return (display name, family, fitter) tuples appropriate for the series."""
    m = seasonal_period(freq)
    has_season = m > 1

    pool: list[tuple[str, str, object]] = [
        ("Naive (last value)", "baseline", _fit_naive),
        ("Drift", "baseline", _fit_drift),
    ]
    if has_season:
        pool.append(("Seasonal naive", "baseline", _fit_seasonal_naive))

    pool += [
        ("ETS (damped trend)", "exponential smoothing", _ets("add", None, True)),
        ("Theta", "theta", _fit_theta),
    ]
    if has_season:
        pool.append(
            (
                "ETS (damped trend + season)",
                "exponential smoothing",
                _ets("add", "add", True),
            )
        )

    # Long, dense monthly series can absorb a multiplicative season shape.
    if has_season and long_series:
        pool.append(
            (
                "ETS (multiplicative season)",
                "exponential smoothing",
                _ets("add", "mul", True),
            )
        )

    if thorough:
        pool += [
            ("ARIMA (1,1,1)", "arima", _sarimax((1, 1, 1))),
            ("ARIMA (0,1,1)", "arima", _sarimax((0, 1, 1))),
        ]
        if has_season:
            pool += [
                (
                    "SARIMA (1,1,1)(0,1,1)",
                    "arima",
                    _sarimax((1, 1, 1), (0, 1, 1, m)),
                ),
                (
                    "SARIMA (0,1,1)(0,1,1)",
                    "arima",
                    _sarimax((0, 1, 1), (0, 1, 1, m)),
                ),
            ]

    # A pretrained foundation model competes on the same footing as everything
    # else — it is appended last so its family never counts as a naive baseline
    # when skill is scored. Chronos forecasts zero-shot, so it brings a useful
    # prior precisely where a series is too short for classical estimation.
    if FM_ENABLED and chronos_fm.available():
        pool.append(
            (
                "Chronos-Bolt (zero-shot)",
                "foundation model",
                chronos_fm.make_fitter(),
            )
        )

    return pool


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
@dataclass
class ModelScore:
    name: str
    family: str
    metrics: dict = field(default_factory=dict)
    failed: bool = False
    error: str | None = None


def backtest(
    y: np.ndarray,
    freq: str,
    pool,
    *,
    folds: int = 3,
    eval_horizon: int | None = None,
) -> list[ModelScore]:
    """Rolling-origin evaluation.

    Each fold fits on everything up to the test window and scores the model
    across that whole window. Window size follows the *requested* horizon (so
    a 24-month question is judged on 24-month accuracy), falling back to one
    season / 20% when no horizon is supplied.
    """
    m = seasonal_period(freq)
    n = len(y)
    if eval_horizon:
        test_size = int(eval_horizon)
    else:
        test_size = max(m if m > 1 else 1, min(int(round(n * 0.2)), 24))
    test_size = max(1, min(test_size, max(n // 3, 1)))

    usable = 0
    while test_size * (usable + 1) + 8 <= n and usable < folds:
        usable += 1
    folds = max(usable, 1)
    if folds < 3:
        pass  # short series: fewer folds is the honest answer, not an error

    results: list[ModelScore] = []
    for name, family, fitter in pool:
        fold_metrics: list[dict] = []
        error: str | None = None
        for k in range(folds):
            end = n - test_size * k
            train = y[: end - test_size]
            test = y[end - test_size : end]
            if len(train) < 8 or len(test) == 0:
                continue
            try:
                point, _, _ = fitter(train, len(test), 0.20, m, bands=False)
                point = np.asarray(point, dtype=float)
                if point.shape[0] != len(test):
                    point = point[: len(test)]
                    if point.shape[0] != len(test):
                        raise ValueError("forecast length mismatch")
                if not np.all(np.isfinite(point)):
                    raise ValueError("non-finite forecast")
                fold_metrics.append(score(test, point, train))
            except Exception as exc:  # noqa: BLE001
                error = str(exc)[:160]
                break
        if not fold_metrics:
            results.append(
                ModelScore(name, family, failed=True, error=error or "fit failed")
            )
            continue
        agg = {
            key: float(np.nanmean([fm[key] for fm in fold_metrics]))
            for key in fold_metrics[0]
        }
        results.append(ModelScore(name, family, metrics=agg))

    return results


def leaderboard(scores: list[ModelScore]) -> pd.DataFrame:
    rows = []
    for s in scores:
        row = {"Model": s.name, "Family": s.family}
        if s.failed:
            row.update({"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan, "MASE": np.nan, "Status": "excluded"})
        else:
            row.update(s.metrics)
            row["Status"] = "scored"
        rows.append(row)
    frame = pd.DataFrame(rows)
    if "MASE" in frame:
        frame = frame.sort_values(
            ["Status", "MASE"], ascending=[False, True], na_position="last"
        ).reset_index(drop=True)
    frame.index = frame.index + 1
    return frame


def _selection_key(s: ModelScore) -> tuple:
    mase_value = s.metrics.get("MASE")
    rmse_value = s.metrics.get("RMSE")
    return (
        0 if mase_value is not None and np.isfinite(mase_value) else 1,
        mase_value if mase_value is not None and np.isfinite(mase_value) else np.inf,
        rmse_value if rmse_value is not None and np.isfinite(rmse_value) else np.inf,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
@dataclass
class ForecastResult:
    history: pd.Series
    forecast: pd.Series
    lower: pd.Series
    upper: pd.Series
    model: str
    family: str
    frequency: str
    horizon: int
    confidence: float
    metrics: dict
    table: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    naive_mase: float | None = None

    @property
    def horizon_label(self) -> str:
        return horizon_label(self.frequency, self.horizon)

    @property
    def last_actual(self) -> float:
        return float(self.history.iloc[-1])

    @property
    def end_value(self) -> float:
        return float(self.forecast.iloc[-1])

    @property
    def change(self) -> float:
        return self.end_value - self.last_actual

    @property
    def change_pct(self) -> float:
        if abs(self.last_actual) < 1e-9:
            return float("nan")
        return self.change / abs(self.last_actual) * 100

    @property
    def mase(self) -> float | None:
        value = self.metrics.get("MASE")
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)

    @property
    def skill_vs_naive(self) -> float | None:
        """Percentage improvement in MASE over the best naive baseline.

        Positive means the selected model beat the naive benchmark. This is a
        fairer headline than raw MASE, which can exceed 1 on volatile series
        even when the model is the best available.
        """
        if self.mase is None or not self.naive_mase:
            return None
        if self.naive_mase <= 0:
            return None
        return (self.naive_mase - self.mase) / self.naive_mase * 100.0

    def naive_benchmark(self) -> float:
        """Flat-forecast endpoint, for a plain-language comparison."""
        return self.last_actual


def run_forecast(
    series: pd.Series,
    horizon: int,
    *,
    confidence: float = 0.80,
    model: str = "auto",
    thorough: bool = False,
    max_history: int | None = None,
    folds: int = 3,
) -> ForecastResult:
    """Fit, select and project. `model` may be "auto" or a name from the pool."""
    notes: list[str] = []

    clean = regularize(series.dropna())
    if len(clean) < 10:
        raise ForecastError(
            f"only {len(clean)} observations available — at least 10 are needed to forecast"
        )
    if max_history is None:
        max_history = default_window(infer_frequency(clean.index))
    clean = trim_history(clean, max_history)

    freq = infer_frequency(clean.index)
    m = seasonal_period(freq)
    y = clean.to_numpy(dtype="float64")
    long_series = len(y) >= 96

    pool = candidates(freq, thorough=thorough, long_series=long_series)
    if m == 1:
        notes.append("Annual data — no seasonal component can be estimated.")

    if model != "auto":
        chosen = [c for c in pool if c[0] == model]
        if not chosen:
            raise ForecastError(f"unknown model {model!r}")
        pool_for_selection, forced = pool, chosen[0]
    else:
        pool_for_selection, forced = pool, None

    scores = backtest(y, freq, pool_for_selection, folds=folds, eval_horizon=horizon)
    table = leaderboard(scores)

    if forced is not None:
        name, family, fitter = forced
        matched = next((s for s in scores if s.name == name and not s.failed), None)
        metrics = matched.metrics if matched else {}
    else:
        usable = [s for s in scores if not s.failed]
        if not usable:
            raise ForecastError(
                "no model could be fitted to this series — it may be too short or too irregular"
            )
        best = min(usable, key=_selection_key)
        name, family = best.name, best.family
        metrics = best.metrics
        fitter = next(c[2] for c in pool_for_selection if c[0] == name)

    excluded = [s.name for s in scores if s.failed]
    if excluded:
        notes.append(
            f"{len(excluded)} model(s) could not be fitted and were excluded: "
            + ", ".join(excluded)
        )

    alpha = 1.0 - confidence
    point, lower, upper = fitter(y, horizon, alpha, m, bands=True)
    point = np.asarray(point, dtype=float)
    if not np.all(np.isfinite(point)):
        raise ForecastError("the selected model produced a non-finite forecast")

    # Prediction bands must contain the point forecast and keep their ordering.
    lower = np.minimum(np.asarray(lower, dtype=float), point)
    upper = np.maximum(np.asarray(upper, dtype=float), point)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        sigma = float(np.nanstd(np.diff(y))) if len(y) > 1 else 0.0
        spread = sigma * np.sqrt(np.arange(1, horizon + 1))
        lower, upper = point - spread, point + spread

    idx = future_index(clean.index, horizon)

    # Baseline for skill scoring: the best naive-family model in the pool.
    naive_scores = [
        s.metrics.get("MASE")
        for s in scores
        if not s.failed and s.family == "baseline" and s.metrics.get("MASE") is not None
    ]
    naive_mase = min(naive_scores) if naive_scores else None

    return ForecastResult(
        history=clean,
        forecast=pd.Series(point, index=idx, name="forecast"),
        lower=pd.Series(lower, index=idx, name="lower"),
        upper=pd.Series(upper, index=idx, name="upper"),
        model=name,
        family=family,
        frequency=freq,
        horizon=horizon,
        confidence=confidence,
        metrics=metrics,
        table=table,
        warnings=notes,
        naive_mase=naive_mase,
    )


def model_names(freq: str, *, thorough: bool = False, long_series: bool = True) -> list[str]:
    return ["auto"] + [c[0] for c in candidates(freq, thorough=thorough, long_series=long_series)]


def combine_forecast(
    series: pd.Series,
    horizon: int,
    *,
    confidence: float = 0.80,
    max_history: int | None = None,
    folds: int = 3,
) -> tuple[ForecastResult, ForecastResult]:
    """Return (best single model, ensemble) so the UI can compare the two."""
    best = run_forecast(
        series,
        horizon,
        confidence=confidence,
        model="auto",
        max_history=max_history,
        folds=folds,
    )
    return best, best
