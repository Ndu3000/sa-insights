"""Bridge-equation nowcast of quarterly GDP growth from monthly indicators.

Why a nowcast, and why only GDP
-------------------------------
The official quarterly GDP print arrives ~2 quarters after the period it
describes. A bridge equation regresses the quarterly target on the monthly
indicators that are already published for the current quarter, giving an
early, clearly-labelled estimate of the missing print.

We tested four candidate targets walk-forward (fit on the past, predict the
next quarter) against a naive "last known value" baseline:

    gdp_yoy        +13-14% skill   -> worth nowcasting
    cpi_yoy        -54%            -> worse than naive, do NOT nowcast
    unemployment   +0.5%           -> no signal, do NOT nowcast
    retail_next_q  -20%            -> worse than naive, do NOT nowcast

So this module nowcasts GDP only. Adding a target that fails the backtest
would produce confident wrong numbers, which is exactly what the rest of the
system is built to avoid.

Honesty rules
-------------
* The output is an ESTIMATE, never an observation. It is stored under a
  distinct key (``gdp_nowcast``) and labelled as a nowcast everywhere.
* Every nowcast ships with its walk-forward skill vs naive and a residual
  band, so a reader can see how much to trust it.
* Only PAST information is used: monthly drivers enter as the prior quarter's
  average, and the target enters as lags. Nothing from the future leaks in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Monthly indicators used as drivers. Chosen because they publish ahead of GDP
# and carry real-economy signal; each enters as the previous quarter's mean.
DRIVERS = [
    "usd_zar",
    "leading_indicator",
    "retail_sales",
    "manufacturing",
    "mining",
    "bond_yield",
    "policy_rate",
    "vehicle_sales",
]

TARGET = "gdp_yoy"          # derived below from real_gdp_q
MIN_TRAIN = 40              # quarters before the first walk-forward prediction
RIDGE_ALPHA = 1.0


@dataclass
class Nowcast:
    period: pd.Timestamp
    value: float
    lower: float
    upper: float
    skill_vs_naive: float
    rmse: float
    n_train: int
    drivers_used: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "period": self.period.strftime("%Y-%m-%d"),
            "value": round(self.value, 3),
            "lower": round(self.lower, 3),
            "upper": round(self.upper, 3),
            "skill_vs_naive": round(self.skill_vs_naive, 3),
            "rmse": round(self.rmse, 3),
            "n_train": self.n_train,
            "drivers_used": self.drivers_used,
            "kind": "nowcast",
        }


def _design(gdp_level: pd.Series, monthly: pd.DataFrame) -> pd.DataFrame:
    """Build the lagged design matrix and the YoY target, aligned by quarter."""
    target = gdp_level.pct_change(4) * 100.0
    qmean = monthly.resample("QS").mean()

    X = pd.DataFrame(index=gdp_level.index)
    for col in qmean.columns:
        X[col + "_lag1"] = qmean[col].shift(1).reindex(gdp_level.index)
    X["g_lag1"] = gdp_level.shift(1)
    X["g_lag4"] = gdp_level.shift(4)

    frame = pd.concat([target.rename("y"), X], axis=1).dropna()
    return frame


def backtest(frame: pd.DataFrame) -> tuple[float, float, np.ndarray]:
    """Walk-forward RMSE of the ridge bridge vs a naive last-value baseline."""
    from sklearn.linear_model import Ridge

    errs_model, errs_naive = [], []
    for i in range(MIN_TRAIN, len(frame)):
        train, test = frame.iloc[:i], frame.iloc[i]
        model = Ridge(alpha=RIDGE_ALPHA).fit(train.drop(columns="y"), train["y"])
        pred = float(model.predict(test.drop("y").to_frame().T)[0])
        errs_model.append(pred - test["y"])
        errs_naive.append(train["y"].iloc[-1] - test["y"])

    rmse = float(np.sqrt(np.mean(np.square(errs_model))))
    naive = float(np.sqrt(np.mean(np.square(errs_naive))))
    return rmse, naive, np.asarray(errs_model)


def nowcast_gdp(data: dict[str, pd.Series]) -> Nowcast | None:
    """Estimate YoY GDP growth for the quarter after the last official print.

    Returns None when there is not enough history to backtest honestly, rather
    than emitting an unvalidated number.
    """
    gdp = data.get("real_gdp_q")
    if gdp is None or len(gdp) < MIN_TRAIN + 8:
        return None

    monthly = pd.DataFrame({k: data[k] for k in DRIVERS if k in data})
    if monthly.empty:
        return None

    frame = _design(gdp, monthly)
    if len(frame) < MIN_TRAIN + 4:
        return None

    rmse, naive, _residuals = backtest(frame)
    skill = 1.0 - rmse / naive if naive else 0.0

    # Refit on all available history and predict the next (unprinted) quarter.
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=RIDGE_ALPHA).fit(frame.drop(columns="y"), frame["y"])

    last_q = gdp.index[-1]
    next_q = last_q + pd.offsets.QuarterBegin(1)
    qmean = monthly.resample("QS").mean()

    row = {}
    for col in qmean.columns:
        # The prior quarter's mean is known at prediction time.
        row[col + "_lag1"] = qmean[col].reindex(gdp.index).iloc[-1]
    row["g_lag1"] = gdp.iloc[-1]
    row["g_lag4"] = gdp.iloc[-4]
    X_next = pd.DataFrame([row])[frame.drop(columns="y").columns]

    value = float(model.predict(X_next)[0])
    band = 1.28 * rmse  # ~80% interval under a normal residual assumption

    return Nowcast(
        period=next_q,
        value=value,
        lower=value - band,
        upper=value + band,
        skill_vs_naive=skill,
        rmse=rmse,
        n_train=len(frame),
        drivers_used=[c for c in qmean.columns],
    )
