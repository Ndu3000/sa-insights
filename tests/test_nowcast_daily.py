"""Tests for the daily-cadence support and the GDP nowcast.

Two guarantees are asserted here:

1. Daily series behave like daily series — frequency inference, and a
   calendar-based year-on-year comparison that does not drift with holidays.
2. The nowcast is honest — it refuses to emit a number without enough history
   to backtest, it beats naive on data with real signal, and it never uses
   future information (drivers enter as the prior quarter's mean).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sa_insights import nowcast, store
from sa_insights.chat import tools
from sa_insights.forecast import infer_frequency


# --------------------------------------------------------------------------- #
# Daily cadence
# --------------------------------------------------------------------------- #
def test_infer_frequency_detects_business_days() -> None:
    idx = pd.bdate_range("2024-01-01", periods=300)
    assert infer_frequency(idx) == "D"


def test_infer_frequency_still_detects_monthly() -> None:
    idx = pd.date_range("2010-01-01", periods=120, freq="MS")
    assert infer_frequency(idx) == "M"


@pytest.fixture()
def daily_db(tmp_path, monkeypatch):
    """A throwaway warehouse holding one daily series with a known YoY move."""
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.duckdb")
    monkeypatch.setattr(store, "LOCK_PATH", tmp_path / "test.lock")

    idx = pd.bdate_range("2024-01-01", "2026-11-18")
    # 10.00 until 2026, then a step to 12.00: YoY at the end (Nov 2026 vs
    # Nov 2025) is +2.00.
    values = np.where(idx < pd.Timestamp("2026-01-01"), 10.0, 12.0)
    series = pd.Series(values, index=idx, name="fx_test")
    store.write_observations({"fx_test": series})
    store.write_indicator_metadata(
        [
            {
                "series_key": "fx_test",
                "label": "Test rand per dollar",
                "short_label": "FX test",
                "category": "External Sector",
                "unit": "ZAR",
                "frequency": "D",
                "provider": "test",
                "description": "synthetic",
                "higher_is_better": "False",
                "decimals": 2,
                "first_period": idx[0].date(),
                "last_period": idx[-1].date(),
                "observation_count": len(series),
                "updated_at": pd.Timestamp.now(),
            }
        ]
    )
    return series


def test_daily_yoy_matches_calendar_not_row_offset(daily_db) -> None:
    snap = tools.get_series("fx_test", periods=30)
    assert snap.yoy is not None
    # A row offset of 252 would land on a different calendar day because of
    # holidays; the calendar match must still find 10.00 a year earlier.
    assert snap.yoy.from_value == pytest.approx(10.0)
    assert snap.yoy.absolute == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# Nowcast
# --------------------------------------------------------------------------- #
def _synthetic_data(n_quarters: int = 80, seed: int = 7) -> dict[str, pd.Series]:
    """GDP driven by a monthly indicator, so a bridge equation has real signal."""
    rng = np.random.default_rng(seed)
    q_idx = pd.date_range("2004-01-01", periods=n_quarters, freq="QS")
    m_idx = pd.date_range("2003-01-01", periods=n_quarters * 3 + 12, freq="MS")

    driver = pd.Series(
        100 + np.cumsum(rng.normal(0.3, 1.0, len(m_idx))), index=m_idx
    )
    q_driver = driver.resample("QS").mean().reindex(q_idx)
    # GDP level responds to the *previous* quarter's driver plus noise.
    level = 1000 + 5.0 * q_driver.shift(1).fillna(q_driver.iloc[0])
    level = level + np.cumsum(rng.normal(2.0, 4.0, n_quarters))
    gdp = pd.Series(level.to_numpy(), index=q_idx)
    return {"real_gdp_q": gdp, "leading_indicator": driver}


def test_nowcast_refuses_short_history() -> None:
    data = _synthetic_data(n_quarters=20)
    assert nowcast.nowcast_gdp(data) is None


def test_nowcast_refuses_without_drivers() -> None:
    data = _synthetic_data()
    data.pop("leading_indicator")
    assert nowcast.nowcast_gdp(data) is None


def test_nowcast_beats_naive_and_labels_itself() -> None:
    result = nowcast.nowcast_gdp(_synthetic_data())
    assert result is not None
    assert result.skill_vs_naive > 0
    assert result.lower < result.value < result.upper
    assert result.period > pd.Timestamp("2004-01-01")
    payload = result.as_dict()
    assert payload["kind"] == "nowcast"
    assert payload["drivers_used"] == ["leading_indicator"]


def test_nowcast_targets_the_unprinted_quarter() -> None:
    data = _synthetic_data()
    result = nowcast.nowcast_gdp(data)
    assert result is not None
    last_q = data["real_gdp_q"].index[-1]
    assert result.period == last_q + pd.offsets.QuarterBegin(1)


def test_design_uses_only_past_driver_information() -> None:
    """Leakage guard: the driver column for quarter t must equal t-1's mean."""
    data = _synthetic_data()
    gdp = data["real_gdp_q"]
    monthly = pd.DataFrame({"leading_indicator": data["leading_indicator"]})
    frame = nowcast._design(gdp, monthly)
    qmean = monthly.resample("QS").mean()["leading_indicator"]
    aligned = frame["leading_indicator_lag1"]
    expected = qmean.shift(1).reindex(aligned.index)
    pd.testing.assert_series_equal(aligned, expected, check_names=False)