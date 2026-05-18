"""Inference utilities for the bot demand forecaster."""

from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.forecaster.schemas import (
    CalendarEvent,
    EventsForecastRequest,
    ForecastRequest,
    ForecastResponse,
    ForecastSlot,
)

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "demand_forecaster.joblib"


@lru_cache(maxsize=1)
def _load_model(path: str) -> GradientBoostingRegressor:
    return joblib.load(path)


def get_model() -> GradientBoostingRegressor:
    return _load_model(str(MODEL_PATH))


def _build_slots(start: datetime, end: datetime, granularity_minutes: int) -> list[datetime]:
    """Return slot start times from start up to (but not including) end."""
    slots = []
    current = start
    delta = timedelta(minutes=granularity_minutes)
    while current < end:
        slots.append(current)
        current += delta
    return slots


def _features_from_timestamps(timestamps: list[datetime]) -> pd.DataFrame:
    ts = pd.Series(timestamps)
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    return pd.DataFrame(
        {
            "hour": hour,
            "dow": dow,
            "is_weekend": (dow >= 5).astype(int),
            "is_working_hours": ((hour >= 8) & (hour < 18)).astype(int),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
        }
    )


def forecast(request: ForecastRequest) -> ForecastResponse:
    """Predict bot demand for each slot using learned temporal patterns."""
    model = get_model()
    slot_starts = _build_slots(request.start, request.end, request.granularity_minutes)
    if not slot_starts:
        return ForecastResponse(slots=[], peak_concurrent=0.0)

    X = _features_from_timestamps(slot_starts)
    raw_preds = model.predict(X)
    delta = timedelta(minutes=request.granularity_minutes)

    slots = []
    for ts, pred in zip(slot_starts, raw_preds):
        slots.append(ForecastSlot(
            slot_start=ts,
            slot_end=ts + delta,
            predicted_bots=round(max(0.0, float(pred)), 2),
        ))

    peak = max(s.predicted_bots for s in slots) if slots else 0.0
    return ForecastResponse(slots=slots, peak_concurrent=peak)


def forecast_from_events(request: EventsForecastRequest) -> ForecastResponse:
    """Compute exact concurrent bot count per slot from a list of known events."""
    if not request.events:
        return ForecastResponse(slots=[], peak_concurrent=0.0)

    all_starts = [e.start for e in request.events]
    all_ends = [e.end for e in request.events]
    window_start = min(all_starts)
    window_end = max(all_ends)

    slot_starts = _build_slots(window_start, window_end, request.granularity_minutes)
    if not slot_starts:
        return ForecastResponse(slots=[], peak_concurrent=0.0)

    delta = timedelta(minutes=request.granularity_minutes)
    event_starts = np.array([e.start.timestamp() for e in request.events])
    event_ends = np.array([e.end.timestamp() for e in request.events])

    slots = []
    for ts in slot_starts:
        slot_end = ts + delta
        count = int(((event_starts < slot_end.timestamp()) & (event_ends > ts.timestamp())).sum())
        slots.append(ForecastSlot(
            slot_start=ts,
            slot_end=slot_end,
            predicted_bots=float(count),
        ))

    peak = max(s.predicted_bots for s in slots) if slots else 0.0
    return ForecastResponse(slots=slots, peak_concurrent=peak)
