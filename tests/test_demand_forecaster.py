"""Tests for the bot demand forecaster API endpoints."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

# Monday 9 AM UTC — a reliably busy slot
BUSY_START = "2025-06-02T09:00:00"
BUSY_END = "2025-06-02T17:00:00"


def test_forecast_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_forecast_returns_slots():
    response = client.post("/forecast", json={"start": BUSY_START, "end": BUSY_END})
    assert response.status_code == 200
    data = response.json()
    assert "slots" in data
    assert "peak_concurrent" in data
    # 8-hour window at 60-min granularity → 8 slots
    assert len(data["slots"]) == 8


def test_forecast_slot_structure():
    response = client.post("/forecast", json={"start": BUSY_START, "end": BUSY_END})
    slot = response.json()["slots"][0]
    assert "slot_start" in slot
    assert "slot_end" in slot
    assert "predicted_bots" in slot
    assert slot["predicted_bots"] >= 0.0


def test_forecast_peak_matches_slots():
    response = client.post("/forecast", json={"start": BUSY_START, "end": BUSY_END})
    data = response.json()
    slot_max = max(s["predicted_bots"] for s in data["slots"])
    assert abs(data["peak_concurrent"] - slot_max) < 0.01


def test_forecast_weekday_busier_than_weekend():
    """Monday morning should yield higher predicted demand than Sunday morning."""
    monday = {"start": "2025-06-02T09:00:00", "end": "2025-06-02T12:00:00"}
    sunday = {"start": "2025-06-01T09:00:00", "end": "2025-06-01T12:00:00"}
    monday_peak = client.post("/forecast", json=monday).json()["peak_concurrent"]
    sunday_peak = client.post("/forecast", json=sunday).json()["peak_concurrent"]
    assert monday_peak > sunday_peak


def test_forecast_working_hours_busier_than_night():
    """9 AM peak should be higher than 2 AM."""
    day = {"start": "2025-06-02T09:00:00", "end": "2025-06-02T11:00:00"}
    night = {"start": "2025-06-02T02:00:00", "end": "2025-06-02T04:00:00"}
    day_peak = client.post("/forecast", json=day).json()["peak_concurrent"]
    night_peak = client.post("/forecast", json=night).json()["peak_concurrent"]
    assert day_peak > night_peak


def test_forecast_custom_granularity():
    response = client.post(
        "/forecast",
        json={"start": BUSY_START, "end": BUSY_END, "granularity_minutes": 30},
    )
    assert response.status_code == 200
    # 8 hours / 30 min = 16 slots
    assert len(response.json()["slots"]) == 16


def test_forecast_end_before_start_rejected():
    response = client.post(
        "/forecast",
        json={"start": BUSY_END, "end": BUSY_START},
    )
    assert response.status_code == 422


def test_forecast_end_equals_start_rejected():
    response = client.post(
        "/forecast",
        json={"start": BUSY_START, "end": BUSY_START},
    )
    assert response.status_code == 422


def test_forecast_window_exceeds_7_days_rejected():
    response = client.post(
        "/forecast",
        json={"start": "2025-06-01T00:00:00", "end": "2025-06-09T00:00:00"},
    )
    assert response.status_code == 422


def test_forecast_granularity_too_small_rejected():
    response = client.post(
        "/forecast",
        json={"start": BUSY_START, "end": BUSY_END, "granularity_minutes": 5},
    )
    assert response.status_code == 422


EVENTS_PAYLOAD = {
    "events": [
        {"start": "2025-06-02T09:00:00", "end": "2025-06-02T10:00:00"},
        {"start": "2025-06-02T09:30:00", "end": "2025-06-02T10:30:00"},
        {"start": "2025-06-02T11:00:00", "end": "2025-06-02T12:00:00"},
    ]
}


def test_from_events_returns_200():
    response = client.post("/forecast/from-events", json=EVENTS_PAYLOAD)
    assert response.status_code == 200


def test_from_events_exact_overlap():
    """Two overlapping events in the 9 AM slot → count of 2."""
    response = client.post("/forecast/from-events", json=EVENTS_PAYLOAD)
    slots = response.json()["slots"]
    # first slot covers 09:00–10:00 — both first two events overlap it
    nine_am_slots = [s for s in slots if "T09:00:00" in s["slot_start"]]
    assert len(nine_am_slots) == 1
    assert nine_am_slots[0]["predicted_bots"] == 2.0


def test_from_events_non_overlapping_slot():
    """The 11 AM event should be isolated (count of 1)."""
    response = client.post("/forecast/from-events", json=EVENTS_PAYLOAD)
    slots = response.json()["slots"]
    eleven_am_slots = [s for s in slots if "T11:00:00" in s["slot_start"]]
    assert len(eleven_am_slots) == 1
    assert eleven_am_slots[0]["predicted_bots"] == 1.0


def test_from_events_peak_concurrent():
    response = client.post("/forecast/from-events", json=EVENTS_PAYLOAD)
    assert response.json()["peak_concurrent"] == 2.0


def test_from_events_custom_granularity():
    payload = {**EVENTS_PAYLOAD, "granularity_minutes": 30}
    response = client.post("/forecast/from-events", json=payload)
    assert response.status_code == 200
    slots = response.json()["slots"]
    # 09:00–12:00 window at 30 min = 6 slots
    assert len(slots) == 6


def test_from_events_empty_list_rejected():
    response = client.post("/forecast/from-events", json={"events": []})
    assert response.status_code == 422


def test_from_events_invalid_event_rejected():
    payload = {
        "events": [{"start": "2025-06-02T10:00:00", "end": "2025-06-02T09:00:00"}]
    }
    response = client.post("/forecast/from-events", json=payload)
    assert response.status_code == 422


def test_from_events_single_event():
    payload = {
        "events": [{"start": "2025-06-02T14:00:00", "end": "2025-06-02T15:00:00"}]
    }
    response = client.post("/forecast/from-events", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["peak_concurrent"] == 1.0
