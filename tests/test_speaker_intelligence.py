"""Tests for the speaker intelligence API endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_ENGAGEMENTS = {"balanced", "presenter_led", "dominated"}


def _analyze(meeting_id: str, duration: float, segments: list[dict]) -> dict:
    response = client.post("/speaker/analyze", json={
        "meeting_id": meeting_id,
        "duration_seconds": duration,
        "segments": segments,
    })
    assert response.status_code == 200
    return response.json()


# ── Helpers: realistic meeting fixtures ─────────────────────────────────────

def _make_balanced_segments(duration: float = 3600.0) -> list[dict]:
    """Four speakers sharing time roughly equally in alternating 2-min turns."""
    speakers = ["Alice", "Bob", "Carol", "Dave"]
    segs = []
    t = 0.0
    while t + 120 <= duration - 60:  # leave some silence at end
        for sp in speakers:
            if t + 120 > duration:
                break
            segs.append({"speaker": sp, "start": t, "end": t + 110.0})
            t += 120.0
    return segs


def _make_dominated_segments(duration: float = 3600.0) -> list[dict]:
    """One speaker talks 85%, three others share the remaining 15%."""
    segs = []
    # dominator: 0.85 * 3600 = 3060s in many turns
    t = 0.0
    while t + 300 <= duration * 0.85:
        segs.append({"speaker": "Boss", "start": t, "end": t + 290.0})
        t += 300.0
    # others share remaining ~540s
    others = ["A", "B", "C"]
    chunk = 60.0
    for i, sp in enumerate(others):
        start = duration * 0.85 + i * chunk * 3
        end = start + chunk * 2
        if end <= duration:
            segs.append({"speaker": sp, "start": start, "end": end})
    return segs


def _make_presenter_led_segments(duration: float = 3600.0) -> list[dict]:
    """Presenter at ~62%, three participants sharing the rest."""
    segs = []
    target_presenter = duration * 0.62
    chunk = 300.0
    t = 0.0
    while t + chunk <= target_presenter:
        segs.append({"speaker": "Presenter", "start": t, "end": t + chunk - 10})
        t += chunk
    # participants share remaining ~38%
    participants = ["P1", "P2", "P3"]
    p_start = target_presenter
    p_chunk = (duration - target_presenter) / len(participants)
    for sp in participants:
        segs.append({"speaker": sp, "start": p_start, "end": p_start + p_chunk - 10})
        p_start += p_chunk
    return segs


# ── Response structure ───────────────────────────────────────────────────────

def test_analyze_returns_200():
    segs = [{"speaker": "Alice", "start": 0.0, "end": 60.0}]
    response = client.post("/speaker/analyze", json={
        "meeting_id": "m1", "duration_seconds": 120.0, "segments": segs
    })
    assert response.status_code == 200


def test_analyze_response_structure():
    data = _analyze("m1", 3600.0, _make_balanced_segments())
    assert "meeting_id" in data
    assert "duration_seconds" in data
    assert "speaker_count" in data
    assert "speaker_stats" in data
    assert "silence_ratio" in data
    assert "interruption_count" in data
    assert "dominance_index" in data
    assert "engagement" in data
    assert "engagement_confidence" in data


def test_speaker_stats_structure():
    data = _analyze("m1", 3600.0, _make_balanced_segments())
    for stat in data["speaker_stats"]:
        assert "speaker" in stat
        assert "talk_time_seconds" in stat
        assert "talk_ratio" in stat
        assert "turn_count" in stat
        assert "longest_monologue_seconds" in stat
        assert "avg_turn_duration_seconds" in stat


def test_echoes_meeting_id():
    data = _analyze("unique-id-xyz", 600.0, [{"speaker": "A", "start": 0.0, "end": 100.0}])
    assert data["meeting_id"] == "unique-id-xyz"


def test_confidence_in_range():
    data = _analyze("m1", 3600.0, _make_balanced_segments())
    assert 0.0 <= data["engagement_confidence"] <= 1.0


def test_engagement_is_valid_label():
    data = _analyze("m1", 3600.0, _make_balanced_segments())
    assert data["engagement"] in VALID_ENGAGEMENTS


# ── Per-speaker analytics correctness ───────────────────────────────────────

def test_talk_ratios_sum_to_at_most_one():
    data = _analyze("m1", 3600.0, _make_balanced_segments())
    total = sum(s["talk_ratio"] for s in data["speaker_stats"])
    assert total <= 1.001  # floating point tolerance


def test_silence_ratio_plus_talk_approx_one():
    data = _analyze("m1", 3600.0, _make_balanced_segments())
    total_talk = sum(s["talk_ratio"] for s in data["speaker_stats"])
    assert abs((total_talk + data["silence_ratio"]) - 1.0) < 0.01


def test_speaker_count_correct():
    data = _analyze("m1", 3600.0, _make_balanced_segments())
    assert data["speaker_count"] == len(data["speaker_stats"])


def test_stats_sorted_by_talk_time_desc():
    data = _analyze("m1", 3600.0, _make_dominated_segments())
    times = [s["talk_time_seconds"] for s in data["speaker_stats"]]
    assert times == sorted(times, reverse=True)


def test_single_speaker_dominance_near_one():
    segs = [{"speaker": "Alice", "start": 0.0, "end": 590.0}]
    data = _analyze("m1", 600.0, segs)
    # single speaker → dominance ≈ talk_ratio²
    assert data["dominance_index"] > 0.9


def test_turn_count_matches_segments():
    segs = [
        {"speaker": "Alice", "start": 0.0,   "end": 30.0},
        {"speaker": "Bob",   "start": 35.0,  "end": 65.0},
        {"speaker": "Alice", "start": 70.0,  "end": 100.0},
        {"speaker": "Alice", "start": 110.0, "end": 140.0},
    ]
    data = _analyze("m1", 200.0, segs)
    alice = next(s for s in data["speaker_stats"] if s["speaker"] == "Alice")
    bob = next(s for s in data["speaker_stats"] if s["speaker"] == "Bob")
    assert alice["turn_count"] == 3
    assert bob["turn_count"] == 1


def test_longest_monologue_correct():
    segs = [
        {"speaker": "Alice", "start": 0.0,  "end": 10.0},
        {"speaker": "Alice", "start": 20.0, "end": 80.0},  # longest: 60s
        {"speaker": "Alice", "start": 90.0, "end": 100.0},
    ]
    data = _analyze("m1", 120.0, segs)
    alice = next(s for s in data["speaker_stats"] if s["speaker"] == "Alice")
    assert abs(alice["longest_monologue_seconds"] - 60.0) < 0.1


def test_silence_with_gap():
    # Alice talks 0-50, then silence 50-90, then Bob 90-100
    segs = [
        {"speaker": "Alice", "start": 0.0,  "end": 50.0},
        {"speaker": "Bob",   "start": 90.0, "end": 100.0},
    ]
    data = _analyze("m1", 100.0, segs)
    # silence = 40s out of 100s = 0.40
    assert abs(data["silence_ratio"] - 0.40) < 0.01


def test_interruption_detected():
    # Bob starts 0.5s after Alice ends → interruption
    segs = [
        {"speaker": "Alice", "start": 0.0,  "end": 30.0},
        {"speaker": "Bob",   "start": 30.5, "end": 60.0},
    ]
    data = _analyze("m1", 60.0, segs)
    assert data["interruption_count"] >= 1


def test_no_interruption_with_long_gap():
    segs = [
        {"speaker": "Alice", "start": 0.0,  "end": 30.0},
        {"speaker": "Bob",   "start": 40.0, "end": 60.0},  # 10s gap → not interruption
    ]
    data = _analyze("m1", 60.0, segs)
    assert data["interruption_count"] == 0


# ── Engagement classification (golden) ──────────────────────────────────────

def test_balanced_engagement():
    data = _analyze("balanced", 3600.0, _make_balanced_segments())
    assert data["engagement"] == "balanced"


def test_dominated_engagement():
    data = _analyze("dominated", 3600.0, _make_dominated_segments())
    assert data["engagement"] == "dominated"


def test_presenter_led_engagement():
    data = _analyze("presenter_led", 3600.0, _make_presenter_led_segments())
    assert data["engagement"] == "presenter_led"


def test_single_speaker_is_dominated():
    # Alice speaks in 90s blocks covering ~90% of the 3600s meeting
    segs = [{"speaker": "Alice", "start": float(i * 100), "end": float(i * 100 + 90)} for i in range(36)]
    data = _analyze("solo", 3600.0, segs)
    assert data["engagement"] == "dominated"


# ── Validation edge cases ────────────────────────────────────────────────────

def test_segment_end_before_start_rejected():
    response = client.post("/speaker/analyze", json={
        "meeting_id": "m1",
        "duration_seconds": 60.0,
        "segments": [{"speaker": "Alice", "start": 30.0, "end": 10.0}],
    })
    assert response.status_code == 422


def test_empty_segments_rejected():
    response = client.post("/speaker/analyze", json={
        "meeting_id": "m1",
        "duration_seconds": 60.0,
        "segments": [],
    })
    assert response.status_code == 422


def test_zero_duration_rejected():
    response = client.post("/speaker/analyze", json={
        "meeting_id": "m1",
        "duration_seconds": 0.0,
        "segments": [{"speaker": "A", "start": 0.0, "end": 10.0}],
    })
    assert response.status_code == 422


def test_duration_over_24h_rejected():
    response = client.post("/speaker/analyze", json={
        "meeting_id": "m1",
        "duration_seconds": 86401.0,
        "segments": [{"speaker": "A", "start": 0.0, "end": 10.0}],
    })
    assert response.status_code == 422


def test_empty_speaker_name_rejected():
    response = client.post("/speaker/analyze", json={
        "meeting_id": "m1",
        "duration_seconds": 60.0,
        "segments": [{"speaker": "", "start": 0.0, "end": 10.0}],
    })
    assert response.status_code == 422


# ── Batch endpoint ───────────────────────────────────────────────────────────

BATCH_PAYLOAD = {
    "meetings": [
        {
            "meeting_id": "b1",
            "duration_seconds": 3600.0,
            "segments": _make_balanced_segments(),
        },
        {
            "meeting_id": "b2",
            "duration_seconds": 3600.0,
            "segments": _make_dominated_segments(),
        },
    ]
}


def test_batch_returns_200():
    response = client.post("/speaker/analyze/batch", json=BATCH_PAYLOAD)
    assert response.status_code == 200


def test_batch_returns_correct_count():
    response = client.post("/speaker/analyze/batch", json=BATCH_PAYLOAD)
    assert len(response.json()["results"]) == 2


def test_batch_preserves_order():
    response = client.post("/speaker/analyze/batch", json=BATCH_PAYLOAD)
    ids = [r["meeting_id"] for r in response.json()["results"]]
    assert ids == ["b1", "b2"]


def test_batch_engagements_correct():
    response = client.post("/speaker/analyze/batch", json=BATCH_PAYLOAD)
    results = response.json()["results"]
    assert results[0]["engagement"] == "balanced"
    assert results[1]["engagement"] == "dominated"


def test_batch_empty_rejected():
    response = client.post("/speaker/analyze/batch", json={"meetings": []})
    assert response.status_code == 422
