"""Tests for the agent config recommender API endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_RECORDING_MODES = {"audio_only", "audio_video"}
VALID_SUMMARY_DEPTHS = {"brief", "standard", "detailed"}
VALID_OFFSETS = {0, -30, -60, -120}


def _recommend(meeting_type: str, attendees: int, duration: int) -> dict:
    response = client.post("/recommend", json={
        "meeting_type": meeting_type,
        "attendee_count": attendees,
        "duration_minutes": duration,
    })
    assert response.status_code == 200
    return response.json()


# ── Schema and structure ─────────────────────────────────────────────────────


def test_recommend_returns_200():
    response = client.post("/recommend", json={
        "meeting_type": "standup",
        "attendee_count": 8,
        "duration_minutes": 15,
    })
    assert response.status_code == 200


def test_recommend_response_structure():
    data = _recommend("planning", 6, 60)
    assert "meeting_type" in data
    assert "attendee_count" in data
    assert "duration_minutes" in data
    assert "config" in data
    assert "confidence" in data

    cfg = data["config"]
    assert isinstance(cfg["summary_enabled"], bool)
    assert isinstance(cfg["action_items_enabled"], bool)
    assert isinstance(cfg["speaker_diarization"], bool)
    assert cfg["recording_mode"] in VALID_RECORDING_MODES
    assert cfg["summary_depth"] in VALID_SUMMARY_DEPTHS
    assert cfg["bot_join_offset_seconds"] in VALID_OFFSETS


def test_recommend_confidence_in_range():
    data = _recommend("client", 10, 60)
    assert 0.0 <= data["confidence"] <= 1.0


def test_recommend_echoes_request_fields():
    data = _recommend("interview", 3, 45)
    assert data["meeting_type"] == "interview"
    assert data["attendee_count"] == 3
    assert data["duration_minutes"] == 45


# ── Golden assertions — deterministic fields (100% CV accuracy) ──────────────


@pytest.mark.parametrize("mtype,att,dur,expected_mode", [
    ("client",    10, 60, "audio_video"),
    ("all_hands", 50, 60, "audio_video"),
    ("interview",  3, 45, "audio_video"),
    ("standup",    8, 15, "audio_only"),
    ("planning",   6, 60, "audio_only"),
    ("one_on_one", 2, 30, "audio_only"),
    ("workshop",   8, 90, "audio_only"),   # under 10 att → no video upgrade
    ("social",     8, 45, "audio_only"),   # under 10 att → no video upgrade
])
def test_recording_mode_golden(mtype, att, dur, expected_mode):
    cfg = _recommend(mtype, att, dur)["config"]
    assert cfg["recording_mode"] == expected_mode, (
        f"{mtype} ({att} att, {dur} min) → got {cfg['recording_mode']!r}, expected {expected_mode!r}"
    )


@pytest.mark.parametrize("mtype,att,dur,expected_depth", [
    ("standup",    8, 15, "brief"),
    ("planning",   6, 90, "detailed"),
    ("client",    10, 60, "detailed"),
    ("interview",  3, 45, "detailed"),
    ("workshop",  10, 90, "detailed"),
    ("social",    15, 45, "brief"),
    ("all_hands", 50, 60, "standard"),
])
def test_summary_depth_golden(mtype, att, dur, expected_depth):
    cfg = _recommend(mtype, att, dur)["config"]
    assert cfg["summary_depth"] == expected_depth


@pytest.mark.parametrize("mtype,att,dur,expected_offset", [
    ("client",    10, 60, -120),
    ("all_hands", 50, 60, -120),
    ("planning",   6, 60, -60),
    ("interview",  3, 45, -60),
    ("workshop",  10, 90, -60),
    ("standup",    8, 15, 0),
    ("one_on_one", 2, 30, 0),
    ("social",    15, 45, 0),
])
def test_bot_join_offset_golden(mtype, att, dur, expected_offset):
    cfg = _recommend(mtype, att, dur)["config"]
    assert cfg["bot_join_offset_seconds"] == expected_offset


# ── Attendee-count driven logic ──────────────────────────────────────────────


def test_large_meeting_gets_audio_video():
    """Any type with 10+ attendees should get audio_video recording."""
    # workshop base is audio_only but large crowds warrant video
    cfg = _recommend("workshop", 15, 90)["config"]
    assert cfg["recording_mode"] == "audio_video"


def test_one_on_one_no_diarization():
    """Two-person meeting doesn't need speaker diarization."""
    cfg = _recommend("one_on_one", 2, 30)["config"]
    assert cfg["speaker_diarization"] is False


def test_group_meeting_gets_diarization():
    """Meetings with 3+ attendees should have diarization enabled."""
    cfg = _recommend("planning", 6, 60)["config"]
    assert cfg["speaker_diarization"] is True


# ── Validation edge cases ────────────────────────────────────────────────────


def test_invalid_meeting_type_rejected():
    response = client.post("/recommend", json={
        "meeting_type": "brainstorm",
        "attendee_count": 5,
        "duration_minutes": 60,
    })
    assert response.status_code == 422


def test_zero_attendees_rejected():
    response = client.post("/recommend", json={
        "meeting_type": "standup",
        "attendee_count": 0,
        "duration_minutes": 15,
    })
    assert response.status_code == 422


def test_zero_duration_rejected():
    response = client.post("/recommend", json={
        "meeting_type": "standup",
        "attendee_count": 5,
        "duration_minutes": 0,
    })
    assert response.status_code == 422


def test_attendee_count_over_limit_rejected():
    response = client.post("/recommend", json={
        "meeting_type": "all_hands",
        "attendee_count": 501,
        "duration_minutes": 60,
    })
    assert response.status_code == 422


def test_duration_over_limit_rejected():
    response = client.post("/recommend", json={
        "meeting_type": "workshop",
        "attendee_count": 10,
        "duration_minutes": 481,
    })
    assert response.status_code == 422


# ── Batch endpoint ───────────────────────────────────────────────────────────


BATCH_PAYLOAD = {
    "meetings": [
        {"meeting_type": "standup",    "attendee_count": 8,  "duration_minutes": 15},
        {"meeting_type": "client",     "attendee_count": 12, "duration_minutes": 60},
        {"meeting_type": "one_on_one", "attendee_count": 2,  "duration_minutes": 30},
    ]
}


def test_batch_returns_200():
    response = client.post("/recommend/batch", json=BATCH_PAYLOAD)
    assert response.status_code == 200


def test_batch_returns_correct_count():
    response = client.post("/recommend/batch", json=BATCH_PAYLOAD)
    assert len(response.json()["results"]) == 3


def test_batch_preserves_order():
    response = client.post("/recommend/batch", json=BATCH_PAYLOAD)
    types = [r["meeting_type"] for r in response.json()["results"]]
    assert types == ["standup", "client", "one_on_one"]


def test_batch_all_valid_configs():
    response = client.post("/recommend/batch", json=BATCH_PAYLOAD)
    for result in response.json()["results"]:
        cfg = result["config"]
        assert cfg["recording_mode"] in VALID_RECORDING_MODES
        assert cfg["summary_depth"] in VALID_SUMMARY_DEPTHS
        assert cfg["bot_join_offset_seconds"] in VALID_OFFSETS
        assert 0.0 <= result["confidence"] <= 1.0


def test_batch_empty_list_rejected():
    response = client.post("/recommend/batch", json={"meetings": []})
    assert response.status_code == 422
