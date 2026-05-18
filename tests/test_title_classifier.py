"""Tests for the meeting title classifier API endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_LABELS = {
    "standup", "planning", "one_on_one", "client",
    "all_hands", "interview", "workshop", "social",
}

GOLDEN = [
    ("Eng Standup", "standup"),
    ("Daily Scrum - Backend", "standup"),
    ("Sprint Planning Q2", "planning"),
    ("Alice <> Bob", "one_on_one"),
    ("1:1 Alice / Bob", "one_on_one"),
    ("Acme QBR", "client"),
    ("Company All Hands", "all_hands"),
    ("Q3 All-Hands", "all_hands"),
    ("Julia Smith Interview", "interview"),
    ("SWE Interview: Jordan Lee", "interview"),
    ("Design Workshop", "workshop"),
    ("Team Lunch", "social"),
    ("Virtual Happy Hour", "social"),
]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classify_returns_valid_label():
    response = client.post("/classify", json={"title": "Eng Standup"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in VALID_LABELS
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["title"] == "Eng Standup"


@pytest.mark.parametrize("title,expected", GOLDEN)
def test_golden_examples(title: str, expected: str):
    response = client.post("/classify", json={"title": title})
    assert response.status_code == 200
    assert response.json()["label"] == expected, (
        f"'{title}' → got '{response.json()['label']}', expected '{expected}'"
    )


def test_batch_classify():
    titles = ["Eng Standup", "Sprint Planning", "Alice <> Bob"]
    response = client.post("/classify/batch", json={"titles": titles})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 3
    for r in results:
        assert r["label"] in VALID_LABELS
        assert 0.0 <= r["confidence"] <= 1.0


def test_batch_preserves_order():
    titles = ["Team Lunch", "Julia Smith Interview", "Company All Hands"]
    response = client.post("/classify/batch", json={"titles": titles})
    results = response.json()["results"]
    assert [r["title"] for r in results] == titles


def test_classify_empty_title_rejected():
    response = client.post("/classify", json={"title": ""})
    assert response.status_code == 422


def test_classify_title_too_long_rejected():
    response = client.post("/classify", json={"title": "x" * 201})
    assert response.status_code == 422


def test_batch_empty_list_rejected():
    response = client.post("/classify/batch", json={"titles": []})
    assert response.status_code == 422
