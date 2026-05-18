"""Inference utilities for the agent config recommender."""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.recommender.schemas import (
    AgentConfig,
    BatchRecommendRequest,
    BatchRecommendResponse,
    RecommendRequest,
    RecommendResponse,
)

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "config_recommender.joblib"

BOOL_FIELDS = {"summary_enabled", "action_items_enabled", "speaker_diarization"}


@lru_cache(maxsize=1)
def _load_models(path: str) -> dict[str, Pipeline]:
    return joblib.load(path)


def get_models() -> dict[str, Pipeline]:
    return _load_models(str(MODEL_PATH))


def _to_row(request: RecommendRequest) -> pd.DataFrame:
    return pd.DataFrame([{
        "meeting_type": request.meeting_type,
        "attendee_count": request.attendee_count,
        "duration_minutes": request.duration_minutes,
    }])


def recommend(request: RecommendRequest) -> RecommendResponse:
    """Recommend an agent config for a single meeting."""
    models = get_models()
    X = _to_row(request)

    config_raw: dict = {}
    confidences: list[float] = []

    for field, model in models.items():
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        confidences.append(float(proba.max()))
        config_raw[field] = bool(pred) if field in BOOL_FIELDS else pred

    config = AgentConfig(**config_raw)
    confidence = round(float(np.mean(confidences)), 4)

    return RecommendResponse(
        meeting_type=request.meeting_type,
        attendee_count=request.attendee_count,
        duration_minutes=request.duration_minutes,
        config=config,
        confidence=confidence,
    )


def recommend_batch(request: BatchRecommendRequest) -> BatchRecommendResponse:
    """Recommend agent configs for a list of meetings."""
    return BatchRecommendResponse(results=[recommend(m) for m in request.meetings])
