"""Inference utilities for the speaker intelligence module."""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.speaker.analytics import compute
from src.speaker.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
)

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "engagement_classifier.joblib"
FEATURE_COLS = ["dominance_index", "max_talk_ratio", "speaker_count", "silence_ratio", "cv_talk_ratio"]


@lru_cache(maxsize=1)
def _load_model(path: str) -> RandomForestClassifier:
    return joblib.load(path)


def get_model() -> RandomForestClassifier:
    return _load_model(str(MODEL_PATH))


def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Compute speaker analytics and classify meeting engagement."""
    model = get_model()
    result = compute(request)

    features = result["_features"]
    X = pd.DataFrame([features])[FEATURE_COLS]
    engagement = str(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    confidence = round(float(proba.max()), 4)

    return AnalyzeResponse(
        meeting_id=request.meeting_id,
        duration_seconds=request.duration_seconds,
        speaker_count=result["speaker_count"],
        speaker_stats=result["speaker_stats"],
        silence_ratio=result["silence_ratio"],
        interruption_count=result["interruption_count"],
        dominance_index=result["dominance_index"],
        engagement=engagement,
        engagement_confidence=confidence,
    )


def analyze_batch(request: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    """Analyze speaker timelines for a list of meetings."""
    return BatchAnalyzeResponse(results=[analyze(m) for m in request.meetings])
