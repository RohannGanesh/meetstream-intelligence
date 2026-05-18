"""Inference utilities for the meeting title classifier."""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from src.classifiers.schemas import ClassifyResponse

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "title_classifier.joblib"


@lru_cache(maxsize=1)
def _load_model(path: str) -> Pipeline:
    return joblib.load(path)


def get_model() -> Pipeline:
    return _load_model(str(MODEL_PATH))


def classify(title: str) -> ClassifyResponse:
    """Classify a single meeting title and return the predicted label with confidence."""
    model = get_model()
    proba = model.predict_proba([title])[0]
    idx = int(np.argmax(proba))
    label = model.classes_[idx]
    return ClassifyResponse(title=title, label=label, confidence=round(float(proba[idx]), 4))


def classify_batch(titles: list[str]) -> list[ClassifyResponse]:
    """Classify a list of meeting titles."""
    model = get_model()
    probas = model.predict_proba(titles)
    results = []
    for title, proba in zip(titles, probas):
        idx = int(np.argmax(proba))
        results.append(ClassifyResponse(
            title=title,
            label=model.classes_[idx],
            confidence=round(float(proba[idx]), 4),
        ))
    return results
