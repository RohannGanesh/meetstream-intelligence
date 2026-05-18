"""
Train the meeting title classifier and save the model artifact.

Usage:
    python -m src.classifiers.train
    python -m src.classifiers.train --data data/synthetic/meeting_titles.csv --out models/title_classifier.joblib
"""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_DATA = Path(__file__).parent.parent.parent / "data" / "synthetic" / "meeting_titles.csv"
DEFAULT_OUT = Path(__file__).parent.parent.parent / "models" / "title_classifier.joblib"


def build_pipeline() -> Pipeline:
    """Return an untrained TF-IDF + LogisticRegression pipeline."""
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=2,
            sublinear_tf=True,
        )),
        # char n-grams capture patterns like "1:1", "<>", "dsu", "l&l"
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=3,
            sublinear_tf=True,
        )),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])


def train(data_path: Path = DEFAULT_DATA, out_path: Path = DEFAULT_OUT) -> None:
    """Train, evaluate, and save the classifier."""
    df = pd.read_csv(data_path)
    X, y = df["title"].tolist(), df["label"].tolist()

    print(f"Dataset: {len(X)} samples, {len(set(y))} classes")
    print(f"Label distribution:\n{pd.Series(y).value_counts().to_string()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"5-fold CV macro-F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("\nHeld-out test set:")
    print(classification_report(y_test, y_pred))

    os.makedirs(out_path.parent, exist_ok=True)
    joblib.dump(pipeline, out_path)
    print(f"Model saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    train(args.data, args.out)
