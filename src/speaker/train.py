"""
Train the meeting engagement classifier and save the model artifact.

Generates synthetic speaker-stat feature vectors across three engagement
patterns, then fits a RandomForestClassifier.

Usage:
    python -m src.speaker.train
    python -m src.speaker.train --samples 400 --out models/engagement_classifier.joblib
"""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split

DEFAULT_OUT = Path(__file__).parent.parent.parent / "models" / "engagement_classifier.joblib"
RNG_SEED = 42
FEATURE_COLS = ["dominance_index", "max_talk_ratio", "speaker_count", "silence_ratio", "cv_talk_ratio"]
LABELS = ["balanced", "presenter_led", "dominated"]


def _gen_balanced(n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        n_sp = int(rng.integers(3, 7))
        alpha = np.ones(n_sp) * rng.uniform(2.5, 6.0)
        silence = rng.uniform(0.05, 0.20)
        ratios = rng.dirichlet(alpha) * (1 - silence)
        rows.append({
            "dominance_index": float(np.sum(ratios ** 2)),
            "max_talk_ratio": float(ratios.max()),
            "speaker_count": float(n_sp),
            "silence_ratio": float(silence),
            "cv_talk_ratio": float(ratios.std() / ratios.mean()) if ratios.mean() > 0 else 0.0,
            "label": "balanced",
        })
    return pd.DataFrame(rows)


def _gen_presenter_led(n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        n_sp = int(rng.integers(3, 7))
        silence = rng.uniform(0.05, 0.25)
        presenter = rng.uniform(0.50, 0.75)
        remaining = max(0.0, 1 - presenter - silence)
        others = rng.dirichlet(np.ones(n_sp - 1)) * remaining
        ratios = np.concatenate([[presenter], others])
        rows.append({
            "dominance_index": float(np.sum(ratios ** 2)),
            "max_talk_ratio": float(ratios.max()),
            "speaker_count": float(n_sp),
            "silence_ratio": float(silence),
            "cv_talk_ratio": float(ratios.std() / ratios.mean()) if ratios.mean() > 0 else 0.0,
            "label": "presenter_led",
        })
    return pd.DataFrame(rows)


def _gen_dominated(n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        n_sp = int(rng.integers(2, 5))
        silence = rng.uniform(0.01, 0.15)
        dominator = rng.uniform(0.75, 0.95)
        remaining = max(0.0, 1 - dominator - silence)
        if n_sp > 1:
            others = rng.dirichlet(np.ones(n_sp - 1)) * remaining
        else:
            others = np.array([])
        ratios = np.concatenate([[dominator], others]) if n_sp > 1 else np.array([dominator])
        rows.append({
            "dominance_index": float(np.sum(ratios ** 2)),
            "max_talk_ratio": float(ratios.max()),
            "speaker_count": float(n_sp),
            "silence_ratio": float(silence),
            "cv_talk_ratio": float(ratios.std() / ratios.mean()) if ratios.mean() > 0 else 0.0,
            "label": "dominated",
        })
    return pd.DataFrame(rows)


def train(samples_per_class: int = 400, out_path: Path = DEFAULT_OUT) -> None:
    rng = np.random.default_rng(RNG_SEED)

    df = pd.concat([
        _gen_balanced(samples_per_class, rng),
        _gen_presenter_led(samples_per_class, rng),
        _gen_dominated(samples_per_class, rng),
    ], ignore_index=True).sample(frac=1, random_state=RNG_SEED)

    print(f"Dataset: {len(df)} samples  ({samples_per_class} per class × 3 classes)")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

    X, y = df[FEATURE_COLS], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RNG_SEED)

    model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RNG_SEED, n_jobs=-1)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
    print(f"5-fold CV macro-F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nHeld-out test set:")
    print(classification_report(y_test, y_pred, target_names=LABELS))

    print("Feature importances:")
    for feat, imp in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:<22} {imp:.4f}  {'█' * int(imp * 50)}")

    os.makedirs(out_path.parent, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"\nModel saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    train(args.samples, args.out)
