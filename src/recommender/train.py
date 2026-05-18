"""
Train the agent config recommender and save the model artifact.

Generates synthetic (meeting_type, attendee_count, duration_minutes) samples
with ground-truth config labels derived from domain rules + noise, then fits
one GradientBoostingClassifier per config field.

Usage:
    python -m src.recommender.train
    python -m src.recommender.train --samples 300 --out models/config_recommender.joblib
"""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DEFAULT_OUT = Path(__file__).parent.parent.parent / "models" / "config_recommender.joblib"
RNG_SEED = 42

MEETING_TYPES = [
    "standup", "planning", "one_on_one", "client",
    "all_hands", "interview", "workshop", "social",
]

# Domain-derived base configs per meeting type
_BASE = {
    "standup":    dict(summary_enabled=0, action_items_enabled=0, recording_mode="audio_only",  summary_depth="brief",    bot_join_offset_seconds=0),
    "planning":   dict(summary_enabled=1, action_items_enabled=1, recording_mode="audio_only",  summary_depth="detailed", bot_join_offset_seconds=-60),
    "one_on_one": dict(summary_enabled=1, action_items_enabled=1, recording_mode="audio_only",  summary_depth="standard", bot_join_offset_seconds=0),
    "client":     dict(summary_enabled=1, action_items_enabled=1, recording_mode="audio_video", summary_depth="detailed", bot_join_offset_seconds=-120),
    "all_hands":  dict(summary_enabled=1, action_items_enabled=0, recording_mode="audio_video", summary_depth="standard", bot_join_offset_seconds=-120),
    "interview":  dict(summary_enabled=1, action_items_enabled=0, recording_mode="audio_video", summary_depth="detailed", bot_join_offset_seconds=-60),
    "workshop":   dict(summary_enabled=1, action_items_enabled=1, recording_mode="audio_only",  summary_depth="detailed", bot_join_offset_seconds=-60),
    "social":     dict(summary_enabled=0, action_items_enabled=0, recording_mode="audio_only",  summary_depth="brief",    bot_join_offset_seconds=0),
}

_TYPE_PARAMS = {
    "standup":    dict(att=(4, 12),   dur_choices=[15, 20, 30]),
    "planning":   dict(att=(4, 10),   dur_choices=[60, 75, 90]),
    "one_on_one": dict(att=(2, 3),    dur_choices=[30, 45, 60]),
    "client":     dict(att=(4, 20),   dur_choices=[60, 75, 90]),
    "all_hands":  dict(att=(20, 200), dur_choices=[60, 75, 90]),
    "interview":  dict(att=(2, 5),    dur_choices=[45, 60]),
    "workshop":   dict(att=(5, 30),   dur_choices=[60, 90, 120, 180]),
    "social":     dict(att=(5, 50),   dur_choices=[30, 45, 60]),
}

TARGET_FIELDS = [
    "summary_enabled",
    "action_items_enabled",
    "speaker_diarization",
    "recording_mode",
    "summary_depth",
    "bot_join_offset_seconds",
]


def _generate_data(samples_per_type: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for mtype in MEETING_TYPES:
        params = _TYPE_PARAMS[mtype]
        for _ in range(samples_per_type):
            att = int(rng.integers(params["att"][0], params["att"][1] + 1))
            dur = int(rng.choice(params["dur_choices"]))

            cfg = dict(_BASE[mtype])

            cfg["speaker_diarization"] = 1 if att >= 3 else 0
            if att >= 10:
                cfg["recording_mode"] = "audio_video"

            if dur >= 90 and cfg["summary_depth"] == "standard":
                cfg["summary_depth"] = "detailed"
            if dur <= 20 and cfg["summary_depth"] == "standard":
                cfg["summary_depth"] = "brief"

            # label noise (5% flip on boolean fields)
            for field in ("summary_enabled", "action_items_enabled", "speaker_diarization"):
                if rng.random() < 0.05:
                    cfg[field] = 1 - cfg[field]

            rows.append({"meeting_type": mtype, "attendee_count": att, "duration_minutes": dur, **cfg})

    return pd.DataFrame(rows)


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("type_ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["meeting_type"]),
            ("nums", "passthrough", ["attendee_count", "duration_minutes"]),
        ]
    )


def _make_clf() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RNG_SEED)


def train(samples_per_type: int = 300, out_path: Path = DEFAULT_OUT) -> None:
    rng = np.random.default_rng(RNG_SEED)
    df = _generate_data(samples_per_type, rng)

    print(f"Dataset: {len(df)} samples  ({samples_per_type} per type × {len(MEETING_TYPES)} types)\n")

    X = df[["meeting_type", "attendee_count", "duration_minutes"]]
    X_train, X_test, df_train, df_test = train_test_split(X, df, test_size=0.2, random_state=RNG_SEED)

    models: dict[str, Pipeline] = {}

    for field in TARGET_FIELDS:
        y_train = df_train[field].values
        y_test_field = df_test[field].values

        pipe = Pipeline([
            ("prep", _make_preprocessor()),
            ("clf", _make_clf()),
        ])

        cv_acc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test_field, y_pred)

        print(f"  [{field}]  CV acc: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}  |  test acc: {acc:.3f}")
        models[field] = pipe

    os.makedirs(out_path.parent, exist_ok=True)
    joblib.dump(models, out_path)
    print(f"\nModels saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    train(args.samples, args.out)
