"""
Train the bot demand forecaster and save the model artifact.

Generates synthetic calendar data for 90 days, computes concurrent
bot demand per slot, and fits a GradientBoostingRegressor on temporal features.

Usage:
    python -m src.forecaster.train
    python -m src.forecaster.train --days 90 --out models/demand_forecaster.joblib
"""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

DEFAULT_OUT = Path(__file__).parent.parent.parent / "models" / "demand_forecaster.joblib"
GRANULARITY_MINUTES = 60
RNG_SEED = 42


def _generate_events(n_days: int, rng: np.random.Generator) -> pd.DataFrame:
    """Produce synthetic calendar events mimicking a ~200-person company."""
    base = datetime(2025, 1, 6)  # Monday
    rows = []
    for day_offset in range(n_days):
        dt = base + timedelta(days=day_offset)
        dow = dt.weekday()

        if dow >= 5:
            n_meetings = int(rng.poisson(2))
        elif dow == 4:
            n_meetings = int(rng.poisson(28))
        else:
            n_meetings = int(rng.poisson(48))

        for _ in range(n_meetings):
            # bimodal start-time distribution: morning 9-11, afternoon 1-4
            if rng.random() < 0.55:
                hour = rng.integers(9, 12)
            else:
                hour = rng.integers(13, 17)
            minute = int(rng.choice([0, 15, 30, 45]))
            duration = int(rng.choice([30, 60, 90], p=[0.35, 0.50, 0.15]))

            start = dt.replace(hour=int(hour), minute=minute, second=0, microsecond=0)
            end = start + timedelta(minutes=duration)
            rows.append({"start": start, "end": end})

    return pd.DataFrame(rows)


def _compute_demand(events: pd.DataFrame, base: datetime, n_days: int) -> pd.DataFrame:
    """Count concurrent events for each hourly slot."""
    slots = []
    for day_offset in range(n_days):
        for hour in range(24):
            slot_start = base + timedelta(days=day_offset, hours=hour)
            slot_end = slot_start + timedelta(minutes=GRANULARITY_MINUTES)
            # overlap: event starts before slot ends AND ends after slot starts
            count = int(((events["start"] < slot_end) & (events["end"] > slot_start)).sum())
            slots.append({"slot_start": slot_start, "count": count})
    return pd.DataFrame(slots)


def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive cyclically-encoded temporal features from slot_start."""
    ts = df["slot_start"]
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    features = pd.DataFrame(
        {
            "hour": hour,
            "dow": dow,
            "is_weekend": (dow >= 5).astype(int),
            "is_working_hours": ((hour >= 8) & (hour < 18)).astype(int),
            # cyclical encodings prevent ordinal discontinuity
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
        }
    )
    return features


def train(n_days: int = 90, out_path: Path = DEFAULT_OUT) -> None:
    """Generate synthetic data, train, evaluate, and save the demand forecaster."""
    rng = np.random.default_rng(RNG_SEED)
    base = datetime(2025, 1, 6)

    print(f"Generating {n_days} days of synthetic calendar events...")
    events = _generate_events(n_days, rng)
    print(f"  {len(events)} events generated")

    demand = _compute_demand(events, base, n_days)
    X = _extract_features(demand)
    y = demand["count"].values

    print(f"\nDemand stats: mean={y.mean():.1f}, max={y.max()}, "
          f"non-zero slots={int((y > 0).sum())}/{len(y)}")

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RNG_SEED,
    )

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    print(f"\n5-fold CV MAE: {-cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    model.fit(X, y)
    y_pred = model.predict(X)
    print(f"Train MAE: {mean_absolute_error(y, y_pred):.3f}  R²: {r2_score(y, y_pred):.4f}")

    os.makedirs(out_path.parent, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"\nModel saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    train(args.days, args.out)
