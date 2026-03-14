"""Machine-learning utilities for conjunction-risk screening."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


FEATURE_COLUMNS = [
    "relative_position_x",
    "relative_position_y",
    "relative_position_z",
    "relative_velocity_x",
    "relative_velocity_y",
    "relative_velocity_z",
    "time_to_conjunction",
    "sat_altitude",
    "debris_altitude",
]


def load_dataset(csv_path: str | Path) -> list[dict]:
    """Load the conjunction dataset from CSV."""

    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _feature_matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[float(row[column]) for column in FEATURE_COLUMNS] for row in rows],
        dtype=float,
    )


def _label_vector(rows: list[dict], column: str) -> np.ndarray:
    return np.asarray([float(row[column]) for row in rows], dtype=float)


def dataset_summary(rows: list[dict]) -> dict:
    """Summarize class balance and target coverage."""

    collision_risk = _label_vector(rows, "collision_risk").astype(int)
    required_delta_v = _label_vector(rows, "required_delta_v")
    return {
        "rows": int(len(rows)),
        "positive_collision_risk": int(collision_risk.sum()),
        "negative_collision_risk": int((collision_risk == 0).sum()),
        "positive_fraction": float(collision_risk.mean()) if len(collision_risk) else 0.0,
        "nonzero_delta_v": int(np.count_nonzero(required_delta_v > 0.0)),
    }


def train_risk_classifier(
    rows: list[dict],
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict:
    """Train a binary classifier for collision-risk screening."""

    X = _feature_matrix(rows)
    y = _label_vector(rows, "collision_risk").astype(int)
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError("Collision-risk labels must contain at least two classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=12,
                    min_samples_leaf=2,
                    random_state=random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    return {
        "model": model,
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_score)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        },
        "feature_columns": FEATURE_COLUMNS,
    }


def train_delta_v_regressor(
    rows: list[dict],
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict:
    """Train a regressor to estimate required delta-v."""

    filtered_rows = [row for row in rows if float(row["required_delta_v"]) > 0.0]
    if len(filtered_rows) < 20:
        raise ValueError(
            "Need at least 20 nonzero-delta-v samples to train the regressor."
        )

    X = _feature_matrix(filtered_rows)
    y = _label_vector(filtered_rows, "required_delta_v")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=12,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": model,
        "metrics": {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "mean_target": float(np.mean(y_test)),
        },
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": len(filtered_rows),
    }


def benchmark_inference(model: Pipeline, rows: list[dict], n_runs: int = 100) -> dict:
    """Estimate average per-run inference time in milliseconds."""

    X = _feature_matrix(rows)
    start = time.perf_counter()
    for _ in range(n_runs):
        model.predict(X)
    elapsed = time.perf_counter() - start
    return {
        "samples": len(rows),
        "runs": n_runs,
        "average_inference_ms": float(1000.0 * elapsed / n_runs),
    }


def save_model(model: Pipeline, output_path: str | Path) -> None:
    """Serialize a trained sklearn pipeline."""

    joblib.dump(model, output_path)


def load_model(model_path: str | Path) -> Pipeline:
    """Load a serialized sklearn pipeline."""

    return joblib.load(model_path)
