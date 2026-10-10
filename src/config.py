from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "bc-malignant-detection"
MODEL_NAME = "bc-malignant-classifier"
SPLIT_SEED = 42
MIN_PRECISION = 0.90
PROMOTION_MARGIN = 0.005


def configure_mlflow() -> None:
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
