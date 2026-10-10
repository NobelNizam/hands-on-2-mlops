from __future__ import annotations

import os
from typing import Any

from src.config import ROOT, SPLIT_SEED

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from src.data import dataset_md5

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
SCORING = {
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def common_tags() -> dict[str, str]:
    return {
        "data_md5": dataset_md5(),
        "split_seed": str(SPLIT_SEED),
        "dataset": "sklearn-breast-cancer",
    }


def training_dataset(X, y):
    frame = X.assign(target=y.to_numpy())
    return mlflow.data.from_pandas(
        frame,
        source="data/train.csv",
        name="bc-train",
        targets="target",
    )


def run_experiment(run_name: str, pipe, X_train, y_train, tags=None, note: str = "") -> str:
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({**common_tags(), "candidate": "true", **(tags or {})})
        mlflow.set_tag("mlflow.note.content", note)
        mlflow.log_input(training_dataset(X_train, y_train), context="training")

        mlflow.log_param("pipeline_steps", " -> ".join(pipe.named_steps))
        classifier_params: dict[str, Any] = pipe.named_steps["clf"].get_params(deep=False)
        mlflow.log_params({f"clf__{key}": value for key, value in classifier_params.items()})

        scores = cross_validate(pipe, X_train, y_train, cv=CV, scoring=SCORING)
        for metric in SCORING:
            mlflow.log_metric(f"cv_{metric}_mean", scores[f"test_{metric}"].mean())
            mlflow.log_metric(f"cv_{metric}_std", scores[f"test_{metric}"].std())
        mlflow.log_metric("fit_time_s", scores["fit_time"].mean())

        y_oof = cross_val_predict(pipe, X_train, y_train, cv=CV)
        display = ConfusionMatrixDisplay.from_predictions(
            y_train,
            y_oof,
            display_labels=["benign", "malignant"],
        )
        mlflow.log_figure(display.figure_, "plots/confusion_matrix_oof.png")
        plt.close(display.figure_)
        mlflow.log_text(
            classification_report(y_train, y_oof, zero_division=0),
            "reports/classification_report_oof.txt",
        )

        pipe.fit(X_train, y_train)
        model_info = mlflow.sklearn.log_model(
            pipe,
            name="model",
            signature=infer_signature(X_train, pipe.predict(X_train)),
            input_example=X_train.head(3),
        )
        mlflow.set_tag("model_uri", model_info.model_uri)
        return run.info.run_id
