from __future__ import annotations

import json
import os

from src.config import ROOT

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import EXPERIMENT_NAME, MIN_PRECISION, TEST_PATH, configure_mlflow
from src.data import load_split


def select_best_candidate():
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=(
            f"tags.candidate = 'true' and metrics.cv_precision_mean >= {MIN_PRECISION}"
        ),
        order_by=["metrics.cv_recall_mean DESC", "metrics.cv_f1_mean DESC"],
    )
    if runs.empty:
        raise RuntimeError("Tidak ada kandidat yang memenuhi precision >= 0.90. Jalankan training dahulu.")
    return runs.iloc[0]


def main() -> None:
    configure_mlflow()
    _, _, X_test, y_test = load_split()
    selected = select_best_candidate()
    run_id = selected["run_id"]

    if selected.get("tags.test_evaluated") == "true":
        print(f"Run {run_id} sudah pernah dievaluasi pada test set; evaluasi tidak diulang.")
        return

    model_uri = selected["tags.model_uri"]
    model = mlflow.sklearn.load_model(model_uri)
    predictions = model.predict(X_test)
    scores = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )
    metrics = {
        "test_recall": recall_score(y_test, predictions),
        "test_precision": precision_score(y_test, predictions, zero_division=0),
        "test_f1": f1_score(y_test, predictions, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, scores),
    }

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        mlflow.log_input(
            mlflow.data.from_pandas(
                X_test.assign(target=y_test.to_numpy()),
                source=str(TEST_PATH.relative_to(ROOT)),
                name="bc-test",
                targets="target",
            ),
            context="testing",
        )
        display = ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            display_labels=["benign", "malignant"],
        )
        mlflow.log_figure(display.figure_, "plots/confusion_matrix_test.png")
        plt.close(display.figure_)
        mlflow.set_tags({"test_evaluated": "true", "selected_candidate": "true"})

    (ROOT / "selected_run.json").write_text(
        json.dumps({"run_id": run_id, "model_uri": model_uri, "metrics": metrics}, indent=2),
        encoding="utf-8",
    )
    print(f"Kandidat terpilih: {selected['tags.mlflow.runName']} ({run_id})")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
