from __future__ import annotations

from src.config import (
    EXPERIMENT_NAME,
    MIN_PRECISION,
    MODEL_NAME,
    PROMOTION_MARGIN,
    configure_mlflow,
)


def passes_quality_gate(test_precision: float, test_recall: float, champion_recall: float) -> bool:
    return test_precision >= MIN_PRECISION and test_recall >= champion_recall + PROMOTION_MARGIN


def evaluated_candidate():
    import mlflow

    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="tags.selected_candidate = 'true' and tags.test_evaluated = 'true'",
        order_by=["metrics.cv_recall_mean DESC", "metrics.cv_f1_mean DESC"],
    )
    if runs.empty:
        raise RuntimeError("Belum ada kandidat yang dievaluasi. Jalankan: python -m src.evaluate")
    return runs.iloc[0]


def champion_context(client) -> tuple[float, str | None]:
    from mlflow.exceptions import MlflowException

    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        return -1.0, None

    if not champion.run_id:
        raise RuntimeError("Champion tidak memiliki lineage run.")
    run = client.get_run(champion.run_id)
    return run.data.metrics["test_recall"], run.data.tags.get("data_md5")


def main() -> None:
    import mlflow
    from mlflow import MlflowClient

    configure_mlflow()
    selected = evaluated_candidate()
    client = MlflowClient()
    champion_recall, champion_data_md5 = champion_context(client)
    challenger_data_md5 = selected["tags.data_md5"]
    if champion_data_md5 and champion_data_md5 != challenger_data_md5:
        raise RuntimeError("Hash data champion berbeda; quality gate tidak dapat dibandingkan secara adil.")

    model_version = mlflow.register_model(selected["tags.model_uri"], MODEL_NAME)
    description = (
        "Dipilih otomatis: CV recall maksimum dengan precision >= 0.90. "
        f"data_md5={challenger_data_md5}"
    )
    client.update_model_version(MODEL_NAME, model_version.version, description=description)
    client.set_model_version_tag(MODEL_NAME, model_version.version, "data_md5", challenger_data_md5)
    client.set_registered_model_alias(MODEL_NAME, "challenger", model_version.version)

    approved = passes_quality_gate(
        selected["metrics.test_precision"],
        selected["metrics.test_recall"],
        champion_recall,
    )
    client.set_model_version_tag(
        MODEL_NAME,
        model_version.version,
        "validation_status",
        "approved" if approved else "rejected",
    )
    if approved:
        client.set_registered_model_alias(MODEL_NAME, "champion", model_version.version)

    print(f"Model {MODEL_NAME} v{model_version.version}: {'approved' if approved else 'rejected'}")
    print("Alias challenger diperbarui; alias champion " + ("dipromosikan." if approved else "tetap."))


if __name__ == "__main__":
    main()
