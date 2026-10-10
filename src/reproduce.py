from __future__ import annotations

from pathlib import Path

import mlflow
from mlflow import MlflowClient

from src.config import MODEL_NAME, configure_mlflow
from src.data import dataset_md5


def main() -> None:
    configure_mlflow()
    client = MlflowClient()
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    if not champion.run_id:
        raise RuntimeError("Champion tidak memiliki lineage run.")
    run = client.get_run(champion.run_id)

    current_hash = dataset_md5()
    recorded_hash = run.data.tags.get("data_md5")
    if current_hash != recorded_hash:
        raise RuntimeError(f"Data berbeda: current={current_hash}, recorded={recorded_hash}")

    print(f"Run        : {run.info.run_name} ({run.info.run_id})")
    print(f"Git commit : {run.data.tags.get('mlflow.source.git.commit', 'tidak tersedia')}")
    print(f"Data MD5   : {recorded_hash}")
    print(f"Params     : {run.data.params}")
    print(f"Metrics    : {run.data.metrics}")

    model_path = Path(
        mlflow.artifacts.download_artifacts(run_id=champion.run_id, artifact_path="model")
    )
    requirements = next(model_path.rglob("requirements.txt"), None)
    print(f"Artifact   : {model_path}")
    if requirements:
        print("Environment:")
        print(requirements.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
