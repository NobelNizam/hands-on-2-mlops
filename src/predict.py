from __future__ import annotations

import mlflow

from src.config import MODEL_NAME, configure_mlflow
from src.data import load_split


def main() -> None:
    configure_mlflow()
    _, _, X_test, _ = load_split()
    model_uri = f"models:/{MODEL_NAME}@champion"
    model = mlflow.pyfunc.load_model(model_uri)
    predictions = model.predict(X_test.head(3))
    print(f"Model: {model_uri}")
    print(f"Prediksi (1 = malignant): {predictions.tolist()}")


if __name__ == "__main__":
    main()
