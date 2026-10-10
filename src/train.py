from __future__ import annotations

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import configure_mlflow
from src.data import load_split, prepare_data
from src.tracking import CV, SCORING, common_tags, run_experiment, training_dataset


def run_baselines(X_train, y_train) -> None:
    run_experiment(
        "00-dummy-most-frequent",
        Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
        X_train,
        y_train,
        tags={"stage": "baseline", "model_family": "dummy"},
        note="Batas bawah: selalu menebak kelas mayoritas (benign).",
    )
    run_experiment(
        "01-logreg-noscale",
        Pipeline([("clf", LogisticRegression(max_iter=5000, random_state=42))]),
        X_train,
        y_train,
        tags={"stage": "baseline", "model_family": "logreg"},
        note="H1: model linear tanpa preprocessing.",
    )
    run_experiment(
        "02-logreg-scaled",
        Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, random_state=42))]
        ),
        X_train,
        y_train,
        tags={"stage": "baseline", "model_family": "logreg"},
        note="H2: hanya menambah StandardScaler; variabel lain tetap.",
    )


def run_model_selection(X_train, y_train) -> None:
    candidates = {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42),
        "gb": GradientBoostingClassifier(random_state=42),
        "svc": SVC(kernel="rbf", random_state=42),
    }
    for index, (family, classifier) in enumerate(candidates.items(), start=3):
        run_experiment(
            f"{index:02d}-{family}-default",
            Pipeline([("scaler", StandardScaler()), ("clf", classifier)]),
            X_train,
            y_train,
            tags={"stage": "model-selection", "model_family": family},
            note="Hyperparameter default; preprocessing dan fold identik untuk semua model.",
        )


def run_svc_tuning(X_train, y_train) -> str:
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", random_state=42))])
    search_space = {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001]}
    search = GridSearchCV(
        pipe,
        search_space,
        cv=CV,
        scoring=SCORING,
        refit="recall",
        n_jobs=-1,
    )

    with mlflow.start_run(
        run_name="10-svc-gridsearch",
        tags={
            **common_tags(),
            "stage": "tuning",
            "model_family": "svc",
            "candidate": "true",
        },
    ) as run:
        mlflow.log_input(training_dataset(X_train, y_train), context="training")
        mlflow.log_dict(search_space, "search_space.json")
        search.fit(X_train, y_train)
        results = search.cv_results_

        for index, params in enumerate(results["params"]):
            with mlflow.start_run(run_name=f"svc-trial-{index:02d}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metrics(
                    {
                        f"cv_{metric}_mean": results[f"mean_test_{metric}"][index]
                        for metric in SCORING
                    }
                )
                mlflow.log_metrics(
                    {
                        f"cv_{metric}_std": results[f"std_test_{metric}"][index]
                        for metric in SCORING
                    }
                )

        best = search.best_index_
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(
            {f"cv_{metric}_mean": results[f"mean_test_{metric}"][best] for metric in SCORING}
        )
        mlflow.log_metrics(
            {f"cv_{metric}_std": results[f"std_test_{metric}"][best] for metric in SCORING}
        )
        model_info = mlflow.sklearn.log_model(
            search.best_estimator_,
            name="model",
            signature=infer_signature(X_train, search.best_estimator_.predict(X_train)),
            input_example=X_train.head(3),
        )
        mlflow.set_tag("model_uri", model_info.model_uri)
        return run.info.run_id


def main() -> None:
    prepare_data()
    configure_mlflow()
    X_train, y_train, _, _ = load_split()
    run_baselines(X_train, y_train)
    run_model_selection(X_train, y_train)
    tuning_run = run_svc_tuning(X_train, y_train)
    print(f"Training selesai. Run tuning: {tuning_run}")


if __name__ == "__main__":
    main()
