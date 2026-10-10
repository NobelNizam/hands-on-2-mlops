from __future__ import annotations

import argparse
import hashlib

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, SPLIT_SEED, TEST_PATH, TRAIN_PATH


def source_dataframe() -> pd.DataFrame:
    frame = load_breast_cancer(as_frame=True).frame.copy()
    frame["target"] = 1 - frame["target"]  # 1 = malignant
    return frame


def dataset_md5(frame: pd.DataFrame | None = None) -> str:
    frame = source_dataframe() if frame is None else frame
    values = pd.util.hash_pandas_object(frame, index=True).values
    return hashlib.md5(values.tobytes()).hexdigest()


def expected_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = source_dataframe()
    return train_test_split(
        frame,
        test_size=0.2,
        stratify=frame["target"],
        random_state=SPLIT_SEED,
    )


def prepare_data(force: bool = False) -> str:
    if force or not (TRAIN_PATH.exists() and TEST_PATH.exists()):
        train_df, test_df = expected_split()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(TRAIN_PATH, index=False)
        test_df.to_csv(TEST_PATH, index=False)
    return dataset_md5()


def load_split() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if not (TRAIN_PATH.exists() and TEST_PATH.exists()):
        raise FileNotFoundError("Data belum dibuat. Jalankan: python -m src.data")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return (
        train_df.drop(columns="target"),
        train_df["target"],
        test_df.drop(columns="target"),
        test_df["target"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="L1: split tetap dan hash dataset")
    parser.add_argument("--force", action="store_true", help="Tulis ulang split deterministik")
    args = parser.parse_args()
    digest = prepare_data(force=args.force)
    X_train, _, X_test, _ = load_split()
    print(f"Train: {len(X_train)} baris -> {TRAIN_PATH}")
    print(f"Test : {len(X_test)} baris -> {TEST_PATH}")
    print(f"Data MD5: {digest}")


if __name__ == "__main__":
    main()
