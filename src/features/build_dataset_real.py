"""Prepare CIC-IDS-2017 MachineLearningCSV files for binary classification."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--input-dir", type=Path, default=None)
    sources.add_argument("--input-zip", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_real"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-features", type=int, default=78)
    return parser.parse_args()


def read_frames_from_directory(input_dir: Path) -> tuple[list[pd.DataFrame], list[str]]:
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir.resolve()}")

    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        print(f"Reading {csv_path.name}")
        frame = pd.read_csv(csv_path, low_memory=False)
        frame.columns = [str(column).strip() for column in frame.columns]
        frames.append(frame)
    return frames, [path.name for path in csv_files]


def read_frames_from_zip(input_zip: Path) -> tuple[list[pd.DataFrame], list[str]]:
    if not input_zip.is_file():
        raise FileNotFoundError(input_zip)
    with zipfile.ZipFile(input_zip) as archive:
        csv_names = sorted(
            name for name in archive.namelist() if name.lower().endswith(".csv")
        )
        if not csv_names:
            raise ValueError(f"No CSV files found in {input_zip}")

        frames: list[pd.DataFrame] = []
        for name in csv_names:
            print(f"Reading {name}")
            with archive.open(name) as csv_stream:
                frame = pd.read_csv(csv_stream, low_memory=False)
            frame.columns = [str(column).strip() for column in frame.columns]
            frames.append(frame)
    return frames, csv_names


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1")

    if args.input_zip:
        frames, input_files = read_frames_from_zip(args.input_zip)
    else:
        input_dir = args.input_dir or Path("data/raw/cicids2017")
        frames, input_files = read_frames_from_directory(input_dir)

    # Fail as a whole if any input cannot be read. Silently skipping a file
    # would change the dataset and make its class distribution misleading.
    df = pd.concat(frames, ignore_index=True, sort=False)
    label_columns = [column for column in df.columns if column.casefold() == "label"]
    if len(label_columns) != 1:
        raise ValueError(f"Expected one Label column; found {label_columns}")
    label_column = label_columns[0]

    labels = df[label_column].astype("string").str.strip()
    valid_labels = labels.notna() & labels.ne("")
    df = df.loc[valid_labels].copy()
    labels = labels.loc[valid_labels]
    raw_label_counts = {
        str(label): int(count)
        for label, count in labels.str.upper().value_counts().items()
    }
    y = labels.str.upper().ne("BENIGN").astype(np.int32).to_numpy()

    candidate_columns = [column for column in df.columns if column != label_column]
    numeric = df[candidate_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    # Remove metadata columns such as IP strings and timestamps.
    numeric = numeric.dropna(axis=1, how="all")
    valid_rows = numeric.notna().all(axis=1).to_numpy()
    numeric = numeric.loc[valid_rows]
    y = y[valid_rows]

    if numeric.shape[1] != args.expected_features:
        raise ValueError(
            f"Expected {args.expected_features} numeric features, found "
            f"{numeric.shape[1]}: {list(numeric.columns)}"
        )
    if np.unique(y).size != 2:
        raise ValueError("The cleaned dataset must contain BENIGN and ATTACK rows")

    X = numeric.to_numpy(dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32).reshape(
        -1, numeric.shape[1], 1
    )
    X_test = scaler.transform(X_test).astype(np.float32).reshape(
        -1, numeric.shape[1], 1
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "X_train.npy": X_train,
        "X_test.npy": X_test,
        "y_train.npy": y_train.astype(np.int32),
        "y_test.npy": y_test.astype(np.int32),
    }
    for filename, array in arrays.items():
        np.save(args.output_dir / filename, array)

    joblib.dump(scaler, args.output_dir / "scaler.joblib")
    (args.output_dir / "feature_names.txt").write_text(
        "\n".join(map(str, numeric.columns)) + "\n", encoding="utf-8"
    )

    summary = {
        "input_files": input_files,
        "rows_after_cleaning": int(len(y)),
        "feature_count": int(numeric.shape[1]),
        "features": list(map(str, numeric.columns)),
        "raw_label_counts_before_feature_cleanup": raw_label_counts,
        "test_size": args.test_size,
        "seed": args.seed,
        "label_mapping": {"BENIGN": 0, "all_other_nonempty_labels": 1},
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "train_class_counts": {
            "BENIGN": int(np.sum(y_train == 0)),
            "ATTACK": int(np.sum(y_train == 1)),
        },
        "test_class_counts": {
            "BENIGN": int(np.sum(y_test == 0)),
            "ATTACK": int(np.sum(y_test == 1)),
        },
    }
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
