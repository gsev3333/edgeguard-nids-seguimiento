"""Choose an operating threshold on the training validation split, then evaluate once on test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keras", type=Path, default=Path("models_real/edgeguard_1dcnn_cicids2017.keras"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed_real"))
    parser.add_argument("--output", type=Path, default=Path("models_real/threshold_calibration_real.json"))
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--target-attack-recall", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")
    if not 0 < args.target_attack_recall <= 1:
        raise ValueError("--target-attack-recall must be in (0, 1]")

    X_train = np.load(args.data_dir / "X_train.npy")
    y_train = np.load(args.data_dir / "y_train.npy").astype(np.int32).ravel()
    X_test = np.load(args.data_dir / "X_test.npy")
    y_test = np.load(args.data_dir / "y_test.npy").astype(np.int32).ravel()
    validation_count = int(len(X_train) * args.validation_fraction)
    if validation_count < 1:
        raise ValueError("Training set is too small for the requested validation split")

    # Keras validation_split reserves the last fraction of array inputs before shuffling.
    X_validation = X_train[-validation_count:]
    y_validation = y_train[-validation_count:]
    model = tf.keras.models.load_model(args.keras, compile=False)
    validation_probabilities = model.predict(
        X_validation, batch_size=args.batch_size, verbose=0
    ).ravel()

    precision, recall, thresholds = precision_recall_curve(
        y_validation, validation_probabilities
    )
    eligible = np.flatnonzero(recall[:-1] >= args.target_attack_recall)
    if len(eligible) == 0:
        raise RuntimeError("No validation threshold reaches the requested attack recall")
    validation_f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], np.finfo(float).eps)
    )
    best_index = max(
        eligible,
        key=lambda index: (validation_f1[index], precision[index], thresholds[index]),
    )
    threshold = float(thresholds[best_index])

    test_probabilities = model.predict(X_test, batch_size=args.batch_size, verbose=0).ravel()
    test_predictions = (test_probabilities >= threshold).astype(np.int32)
    baseline_predictions = (test_probabilities >= 0.5).astype(np.int32)
    result = {
        "threshold_selection": "maximum validation F1 subject to the target attack recall",
        "validation_fraction": args.validation_fraction,
        "validation_samples": int(validation_count),
        "target_attack_recall": args.target_attack_recall,
        "threshold": threshold,
        "validation_precision": float(precision[best_index]),
        "validation_attack_recall": float(recall[best_index]),
        "validation_f1": float(validation_f1[best_index]),
        "test_samples": int(len(y_test)),
        "test": {
            "classification_report": classification_report(
                y_test,
                test_predictions,
                labels=[0, 1],
                target_names=["BENIGN", "ATTACK"],
                output_dict=True,
                zero_division=0,
            ),
            "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
            "confusion_matrix": confusion_matrix(
                y_test, test_predictions, labels=[0, 1]
            ).tolist(),
        },
        "test_baseline_threshold_0_5": {
            "attack_f1": float(f1_score(y_test, baseline_predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(
                y_test, baseline_predictions, labels=[0, 1]
            ).tolist(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
