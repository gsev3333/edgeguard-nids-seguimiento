"""Train and evaluate the EdgeGuard CNN on processed CIC-IDS-2017 arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from architecture import build_1d_cnn, compile_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed_real"))
    parser.add_argument("--output-dir", type=Path, default=Path("models_real"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")

    tf.keras.utils.set_random_seed(args.seed)
    X_train = np.load(args.data_dir / "X_train.npy")
    X_test = np.load(args.data_dir / "X_test.npy")
    y_train = np.load(args.data_dir / "y_train.npy").astype(np.int32).ravel()
    y_test = np.load(args.data_dir / "y_test.npy").astype(np.int32).ravel()

    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError("Expected feature arrays with shape (rows, features, channels)")
    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError("Train and test feature shapes do not match")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and label row counts do not match")
    if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_test)) != {0, 1}:
        raise ValueError("Both train and test sets must contain labels 0 and 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = compile_model(build_1d_cnn(input_shape=X_train.shape[1:]))
    classes = np.array([0, 1], dtype=np.int32)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(label): float(weight) for label, weight in zip(classes, weights)}
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    probabilities = model.predict(X_test, batch_size=1024, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(np.int32)
    report_dict = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history.epoch),
        "batch_size": args.batch_size,
        "threshold": 0.5,
        "class_weight": class_weight,
        "test_samples": int(len(y_test)),
        "classification_report": report_dict,
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    }

    (args.output_dir / "classification_report_real.txt").write_text(
        report_text, encoding="utf-8"
    )
    (args.output_dir / "training_metrics_real.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["BENIGN", "ATTACK"],
        yticklabels=["BENIGN", "ATTACK"],
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("EdgeGuard — CIC-IDS-2017")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix_real.png", dpi=150)
    plt.close()
    model.save(args.output_dir / "edgeguard_1dcnn_cicids2017.keras")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
