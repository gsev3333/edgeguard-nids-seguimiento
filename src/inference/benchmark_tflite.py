"""Measure binary metrics and per-record latency on CPU or Coral Edge TPU."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tflite", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed_real"))
    parser.add_argument("--out-dir", type=Path, default=Path("evidencias/reales"))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "coral"), default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def create_interpreter(model_path: Path, device: str):
    if device == "coral":
        if not model_path.name.endswith("_edgetpu.tflite"):
            raise ValueError(
                "For --device coral, provide the compiled *_edgetpu.tflite model"
            )
        try:
            from pycoral.utils.edgetpu import make_interpreter
        except ImportError as error:
            raise RuntimeError(
                "Coral mode requires the PyCoral runtime on the target device"
            ) from error
        return make_interpreter(str(model_path))

    import tensorflow as tf

    return tf.lite.Interpreter(model_path=str(model_path))


def quantize_input(values: np.ndarray, detail: dict) -> np.ndarray:
    dtype = detail["dtype"]
    scale, zero_point = detail["quantization"]
    if np.issubdtype(dtype, np.integer):
        if scale <= 0:
            raise ValueError("Integer model input has an invalid quantization scale")
        limits = np.iinfo(dtype)
        values = np.rint(values / scale + zero_point)
        values = np.clip(values, limits.min, limits.max)
    return values.astype(dtype)


def dequantize_output(values: np.ndarray, detail: dict) -> np.ndarray:
    dtype = detail["dtype"]
    scale, zero_point = detail["quantization"]
    values = values.astype(np.float32)
    if np.issubdtype(dtype, np.integer):
        if scale <= 0:
            raise ValueError("Integer model output has an invalid quantization scale")
        values = (values - zero_point) * scale
    return values


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.warmup < 0:
        raise ValueError("--samples must be positive and --warmup cannot be negative")
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")
    if not args.tflite.is_file():
        raise FileNotFoundError(args.tflite)

    X_test = np.load(args.data_dir / "X_test.npy")
    y_test = np.load(args.data_dir / "y_test.npy").astype(np.int32).ravel()
    if len(X_test) == 0:
        raise ValueError("X_test is empty")
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test row counts do not match")
    if not set(np.unique(y_test)).issubset({0, 1}):
        raise ValueError("Expected binary labels 0 (BENIGN) and 1 (ATTACK)")

    count = min(args.samples, len(X_test))
    indices = np.random.default_rng(args.seed).choice(
        len(X_test), size=count, replace=False
    )
    X_sample = X_test[indices].astype(np.float32, copy=False)
    y_sample = y_test[indices]

    interpreter = create_interpreter(args.tflite, args.device)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if tuple(input_detail["shape"][1:]) != tuple(X_sample.shape[1:]):
        raise ValueError(
            f"Model expects {input_detail['shape'].tolist()}, data has "
            f"{[1, *X_sample.shape[1:]]}"
        )

    def invoke_one(sample: np.ndarray) -> float:
        quantized = quantize_input(sample[np.newaxis, ...], input_detail)
        interpreter.set_tensor(input_detail["index"], quantized)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"])
        probability = dequantize_output(raw, output_detail).reshape(-1)[0]
        return float(np.clip(probability, 0.0, 1.0))

    warmup_count = min(args.warmup, count)
    for index in range(warmup_count):
        invoke_one(X_sample[index])

    latencies_ms: list[float] = []
    probabilities = np.empty(count, dtype=np.float32)
    for index, sample in enumerate(X_sample):
        start = time.perf_counter()
        probabilities[index] = invoke_one(sample)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    predictions = (probabilities >= args.threshold).astype(np.int32)
    elapsed_seconds = sum(latencies_ms) / 1000.0
    matrix = confusion_matrix(y_sample, predictions, labels=[0, 1])
    result = {
        "device": "coral_edge_tpu" if args.device == "coral" else "cpu",
        "model": str(args.tflite),
        "samples": int(count),
        "warmup": int(warmup_count),
        "seed": args.seed,
        "threshold": args.threshold,
        "class_order": ["BENIGN", "ATTACK"],
        "accuracy": float(accuracy_score(y_sample, predictions)),
        "precision": float(precision_score(y_sample, predictions, zero_division=0)),
        "recall": float(recall_score(y_sample, predictions, zero_division=0)),
        "f1": float(f1_score(y_sample, predictions, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_sample, probabilities))
            if np.unique(y_sample).size == 2
            else None
        ),
        "latency_ms_mean": float(np.mean(latencies_ms)),
        "latency_ms_median": float(np.median(latencies_ms)),
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)),
        "latency_ms_std": float(np.std(latencies_ms)),
        "throughput_inferences_per_second": (
            float(count / elapsed_seconds) if elapsed_seconds > 0 else None
        ),
        "confusion_matrix": matrix.tolist(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_stem = f"benchmark_{args.device}_results"
    (args.out_dir / f"{result_stem}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    throughput = result["throughput_inferences_per_second"]
    throughput_text = f"{throughput:.2f}" if throughput is not None else "n/a"
    report = [
        f"EdgeGuard benchmark ({result['device']})",
        f"Model: {args.tflite}",
        f"Samples: {count}; warm-up: {warmup_count}; threshold: {args.threshold}",
        f"Accuracy: {result['accuracy']:.6f}",
        f"Precision: {result['precision']:.6f}",
        f"Recall: {result['recall']:.6f}",
        f"F1: {result['f1']:.6f}",
        f"ROC AUC: {result['roc_auc']}",
        f"Mean latency (ms): {result['latency_ms_mean']:.4f}",
        f"Median latency (ms): {result['latency_ms_median']:.4f}",
        f"P95 latency (ms): {result['latency_ms_p95']:.4f}",
        f"Throughput (inferences/s): {throughput_text}",
        f"Confusion matrix [BENIGN, ATTACK]: {matrix.tolist()}",
        f"Platform: {result['platform']}",
    ]
    (args.out_dir / f"{result_stem}.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
