"""Convert a trained Keras model to fully quantized INT8 TensorFlow Lite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keras",
        type=Path,
        default=Path("models_real/edgeguard_1dcnn_cicids2017.keras"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed_real"))
    parser.add_argument("--out", type=Path, default=Path("models_real/edgeguard_int8.tflite"))
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")

    X_train = np.load(args.data_dir / "X_train.npy")
    if len(X_train) == 0:
        raise ValueError("X_train is empty")
    if X_train.ndim != 3 or not np.isfinite(X_train).all():
        raise ValueError(
            "X_train must be a finite float array shaped (rows, features, channels)"
        )
    model = tf.keras.models.load_model(args.keras, compile=False)

    sample_count = min(args.num_samples, len(X_train))
    indices = np.random.default_rng(args.seed).choice(
        len(X_train), size=sample_count, replace=False
    )
    representative_samples = X_train[indices].astype(np.float32, copy=False)

    def representative_dataset():
        for sample in representative_samples:
            yield [sample[np.newaxis, ...]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_content = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
        raise RuntimeError("Conversion did not produce INT8 input and output tensors")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(model_content)
    report = {
        "output": str(args.out),
        "representative_samples": int(sample_count),
        "input_shape": input_detail["shape"].tolist(),
        "input_dtype": str(input_detail["dtype"]),
        "input_quantization": list(input_detail["quantization"]),
        "output_shape": output_detail["shape"].tolist(),
        "output_dtype": str(output_detail["dtype"]),
        "output_quantization": list(output_detail["quantization"]),
        "bytes": len(model_content),
        "note": "INT8 TFLite conversion does not compile the model for Edge TPU.",
    }
    report_path = args.out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
