"""Fast checks for the model shape and TFLite input/output conversion."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "model"))
sys.path.insert(0, str(ROOT / "src" / "inference"))

from architecture import build_1d_cnn  # noqa: E402
from benchmark_tflite import dequantize_output, quantize_input  # noqa: E402


class TFLiteConversionHelpersTest(unittest.TestCase):
    def test_int8_input_uses_scale_and_zero_point(self) -> None:
        detail = {"dtype": np.dtype(np.int8), "quantization": (0.1, -2)}
        values = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)

        actual = quantize_input(values, detail)

        np.testing.assert_array_equal(actual, np.array([[-12, -2, 8]], dtype=np.int8))

    def test_int8_input_is_clipped_to_supported_range(self) -> None:
        detail = {"dtype": np.dtype(np.int8), "quantization": (0.1, 0)}
        values = np.array([[-100.0, 100.0]], dtype=np.float32)

        actual = quantize_input(values, detail)

        np.testing.assert_array_equal(actual, np.array([[-128, 127]], dtype=np.int8))

    def test_integer_input_rejects_zero_scale(self) -> None:
        detail = {"dtype": np.dtype(np.int8), "quantization": (0.0, 0)}
        with self.assertRaisesRegex(ValueError, "scale"):
            quantize_input(np.array([[0.0]], dtype=np.float32), detail)

    def test_int8_output_returns_probability_values(self) -> None:
        detail = {"dtype": np.dtype(np.int8), "quantization": (0.1, -2)}
        values = np.array([[-12, -2, 8]], dtype=np.int8)

        actual = dequantize_output(values, detail)

        np.testing.assert_allclose(actual, np.array([[-1.0, 0.0, 1.0]], dtype=np.float32))

    def test_integer_output_rejects_zero_scale(self) -> None:
        detail = {"dtype": np.dtype(np.int8), "quantization": (0.0, 0)}
        with self.assertRaisesRegex(ValueError, "scale"):
            dequantize_output(np.array([[0]], dtype=np.int8), detail)


class ModelSmokeTest(unittest.TestCase):
    def test_model_returns_one_probability_per_flow(self) -> None:
        tf.keras.utils.set_random_seed(42)
        model = build_1d_cnn(input_shape=(78, 1))

        probabilities = model(np.zeros((2, 78, 1), dtype=np.float32), training=False).numpy()

        self.assertEqual(probabilities.shape, (2, 1))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
