import unittest
from unittest.mock import patch

from ai.batch_evaluator import PythonBatchEvaluator
from ai.batch_evaluator_factory import (
    BatchEvaluatorUnavailable,
    create_batch_evaluator,
)
from ai.gpu.backends.base import GPUBackendInfo
from ai.gpu.detector import GPUDetectionResult
from ai.vector_batch_evaluator import VectorizedCPUBatchEvaluator


class TestBatchEvaluatorFactory(unittest.TestCase):
    def test_auto_without_gpu_returns_python_evaluator(self):
        evaluator = create_batch_evaluator(backend="auto", gpu_enabled=False)

        self.assertIsInstance(evaluator, PythonBatchEvaluator)

    def test_python_backend_can_be_requested_explicitly(self):
        evaluator = create_batch_evaluator(backend="python", gpu_enabled=True)

        self.assertIsInstance(evaluator, PythonBatchEvaluator)

    def test_cpu_vector_backend_can_be_requested_explicitly(self):
        evaluator = create_batch_evaluator(backend="cpu-vector", gpu_enabled=False)

        self.assertIsInstance(evaluator, VectorizedCPUBatchEvaluator)

    def test_unknown_backend_raises_value_error(self):
        with self.assertRaises(ValueError):
            create_batch_evaluator(backend="unknown", gpu_enabled=False)

    def test_unavailable_gpu_backend_raises_when_strict(self):
        with self.assertRaises(BatchEvaluatorUnavailable) as context:
            create_batch_evaluator(backend="opencl", gpu_enabled=True)

        self.assertIn("Detalhes:", str(context.exception))

    def test_directml_message_mentions_python_compatibility_when_relevant(self):
        with self.assertRaises(BatchEvaluatorUnavailable) as context:
            create_batch_evaluator(backend="directml", gpu_enabled=True)

        self.assertIn("torch-directml", str(context.exception))

    def test_unavailable_gpu_backend_can_fallback_to_python(self):
        evaluator = create_batch_evaluator(
            backend="opencl",
            gpu_enabled=True,
            strict_gpu=False,
        )

        self.assertIsInstance(evaluator, PythonBatchEvaluator)

    def test_available_directml_backend_is_created(self):
        environment = GPUDetectionResult(
            vendors=["amd"],
            backends=[
                GPUBackendInfo(
                    name="torch-directml",
                    vendor="cross-vendor",
                    framework="torch-directml",
                    available=True,
                    reason="test",
                )
            ],
        )
        sentinel = object()

        with patch(
            "ai.batch_evaluator_factory.detect_gpu_environment",
            return_value=environment,
        ), patch(
            "ai.batch_evaluator_factory.DirectMLBatchEvaluator",
            return_value=sentinel,
        ):
            evaluator = create_batch_evaluator(
                backend="directml",
                gpu_enabled=True,
            )

        self.assertIs(evaluator, sentinel)

    def test_onnx_directml_backend_reports_not_implemented_when_available(self):
        environment = GPUDetectionResult(
            vendors=["amd"],
            backends=[
                GPUBackendInfo(
                    name="onnx-directml",
                    vendor="cross-vendor",
                    framework="onnxruntime-directml",
                    available=True,
                    reason="test",
                )
            ],
        )

        with patch(
            "ai.batch_evaluator_factory.detect_gpu_environment",
            return_value=environment,
        ), self.assertRaises(BatchEvaluatorUnavailable) as context:
            create_batch_evaluator(
                backend="onnx-directml",
                gpu_enabled=True,
            )

        self.assertIn("onnx-directml", str(context.exception))


if __name__ == "__main__":
    unittest.main()
