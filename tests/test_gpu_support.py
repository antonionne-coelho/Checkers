import unittest

from ai.gpu import detect_gpu_environment, validate_gpu_workers_request


class TestGPUSupport(unittest.TestCase):
    def test_zero_gpu_workers_is_valid(self):
        validate_gpu_workers_request(0)

    def test_positive_gpu_workers_fails_until_backend_is_implemented(self):
        with self.assertRaises(RuntimeError):
            validate_gpu_workers_request(1)

    def test_detect_gpu_environment_returns_backend_matrix(self):
        environment = detect_gpu_environment()
        self.assertIsInstance(environment.vendors, list)
        self.assertGreaterEqual(len(environment.backends), 1)
        self.assertTrue(all(backend.name for backend in environment.backends))

    def test_detect_gpu_environment_includes_onnx_directml(self):
        environment = detect_gpu_environment()
        backend_names = {backend.name for backend in environment.backends}

        self.assertIn("onnx-directml", backend_names)


if __name__ == "__main__":
    unittest.main()
