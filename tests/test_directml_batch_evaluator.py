import unittest

from ai.directml_batch_evaluator import DirectMLBatchEvaluator
from ai.gpu.backends.directml import SUPPORTED_TORCH_DIRECTML_TAGS, detect
from ai.python_compat import current_python_tag


class TestDirectMLBatchEvaluator(unittest.TestCase):
    def test_availability_check_returns_boolean(self):
        self.assertIsInstance(DirectMLBatchEvaluator.is_available(), bool)

    def test_detector_reports_unsupported_python_tag(self):
        if current_python_tag() in SUPPORTED_TORCH_DIRECTML_TAGS:
            self.skipTest("Current Python tag is supported by torch-directml.")

        result = detect()[0]

        self.assertFalse(result.available)
        self.assertIn(current_python_tag(), result.reason)


if __name__ == "__main__":
    unittest.main()
