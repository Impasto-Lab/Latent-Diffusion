"""Device-policy tests that do not require physical accelerators."""
import unittest
from unittest.mock import patch

from utils.environment import (
    select_device,
    select_precision,
)


class EnvironmentTests(unittest.TestCase):
    @patch("utils.environment._mps_available", return_value=True)
    @patch("utils.environment.torch.cuda.is_available", return_value=True)
    @patch("utils.environment.torch.cuda.device_count", return_value=2)
    def test_auto_prefers_cuda_and_accepts_numbered_gpu(self, _count, _cuda, _mps):
        self.assertEqual(select_device("auto"), "cuda:0")
        self.assertEqual(select_device("cuda"), "cuda:0")
        self.assertEqual(select_device("CUDA:1"), "cuda:1")

    @patch("utils.environment._mps_available", return_value=True)
    @patch("utils.environment.torch.cuda.is_available", return_value=False)
    def test_auto_falls_back_to_mps_then_cpu(self, _cuda, _mps):
        self.assertEqual(select_device("auto"), "mps")
        _mps.return_value = False
        self.assertEqual(select_device("auto"), "cpu")

    @patch("utils.environment.torch.cuda.is_available", return_value=True)
    @patch("utils.environment.torch.cuda.device_count", return_value=1)
    def test_cuda_index_is_validated(self, _count, _available):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            select_device("cuda:1")

    @patch("utils.environment.torch.cuda.is_available", return_value=False)
    def test_explicit_cuda_requires_cuda_pytorch(self, _available):
        with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
            select_device("cuda")

    def test_auto_precision_matches_backend(self):
        self.assertEqual(select_precision("auto", "cuda:0"), "float16")
        self.assertEqual(select_precision("auto", "mps"), "float16")
        self.assertEqual(select_precision("auto", "cpu"), "float32")
        with self.assertRaisesRegex(ValueError, "CPU inference"):
            select_precision("float16", "cpu")
        with self.assertRaisesRegex(ValueError, "with MPS"):
            select_precision("bfloat16", "mps")


if __name__ == "__main__":
    unittest.main()
