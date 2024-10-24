import unittest
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch
from typing import Callable, Any, List

from src.canguro_processing_tools.utils.parallel_utils import do_parallel

class TestDoParallel(unittest.TestCase):

    def test_zero_workers(self):
        """Test when n_workers is 0."""
        def task_fn(x):
            return x * 2
        arguments = [1, 2, 3]
        expected_result = [2, 4, 6]
        result = do_parallel(task_fn, arguments, 0, False)
        self.assertEqual(result, expected_result)

    def test_positive_workers_without_tqdm(self):
        """Test when n_workers is positive and use_tqdm is False."""
        def task_fn(x):
            return x * 2
        arguments = [1, 2, 3]
        expected_result = [2, 4, 6]
        with patch('concurrent.futures.ProcessPoolExecutor.map') as mock_map:
            mock_map.return_value = iter(expected_result)
            result = do_parallel(task_fn, arguments, 2, False)
            self.assertEqual(result, expected_result)
            mock_map.assert_called_once()

    def test_invalid_n_workers(self):
        """Test when n_workers is invalid."""
        def task_fn(x):
            return x * 2
        arguments = [1, 2, 3]
        with self.assertRaises(AssertionError):
            do_parallel(task_fn, arguments, -1, False)

    def test_empty_arguments(self):
        """Test when arguments list is empty."""
        def task_fn(x):
            return x * 2
        arguments = []
        expected_result = []
        result = do_parallel(task_fn, arguments, 2, False)
        self.assertEqual(result, expected_result)

if __name__ == "__main__":
    unittest.main()
