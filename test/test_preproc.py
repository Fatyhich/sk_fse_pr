import unittest
import numpy as np
from typing import Optional
from src.segmentation import _filter_points


class TestFilterPoints(unittest.TestCase):
    
    def test_none_points(self):
        """Test when points is None."""
        self.assertFalse(_filter_points(None))

    def test_points_below_threshold(self):
        """Test when some points are below the height threshold."""
        points = np.array([[100, 100], [200, 200], [300, 150]])
        self.assertFalse(_filter_points(points))

    def test_points_above_threshold(self):
        """Test when all points are above the height threshold."""
        points = np.array([[100, 400], [200, 500], [300, 450]])
        self.assertTrue(_filter_points(points))

    def test_mixed_points(self):
        """Test when some points are above and some are below the threshold."""
        points = np.array([[100, 200], [200, 400], [300, 100]])
        self.assertFalse(_filter_points(points))

    def test_empty_array(self):
        """Test when points is an empty array."""
        points = np.array([]).reshape(0, 2)  # No points
        self.assertTrue(_filter_points(points))  # No points should be considered as valid

if __name__ == "__main__":
    unittest.main()
