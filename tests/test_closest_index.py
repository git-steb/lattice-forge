import unittest
import numpy as np
from LatticeForge import closestIndex

class TestClosestIndex(unittest.TestCase):
    def test_closest_index(self):
        R = np.array([[1, 0], [0, 1]])
        x = np.array([0.5, 0.5])
        uhat = closestIndex(R, x)
        expected = np.array([[0, 1]])
        np.testing.assert_array_almost_equal(uhat, expected, decimal=5)

if __name__ == "__main__":
    unittest.main()
