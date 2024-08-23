import unittest
import numpy as np
from LatticeForge import closestIndex

class TestClosestIndex(unittest.TestCase):
    def test_closest_index_default(self):
        R = np.array([[1, 0], [0, 1]])
        x = np.array([0.5, 0.5])
        
        # Case 1: Default behaviour, should use epsilon=1e-8, allnn=True
        uhat = closestIndex(R, x)
        expected = np.array([[0, 1]])
        np.testing.assert_array_almost_equal(uhat, expected, decimal=5)
    
    def test_closest_index_explicit_epsilon_zero(self):
        R = np.array([[1, 0], [0, 1]])
        x = np.array([0.5, 0.5])
        
        # Case 2: Explicit epsilon = 0, allnn = False
        uhat = closestIndex(R, x, allnn=False, epsilon=0)
        expected = np.array([[0, 1]])
        np.testing.assert_array_almost_equal(uhat, expected, decimal=5)
    
    def test_closest_index_epsilon_point_five(self):
        R = np.array([[1, 0], [0, 1]])
        x = np.array([0.5, 0.5])
        
        # Case 3: epsilon = 0.5, allnn = False
        uhat = closestIndex(R, x, allnn=False, epsilon=0.5)
        expected = np.array([[0, 1]])
        np.testing.assert_array_almost_equal(uhat, expected, decimal=5)
    
    def test_closest_index_allnn_true_epsilon_small(self):
        R = np.array([[1, 0], [0, 1]])
        x = np.array([0.5, 0.5])
        
        # Case 4: epsilon = 1e-8, allnn = True
        uhat = closestIndex(R, x, allnn=True, epsilon=1e-8)
        expected = np.array([[0, 1]])
        np.testing.assert_array_almost_equal(uhat, expected, decimal=5)
    
    def test_closest_index_allnn_true_epsilon_large(self):
        R = np.array([[1, 0], [0, 1]])
        x = np.array([0.5, 0.5])
        
        # Case 5: epsilon = 1, allnn = True
        uhat = closestIndex(R, x, allnn=True, epsilon=1)
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_array_almost_equal(uhat, expected, decimal=5)
    
    def test_closest_index_epsilon_zero_non_standard_lattice(self):
        R = np.array([[1, 0.5], [0.5, 1]])
        x = np.array([0.5, 0.5])
        
        # Case 6: Edge case with epsilon = 0, non-standard lattice
        uhat = closestIndex(R, x, allnn=False, epsilon=0)
        expected = np.array([[0, 1]])
        np.testing.assert_array_almost_equal(uhat, expected, decimal=5)

if __name__ == "__main__":
    unittest.main()
