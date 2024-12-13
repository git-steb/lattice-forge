import unittest
import numpy as np
from LatticeForge import closestIndex
from LatticeForge import samplecube, normdet

def sorted_rows(a):
    """
    Sort the rows of a 2D or 3D array lexicographically for easy comparison.
    """
    if a.size == 0:
        return a
    sort_keys = tuple(a[:, i] for i in range(a.shape[1]-1, -1, -1))
    return a[np.lexsort(sort_keys)]

class TestCartesianLattices(unittest.TestCase):
    def test_2d_single_neighbor(self):
        """
        Test a 2D Cartesian lattice (identity basis) with a point closer to [1,0].
        """
        R = np.eye(2)
        x = np.array([0.9, 0.1])
        uhat = closestIndex(R, x, allnn=False, epsilon=0)
        expected = np.array([[1, 0]])
        np.testing.assert_array_almost_equal(uhat, expected, decimal=5)

    def test_2d_multiple_neighbors(self):
        """
        Test a 2D Cartesian lattice (identity basis) with a point equidistant from multiple neighbors.
        Here, x = (0.5, 0.5) should be equally close to [0,0], [0,1], [1,0], [1,1].
        Using allnn=True and a small epsilon, we should get all four points.
        """
        R = np.eye(2)  # Identity matrix
        x = np.array([0.5, 0.5])
        uhat = closestIndex(R, x, allnn=True, epsilon=0.1)
        expected = np.array([[0,0], [0,1], [1,0], [1,1]])

    def test_3d_single_neighbor(self):
        """
        Test a 3D Cartesian lattice (identity basis) with a point closer to [1,0,1].
        """
        R = np.eye(3)
        x = np.array([0.9, 0.1, 1.4])
        uhat = closestIndex(R, x, allnn=False, epsilon=0)
        expected = np.array([[1, 0, 1]])
        np.testing.assert_array_almost_equal(uhat, expected, decimal=5)

    def test_3d_multiple_neighbors(self):
        """
        Test a 3D Cartesian lattice (identity basis) with a point equidistant from multiple neighbors.
        x = (0.5, 0.5, 0.5) is equally close to all 2^3 = 8 points in the cube [0,1]^3.
        With allnn=True and a small epsilon, we should get all these 8 neighbors.
        """
        R = np.eye(3)  # Identity matrix
        x = np.array([0.5, 0.5, 0.5])
        neighbors = np.array([[i, j, k] for i in [0,1] for j in [0,1] for k in [0,1]])
        uhat = closestIndex(R, x, allnn=True, epsilon=0.1)
                
        np.testing.assert_array_almost_equal(sorted_rows(uhat), sorted_rows(neighbors), decimal=5)
    
    def test_sheared_unimodal_basis_single_neighbor(self):
        """
        Test a 2D sheared unimodal lattice basis:
        R = [[1, 2],
            [0, 1]]

        For u = [2,1], R*u = [2 + 2*1, 1] = [4, 1].
        We'll choose x = [4.001, 1.001], which should be closest to [2,1].
        """
        R = np.array([[1, 2],
                      [0, 1]])
        x = np.array([4.001, 1.001])
        uhat = closestIndex(R, x, allnn=False, epsilon=0, debug=True)  # Enable debugging
        expected = np.array([[2, 1]])
        np.testing.assert_array_almost_equal(uhat, expected, decimal=5)

if __name__ == "__main__":
    unittest.main()
