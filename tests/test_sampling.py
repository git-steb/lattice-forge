#%% tests/test_sampling.py
import unittest
import numpy as np
from LatticeForge.lattice_utils import normdet, striplattice, ndgridmat, rasterize, samplecube

class TestLatticeUtils(unittest.TestCase):
    def test_normdet(self):
        """
        Test that normdet normalizes the determinant of a matrix to the target value.
        """
        R = np.array([[2, 0], [0, 3]])
        normalized_R = normdet(R, target_det=1.0)
        self.assertAlmostEqual(np.abs(np.linalg.det(normalized_R)), 1.0, places=5)

    def test_ndgridmat(self):
        """
        Test ndgridmat to ensure it generates correct N-dimensional grids.
        """
        ranges = [np.array([0, 1]), np.array([0, 1])]
        expected = np.array([[0, 0, 1, 1],
                             [0, 1, 0, 1]])
        result = ndgridmat(ranges)
        np.testing.assert_array_equal(result, expected)

    def test_striplattice(self):
        """
        Test striplattice with a simple lattice basis.
        """
        R = np.array([[2, 1], [1, 1]], dtype=np.float64)
        reduced_R = striplattice(R)
        # Check that reduced_R is still a valid basis (full rank)
        self.assertEqual(np.linalg.matrix_rank(reduced_R), R.shape[0])
        # Check that the entries of reduced_R are not excessively large
        self.assertTrue(np.all(np.abs(reduced_R) <= 10))


        
# tests/test_sampling.py
import unittest
import numpy as np

# Import the package after installing it in editable mode
from LatticeForge.lattice_utils import rasterize, normdet

class TestRasterize(unittest.TestCase):
    
    def test_rasterize_identity(self):
        """
        Test rasterize with an identity matrix as the lattice basis.
        """
        R = np.eye(2)
        points = rasterize(R)
        expected_points = np.array([[0, 0],
                                    [0, 1],
                                    [1, 0],
                                    [1, 1]])
        self.assertTrue(np.array_equal(points, expected_points), "Rasterize did not produce expected points for identity matrix.")

class TestSampleCube(unittest.TestCase):
    def test_samplecube_identity(self):
        """
        Test samplecube with an identity matrix as the lattice basis.
        """
        R = np.eye(2)
        points = samplecube(R, 
                            epsdirub=1 # include points at the upper bound
                            )
        expected_points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        np.testing.assert_array_almost_equal(np.sort(points, axis=0), np.sort(expected_points, axis=0))


    def test_samplecube_shift(self):
        """
        Test samplecube with a shift applied to the lattice.
        """
        R = np.eye(2)
        xofs = np.array([0.25, 0.25])
        points = samplecube(R, xofs=xofs, eps=1e-6)  # Increase tolerance slightly
        expected_points = np.array([[0.25, 0.25]])
        np.testing.assert_array_almost_equal(np.sort(points, axis=0), np.sort(expected_points, axis=0))



if __name__ == "__main__":
    unittest.main()
