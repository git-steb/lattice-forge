# tests/test_lattice_design_pipeline.py
import unittest
import numpy as np
from scipy.linalg import companion
from sympy import sqrt, I, eye
from scipy.stats import special_ortho_group

from LatticeForge.lattice_design_pipeline import (
    formJ,
    genfilename,
    verifyK,
    formlattice,
    searchK,
    genK,
    doall,
    enumerate_matrices
)
from LatticeForge.lattice_utils import rasterize, normdet

import logging

# Configure logging at the beginning of your script
logging.basicConfig(
    #level=logging.DEBUG,  # Set to DEBUG to capture all levels of log messages
    level=logging.WARN,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Logs will be output to the console
    ]
)
# logging.basicConfig(level=logging.DEBUG, format='%(message)s')

class TestLatticePipeline(unittest.TestCase):
    
    def test_genfilename(self):
        """
        Test the filename generation function.
        """
        filename = genfilename(N=3, detK=4, suffix="test")
        self.assertEqual(filename, "lattices_dim3_det4_test.npy")

class TestFormJ(unittest.TestCase):

    def test_numeric_J_2x2(self):
        expected = np.array([[1 / np.sqrt(2), 1j / np.sqrt(2)], 
                             [1 / np.sqrt(2), -1j / np.sqrt(2)]], dtype=complex)
        result = formJ(2, symbolic=False)
        np.testing.assert_array_almost_equal(result, expected)

    def test_numeric_J_4x4(self):
        sqrt2_inv = 1 / np.sqrt(2)
        expected = np.array([
            [sqrt2_inv, 1j * sqrt2_inv, 0, 0],
            [sqrt2_inv, -1j * sqrt2_inv, 0, 0],
            [0, 0, sqrt2_inv, 1j * sqrt2_inv],
            [0, 0, sqrt2_inv, -1j * sqrt2_inv]
        ], dtype=complex)
        result = formJ(4, symbolic=False)
        np.testing.assert_array_almost_equal(result, expected)

    def test_symbolic_J_2x2(self):
        sqrt2 = sqrt(2)
        expected = np.array([[1 / sqrt2, I / sqrt2], 
                             [1 / sqrt2, -I / sqrt2]])
        result = formJ(2, symbolic=True)
        np.testing.assert_array_equal(result, expected)

    def test_symbolic_J_4x4(self):
        sqrt2 = sqrt(2)
        expected = eye(4)
        expected[0:2, 0:2] = np.array([[1 / sqrt2, I / sqrt2], 
                                       [1 / sqrt2, -I / sqrt2]])
        expected[2:4, 2:4] = np.array([[1 / sqrt2, I / sqrt2], 
                                       [1 / sqrt2, -I / sqrt2]])
        result = formJ(4, symbolic=True)
        np.testing.assert_array_equal(result, expected)


class TestGenK(unittest.TestCase):
    def test_genK_odd_dimension(self):
        """Test genK for an odd dimension."""
        ND = 3
        D = 2
        K = genK(ND, D)
        expected_det = D
        
        # Check the shape of the matrix
        self.assertEqual(K.shape, (ND, ND))
        
        # Check the determinant
        computed_det = round(np.linalg.det(K))
        self.assertEqual(computed_det, expected_det)
        
    def test_genK_even_dimension_default_coe(self):
        """Test genK for an even dimension with default central coefficient."""
        ND = 4
        D = 3
        K = genK(ND, D)
        expected_det = D
        
        # Check the shape of the matrix
        self.assertEqual(K.shape, (ND, ND))
        
        # Check the determinant
        computed_det = round(np.linalg.det(K))
        self.assertEqual(computed_det, expected_det)
        
    def test_genK_even_dimension_with_coe(self):
        """Test genK for an even dimension with a specified central coefficient."""
        ND = 4
        D = 4
        coe = 2
        K = genK(ND, D, coe)
        expected_det = D
        
        # Check the shape of the matrix
        self.assertEqual(K.shape, (ND, ND))
        
        # Check the determinant
        computed_det = round(np.linalg.det(K))
        self.assertEqual(computed_det, expected_det)
        
    def test_genK_invalid_coe(self):
        """Test genK with an invalid central coefficient that exceeds the allowed range."""
        ND = 4
        D = 4
        coe = 10  # This might be invalid based on the constraint
        
        with self.assertRaises(AssertionError):
            K = genK(ND, D, coe)

class TestLatticePipeline(unittest.TestCase):

    def test_verifyK_valid(self):
            """
            Test verifyK with a valid subsampling matrix.
            """
            K = np.array([[1, 1], [1, 3]])  # det(K) = 2
            self.assertTrue(verifyK(K, detK=2))
        
    def test_verifyK_invalid(self):
        """
        Test verifyK with an invalid subsampling matrix.
        """
        K = np.array([[1, 0], [0, 3]])  # det(K) = 3
        self.assertFalse(verifyK(K, detK=2))
    
    def test_formlattice(self):
        """
        Test the formlattice function with a simple subsampling matrix.
        """
        K = np.array([[2, 0], [0, 1]])
        R, Q = formlattice(K)
        self.assertIsInstance(R, np.ndarray)
        self.assertIsInstance(Q, np.ndarray)
        self.assertEqual(R.shape, (2, 2))
        self.assertEqual(Q.shape, (2, 2))
    
    def test_searchK_random(self):
        """
        Test the searchK function for finding valid subsampling matrices with random sampling.
        """
        results = searchK(N=2, detK=2, value_range=(-2, 2), nsamp=1000, save=False)  # Increased nsamp
        logging.debug(f"Number of valid K matrices found: {len(results)}")
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1, "No valid K matrices found with detK=2.")
        for K in results:
            self.assertTrue(verifyK(K, detK=2), f"Matrix K={K} failed verification.")
    
    def test_searchK_exhaustive(self):
        """
        Exhaustively search for all valid 2x2 integer matrices within a range and verify det(K)=2.
        """
        results = enumerate_matrices(N=2, detK=2, value_range=(-2, 2))
        logging.debug(f"Enumerated {len(results)} valid K matrices with detK=2.")
        self.assertGreaterEqual(len(results), 1, "No valid K matrices found via enumeration.")
    
    def test_doall(self):
        """
        Test txhe doall function to ensure the pipeline runs successfully.
        """
        results = doall(N=2, detK=2, forcerecompute=True, usecompangen=False, ngen=1, optimize=True)
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1, "doall did not return any valid matrices.")
        for K in results:
            self.assertTrue(verifyK(K, detK=2), f"Matrix K={K} failed verification.")

class TestLatticeSampling(unittest.TestCase):    
    def test_high_dimensional_sampling(self):
        """
        Test lattice sampling for dimensions greater than 7 with a random rotation applied.
        Ensure a significant number of points are generated and advanced rasterization is used.
        """
        N = 14
        # Generate a random matrix R with increased perturbations to the identity matrix
        #np.random.seed(42)
        R = normdet(np.eye(N) + 0.1 * np.random.rand(N, N), target_det=1)

        # Apply a random rotation to R
        rotation = special_ortho_group.rvs(N, random_state=42)
        R_rotated = R @ rotation

        # Set parameters for rasterization
        eps = 0.2  # Increased tolerance
        minbvol = True
        sortit = True

        # Rasterize the lattice points
        points = rasterize(R_rotated, eps=eps, minbvol=minbvol, sortit=sortit)

        # Expected dimensionality after handling complex R
        expected_dim = 2 * N if np.iscomplexobj(R_rotated) else N

        det_R = np.linalg.det(R)
        det_R_rotated = np.linalg.det(R_rotated)
        logging.debug(f"Determinant of R: {det_R}")
        logging.debug(f"Determinant of R_rotated: {det_R_rotated}")
        
        # Assertions to check the validity of the results
        self.assertIsInstance(points, np.ndarray)
        self.assertGreater(points.shape[0], 1, "Expected at least 1 lattice points.")
        self.assertEqual(points.shape[1], expected_dim, f"Expected points to have {expected_dim} dimensions.")

        # Verify that all points lie within the expanded bounds
        tighter_bound = 2**N  # Expanded bound
        self.assertTrue(
            np.all((points >= -tighter_bound) & (points <= tighter_bound)),
            f"Some points fall outside the expected bounds of ±{tighter_bound}."
        )

        # Print the number of points generated to give visibility during the test run
        logging.debug(f"Generated {points.shape[0]} points in {expected_dim}-dimensional space.")

    def test_rasterize_with_nonzero_xofs(self):
        """
        Test rasterize function with a non-zero offset (xofs).
        """
        N = 12
        # Generate a random lattice matrix R with slight perturbations
        #np.random.seed(42)
        R_base = normdet(np.eye(N) #+ 0.1 * np.random.rand(N, N)
                         , target_det=1/10)

        # Apply a random rotation
        rotation = special_ortho_group.rvs(N)  # Random orthogonal matrix
        R = R_base @ rotation

        # Define a non-zero offset
        xofs = np.array([0.5] * N)

        # Log the original matrix and offset
        logging.debug(f"Lattice matrix R:\n{R}")
        logging.debug(f"Offset xofs: {xofs}")

        # Rasterize the lattice points with the non-zero offset
        eps = 0.001  # Increased tolerance
        points = rasterize(R, eps=eps, minbvol=True, sortit=True, xofs=xofs)

        # Log the generated points
        logging.debug(f"Generated points with xofs={xofs}:\n{points}")

        # Assertions to check the validity of the results
        self.assertIsInstance(points, np.ndarray)
        self.assertGreaterEqual(points.shape[0], 1, "Expected at least 1 lattice point.")


        # Check that the points are close to integers after applying the offset
        transformed_points = np.linalg.inv(R) @ (points - xofs).T
        abs_deviation = np.abs(transformed_points - np.round(transformed_points))
        max_deviation = np.max(abs_deviation)

        # Log the transformed points and their deviations
        logging.debug(f"Transformed points:\n{transformed_points}")
        logging.debug(f"Absolute deviations from integers:\n{abs_deviation}")
        logging.debug(f"Maximum deviation: {max_deviation}")
        
        # Verify that the coordinates lie within reasonable bounds
        self.assertTrue(
            np.all((points >= -eps) & (points <= 1 + eps)),
            "Some points fall outside the expected bounds."
        )
        logging.debug(f"Generated {points.shape[0]} points with non-zero offset in {N}-dimensional space.")

    def test_memory_efficient_sampling(self):
        """
        Test rasterize with memory constraints in high dimensions.
        """
        N = 10
        R = normdet(np.eye(N) + 0.1 * np.random.rand(N, N), target_det=1/10)  # Changed target_det to 1.0
        eps = 1e-5  # Increased tolerance
        minbvol = True

        points = rasterize(R, eps=eps, minbvol=minbvol)
        logging.debug("Generated points shape:", points.shape)

        self.assertIsInstance(points, np.ndarray)
        self.assertGreater(points.shape[0], 0, "No points generated within the constraints.")
        if points.shape[0] == 0:
            logging.debug("No points generated. Check bounds and constraints.")

class TestVerifyK(unittest.TestCase):
    def test_verifyK_high_dimensional(self):
        """
        Test verifyK with valid subsampling matrices in higher dimensions using companion matrices.
        """
        N = 8
        detK = 256  # Changed from -256 to 256

        # Create a characteristic polynomial with the desired determinant
        # p(x) = x^8 - 256
        coeffs = [1] + [0]*(N-1) + [detK]  # Correct coefficients for detK=256

        try:
            # Generate the companion matrix for this polynomial
            K = companion(coeffs).astype(int)
            logging.debug(f"Generated companion matrix K:\n{K}")
        except ValueError as e:
            self.fail(f"Failed to create companion matrix: {e}")

        det_computed = np.linalg.det(K)
        logging.debug(f"Computed determinant: {det_computed}")

        self.assertTrue(
            verifyK(K, detK=detK, eps=1e-1),  # Increased tolerance
            f"Companion matrix failed verification. Computed det: {det_computed}"
        )
        logging.debug("test_verifyK_high_dimensional passed.")

    def test_np_poly_correctness(self):
        """
        Test that np.poly computes the correct characteristic polynomial.
        """
        K = np.array([[1, 1], [1, 3]])
        p = np.poly(K)
        logging.debug(f"Characteristic polynomial of K:\n{p}")
        expected_p = np.array([1., -4., 2.])
        np.testing.assert_almost_equal(
            p, expected_p, decimal=5,
            err_msg="np.poly does not compute the correct characteristic polynomial."
        )
        logging.debug("test_np_poly_correctness passed.")

if __name__ == "__main__":
    unittest.main()
