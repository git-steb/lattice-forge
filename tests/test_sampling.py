#%% tests/test_sampling.py
import unittest
import numpy as np
from LatticeForge.bounded_sampling import (
    CubeTraversalDiagnostics,
    iter_unit_cube_batches,
    sample_unit_cube,
    unit_cube_address_bounds,
)
from LatticeForge.lattice_utils import (
    _incremental_cube_points,
    _integer_grid_chunks,
    _unit_cube_address_bounds,
    ndgridmat,
    normdet,
    rasterize,
    samplecube,
    striplattice,
)

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

    def test_integer_grid_chunks_preserve_ndgrid_order(self):
        ranges = [np.arange(-1, 2), np.arange(2, 6), np.arange(7, 9)]
        expected = ndgridmat(ranges)
        actual = np.concatenate(
            list(_integer_grid_chunks(
                np.array([-1, 2, 7]), np.array([1, 5, 8]), chunk_size=5
            )),
            axis=1,
        )
        np.testing.assert_array_equal(actual, expected)

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
    def test_modern_boundary_default_matches_half_open_manuscript_cube(self):
        points = sample_unit_cube(np.eye(2))
        np.testing.assert_array_equal(points, np.array([[0.0, 0.0]]))

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

    def test_interval_address_bounds_match_explicit_corners(self):
        R = np.array([
            [0.55, 0.10, 0.00],
            [-0.05, 0.60, 0.12],
            [0.08, -0.04, 0.50],
        ])
        xofs = np.array([0.13, -0.07, 0.21])
        R_inv = np.linalg.inv(R)
        corners = np.array(
            np.meshgrid(*([[0.0, 1.0]] * 3), indexing="ij")
        ).reshape(3, -1).T
        explicit = (corners - xofs) @ R_inv.T
        expected_lower = np.floor(explicit.min(axis=0)).astype(np.int64)
        expected_upper = np.ceil(explicit.max(axis=0)).astype(np.int64)
        actual_lower, actual_upper = _unit_cube_address_bounds(R_inv, xofs)
        np.testing.assert_array_equal(actual_lower, expected_lower)
        np.testing.assert_array_equal(actual_upper, expected_upper)

    def test_samplecube_memory_budget_preserves_points_in_eight_dimensions(self):
        diagonal = np.linspace(0.43, 0.71, 8)
        R = np.diag(diagonal)
        R[0, 7] = 0.11
        R[3, 6] = -0.09
        xofs = np.linspace(0.03, 0.24, 8)
        roomy = samplecube(R, xofs=xofs, maxmemMB=64)
        tightly_chunked = samplecube(R, xofs=xofs, maxmemMB=0.01)
        np.testing.assert_allclose(tightly_chunked, roomy, atol=0.0, rtol=0.0)

    def test_incremental_shadow_pruning_matches_complete_address_box(self):
        rng = np.random.default_rng(20260908)
        dimension = 8
        rotation, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        basis = rotation * (1.0 / 50.0) ** (1.0 / dimension)
        xofs = basis @ rng.random(dimension)
        inverse = np.linalg.inv(basis)
        lower, upper = _unit_cube_address_bounds(inverse, xofs)

        ranges = [np.arange(lower[k], upper[k] + 1) for k in range(dimension)]
        complete_addresses = ndgridmat(ranges)
        complete_points = (basis @ complete_addresses).T + xofs
        inside = np.all(
            (complete_points >= -1.0e-10) & (complete_points <= 1.0 + 1.0e-10),
            axis=1,
        )
        expected = np.unique(complete_points[inside], axis=0)
        actual = np.unique(
            _incremental_cube_points(
                basis, xofs, lower, upper, 1.0e-10, 1, maxmemMB=0.02
            ),
            axis=0,
        )
        np.testing.assert_allclose(actual, expected, atol=1.0e-14, rtol=0.0)

    def test_samplecube_rejects_invalid_memory_budget(self):
        with self.assertRaises(ValueError):
            samplecube(np.eye(2), maxmemMB=0)

    def test_iterated_batches_reconstruct_points_and_addresses(self):
        basis = np.array([[0.48, 0.13], [-0.07, 0.52]])
        shift = np.array([0.11, 0.23])
        diagnostics = CubeTraversalDiagnostics()
        batches = list(
            iter_unit_cube_batches(
                basis,
                shift,
                eps=1.0e-10,
                max_memory_mib=0.001,
                diagnostics=diagnostics,
            )
        )
        points = np.unique(
            np.concatenate([batch.points for batch in batches]), axis=0
        )
        expected = sample_unit_cube(
            basis, shift, eps=1.0e-10, max_memory_mib=0.001
        )
        np.testing.assert_allclose(points, expected, atol=0.0, rtol=0.0)
        for batch in batches:
            np.testing.assert_allclose(
                batch.points,
                batch.addresses @ basis.T + shift,
                atol=1.0e-15,
                rtol=0.0,
            )
        self.assertEqual(diagnostics.emitted_point_count, len(expected))
        self.assertGreater(diagnostics.address_box_count, 0)

    def test_public_address_bounds_include_the_declared_tolerance_box(self):
        basis = np.array([[0.55, 0.10], [-0.05, 0.60]])
        shift = np.array([0.13, -0.07])
        lower, upper = unit_cube_address_bounds(
            basis,
            shift,
            eps=1.0e-3,
            upper_tolerance_direction=1,
        )
        inverse = np.linalg.inv(basis)
        corners = np.array(
            np.meshgrid(*([[-1.0e-3, 1.001]] * 2), indexing="ij")
        ).reshape(2, -1).T
        preimage = (corners - shift) @ inverse.T
        np.testing.assert_array_equal(lower, np.floor(preimage.min(axis=0)))
        np.testing.assert_array_equal(upper, np.ceil(preimage.max(axis=0)))



if __name__ == "__main__":
    unittest.main()
