import unittest

import numpy as np

from LatticeForge.design_diagnostics import (
    design_diagnostics,
    fill_distance_grid,
    mesh_ratio_grid,
    metric_gram,
    metric_transform,
    projected_diagnostics,
    separation_distance,
)


class TestDesignDiagnostics(unittest.TestCase):
    def test_unit_square_corners_have_expected_basic_diagnostics(self):
        points = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )

        self.assertAlmostEqual(separation_distance(points), 0.5)
        self.assertAlmostEqual(fill_distance_grid(points, grid_size=3), np.sqrt(0.5))
        self.assertAlmostEqual(mesh_ratio_grid(points, grid_size=3), np.sqrt(2.0))

    def test_design_report_includes_projection_receipts(self):
        points = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )

        report = design_diagnostics(points, grid_size=3)

        self.assertEqual(report["n_points"], 4)
        self.assertEqual(report["dimension"], 2)
        self.assertIn("0", report["projected"])
        self.assertIn("1", report["projected"])
        self.assertIn("0,1", report["projected"])
        self.assertAlmostEqual(report["projected"]["0"]["fill_distance"], 0.5)
        self.assertAlmostEqual(report["projected"]["1"]["fill_distance"], 0.5)

    def test_metric_length_scales_rescale_points_and_distances(self):
        points = np.array([[0.0, 0.0], [2.0, 0.0]])
        transformed = metric_transform(points, length_scales=np.array([2.0, 1.0]))

        np.testing.assert_array_almost_equal(transformed, np.array([[0.0, 0.0], [1.0, 0.0]]))
        self.assertAlmostEqual(
            separation_distance(points, length_scales=np.array([2.0, 1.0])),
            0.5,
        )

    def test_metric_gram_matches_diagonal_length_scale_metric(self):
        basis = np.array([[2.0, 0.0], [0.0, 3.0]])
        gamma = metric_gram(basis, length_scales=np.array([2.0, 3.0]))

        np.testing.assert_array_almost_equal(gamma, np.eye(2))

    def test_projected_diagnostics_respects_requested_projection_order(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
            ]
        )

        projected = projected_diagnostics(points, projection_orders=(2,), grid_size=3)

        self.assertEqual(set(projected.keys()), {"0,1", "0,2", "1,2"})


if __name__ == "__main__":
    unittest.main()
