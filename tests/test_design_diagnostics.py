import unittest

import numpy as np

from LatticeForge.design_diagnostics import (
    addition_diagnostics,
    candidate_addition_diagnostics,
    criterion_direction,
    design_diagnostics,
    diagnostic_delta,
    diagnostic_score,
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

    def test_addition_diagnostics_reports_fill_improvement(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0]])

        report = addition_diagnostics(
            points,
            np.array([0.5, 0.5]),
            grid_size=3,
        )

        self.assertEqual(report["baseline"]["n_points"], 2)
        self.assertEqual(report["after"]["n_points"], 3)
        self.assertLess(report["delta"]["fill_distance"], 0.0)

    def test_candidate_addition_ranking_prefers_center_by_fill(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0]])
        candidates = np.array(
            [
                [0.0, 1.0],
                [0.5, 0.5],
                [1.0, 1.0],
            ]
        )

        report = candidate_addition_diagnostics(
            points,
            candidates,
            grid_size=3,
            rank_by="fill_distance",
        )

        self.assertEqual(report["direction"], "minimize")
        self.assertEqual(report["candidates"][0]["candidate_index"], 1)
        self.assertLess(report["candidates"][0]["score_delta"], 0.0)

    def test_diagnostic_score_and_delta_are_explicit(self):
        before = {"fill_distance": 1.0, "mesh_ratio": 2.0}
        after = {"fill_distance": 0.5, "mesh_ratio": 1.5}

        self.assertEqual(criterion_direction("separation"), "maximize")
        self.assertEqual(criterion_direction("fill_distance"), "minimize")
        self.assertAlmostEqual(diagnostic_score(after, "mesh_ratio"), 1.5)
        self.assertAlmostEqual(diagnostic_delta(before, after)["fill_distance"], -0.5)


if __name__ == "__main__":
    unittest.main()
