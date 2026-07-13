import unittest

import numpy as np

from LatticeForge.design_diagnostics import candidate_addition_diagnostics
from LatticeForge.refinement_candidates import (
    coset_representatives,
    refinement_candidate_points,
    refinement_offsets,
)


class TestRefinementCandidates(unittest.TestCase):
    def test_coset_representatives_for_dyadic_square_refinement(self):
        K = 2 * np.eye(2, dtype=int)

        reps = coset_representatives(K)

        self.assertEqual(reps.shape, (4, 2))
        self.assertTrue(any(np.array_equal(rep, np.array([0, 0])) for rep in reps))
        self.assertTrue(any(np.array_equal(rep, np.array([1, 1])) for rep in reps))

    def test_coset_representatives_for_skew_index_two_dilation(self):
        K = np.array([[1, 1], [1, 3]])

        reps = coset_representatives(K)

        self.assertEqual(reps.shape, (2, 2))
        self.assertEqual(abs(round(np.linalg.det(K))), reps.shape[0])

    def test_refinement_offsets_omit_base_coset_by_default(self):
        G = np.eye(2)
        K = 2 * np.eye(2, dtype=int)

        offsets = refinement_offsets(G, K)

        self.assertEqual(offsets.shape, (3, 2))
        self.assertFalse(np.any(np.all(np.isclose(offsets, 0.0), axis=1)))
        self.assertTrue(any(np.allclose(offset, np.array([0.5, 0.5])) for offset in offsets))

    def test_refinement_candidate_points_match_one_step_grid_candidates(self):
        G = np.eye(2)
        K = 2 * np.eye(2, dtype=int)

        candidates = refinement_candidate_points(G, K)

        expected = np.array(
            [
                [0.0, 0.5],
                [0.5, 0.0],
                [0.5, 0.5],
                [0.5, 1.0],
                [1.0, 0.5],
            ]
        )
        self.assertEqual(candidates.shape, expected.shape)
        for point in expected:
            self.assertTrue(any(np.allclose(point, candidate) for candidate in candidates))

    def test_constructed_candidates_can_feed_existing_fill_ranking(self):
        base = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        candidates = refinement_candidate_points(np.eye(2), 2 * np.eye(2, dtype=int))

        report = candidate_addition_diagnostics(
            base,
            candidates,
            grid_size=3,
            rank_by="fill_distance",
        )

        self.assertEqual(report["direction"], "minimize")
        self.assertEqual(report["candidates"][0]["candidate"], (0.5, 0.5))
        self.assertLess(report["candidates"][0]["score_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
