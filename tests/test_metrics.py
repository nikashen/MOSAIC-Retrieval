from __future__ import annotations

import unittest

import numpy as np

from mosaic.metrics import (
    aggregate_ranks,
    bootstrap_ci,
    evaluate_direction,
    paired_bootstrap_delta_ci,
)


class MetricTests(unittest.TestCase):
    def test_full_catalog_metrics_and_ties_are_stable(self) -> None:
        query = np.eye(3, dtype=np.float32)
        target = np.eye(3, dtype=np.float32)
        metrics, ranks, ndcg = evaluate_direction(query, target, [[0], [1], [2]], ks=(1, 2))
        self.assertEqual(1.0, metrics.recall_at["1"])
        self.assertTrue(np.all(ranks == 1))
        self.assertTrue(np.all(ndcg == 1))

    def test_bootstrap_is_reproducible_and_paired(self) -> None:
        values = np.asarray([1, 0, 1, 0], dtype=float)
        clusters = [1, 1, 2, 2]
        left = bootstrap_ci(values, clusters, replicates=100, seed=3)
        right = bootstrap_ci(values, clusters, replicates=100, seed=3)
        self.assertEqual(left, right)
        delta = paired_bootstrap_delta_ci(values, np.zeros(4), clusters, replicates=100, seed=3)
        self.assertGreater(delta["delta"], 0)

    def test_invalid_ranks_rejected(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_ranks(np.asarray([0]), (1,))


if __name__ == "__main__":
    unittest.main()

