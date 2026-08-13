from __future__ import annotations

import unittest

import numpy as np

from mosaic.reranking import mine_hard_negatives, reranked_ranks, select_alpha


class RerankingTests(unittest.TestCase):
    def test_hard_negative_mining_excludes_positive(self) -> None:
        embeddings = np.eye(4, dtype=np.float32)
        negatives = mine_hard_negatives(embeddings, embeddings, np.arange(4), top_k=2)
        self.assertEqual((4, 2), negatives.shape)
        for row in range(4):
            self.assertNotIn(row, negatives[row].tolist())

    def test_alpha_selection_can_keep_baseline(self) -> None:
        order = np.asarray([[0, 1, 2], [1, 0, 2]])
        candidates = order[:, :2]
        base = np.asarray([[1.0, 0.5], [1.0, 0.5]])
        interaction = np.asarray([[0.0, 2.0], [0.0, 2.0]])
        alpha, _, evidence = select_alpha(order, candidates, base, interaction, np.asarray([0, 1]), alphas=(0.0, 1.0))
        self.assertEqual(0.0, alpha)
        self.assertIn("0.0", evidence)


if __name__ == "__main__":
    unittest.main()

