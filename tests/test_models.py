from __future__ import annotations

import unittest

import torch

from mosaic.models import (
    MosaicDualEncoder,
    hard_negative_margin_loss,
    modality_dropout,
    symmetric_contrastive_loss,
)


class ModelTests(unittest.TestCase):
    def test_gate_and_losses_are_finite(self) -> None:
        torch.manual_seed(7)
        model = MosaicDualEncoder(8, embedding_dim=6, hidden_dim=12)
        image = torch.randn(5, 8)
        text = torch.randn(5, 8)
        result = model(image, text)
        self.assertEqual((5, 6), tuple(result["item"].shape))
        self.assertTrue(torch.isfinite(symmetric_contrastive_loss(result["image"], result["text"], result["temperature"])))
        self.assertTrue(torch.isfinite(hard_negative_margin_loss(result["image"], result["text"])))

    def test_dropout_never_hides_everything(self) -> None:
        image = torch.ones(20, 4)
        text = torch.ones(20, 4)
        result = modality_dropout(image, text, 1.0)
        self.assertTrue(torch.all(result.mask.sum(dim=1) == 1))
        self.assertTrue(torch.all((result.image.abs().sum(dim=1) > 0) | (result.text.abs().sum(dim=1) > 0)))

    def test_gate_rejects_all_missing(self) -> None:
        model = MosaicDualEncoder(4, embedding_dim=4, hidden_dim=8)
        with self.assertRaises(ValueError):
            model.gate(torch.ones(2, 4), torch.ones(2, 4), torch.zeros(2, 2))

    def test_hard_negative_loss_rewards_correct_alignment(self) -> None:
        aligned = torch.eye(4)
        wrong = aligned.roll(1, dims=0)
        good = hard_negative_margin_loss(aligned, aligned, top_k=3)
        bad = hard_negative_margin_loss(aligned, wrong, top_k=3)
        self.assertLess(float(good), float(bad))

    def test_residual_projection_starts_near_clip_geometry(self) -> None:
        torch.manual_seed(3)
        model = MosaicDualEncoder(8, embedding_dim=8, hidden_dim=16)
        value = torch.randn(5, 8)
        cosine = torch.nn.functional.cosine_similarity(
            torch.nn.functional.normalize(value, dim=-1), model.encode_image(value)
        )
        self.assertGreater(float(cosine.min()), 0.99)


if __name__ == "__main__":
    unittest.main()
