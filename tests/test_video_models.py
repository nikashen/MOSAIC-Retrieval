from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from mosaic.video.models import TemporalVideoEncoder


class TemporalVideoEncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.frames = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 1.0, 0.0, 2.0], [3.0, 7.0, 2.0, 1.0]],
                [[2.0, 1.0, 4.0, 3.0], [6.0, 5.0, 1.0, 0.0], [8.0, 2.0, 3.0, 7.0]],
            ]
        )
        self.mask = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.bool)
        self.text = torch.randn(2, 4)

    def test_all_aggregators_produce_finite_normalized_embeddings(self) -> None:
        for aggregator in ("mean", "max", "temporal_attention"):
            with self.subTest(aggregator=aggregator):
                model = TemporalVideoEncoder(
                    4, hidden_dim=8, dropout=0.0, aggregator=aggregator, max_frames=3
                )
                result = model(self.frames, self.text, self.mask)
                self.assertEqual((2, 4), tuple(result["video"].shape))
                self.assertEqual((2, 4), tuple(result["text"].shape))
                self.assertTrue(torch.isfinite(result["video"]).all())
                self.assertTrue(torch.isfinite(result["text"]).all())
                self.assertGreater(float(result["temperature"]), 0.0)
                torch.testing.assert_close(result["video"].norm(dim=1), torch.ones(2))
                torch.testing.assert_close(result["text"].norm(dim=1), torch.ones(2))

    def test_mean_and_max_respect_mask(self) -> None:
        mean_model = TemporalVideoEncoder(4, hidden_dim=8, dropout=0.0, aggregator="mean")
        max_model = TemporalVideoEncoder(4, hidden_dim=8, dropout=0.0, aggregator="max")
        expected_mean = F.normalize(
            torch.stack(((self.frames[0, 0] + self.frames[0, 1]) / 2, (self.frames[1, 1] + self.frames[1, 2]) / 2)),
            dim=-1,
        )
        expected_max = F.normalize(
            torch.stack((torch.maximum(self.frames[0, 0], self.frames[0, 1]), torch.maximum(self.frames[1, 1], self.frames[1, 2]))),
            dim=-1,
        )
        torch.testing.assert_close(mean_model.encode_video(self.frames, self.mask), expected_mean)
        torch.testing.assert_close(max_model.encode_video(self.frames, self.mask), expected_max)

    def test_epoch_zero_attention_is_exact_uniform_mean(self) -> None:
        frames = torch.randn(2, 4, 4)
        mask = torch.tensor([[1, 1, 1, 1], [1, 0, 1, 0]], dtype=torch.bool)
        attention = TemporalVideoEncoder(
            4, hidden_dim=8, dropout=0.4, aggregator="temporal_attention", max_frames=4
        )
        mean = TemporalVideoEncoder(4, hidden_dim=8, dropout=0.4, aggregator="mean", max_frames=4)
        mean.video_projection.load_state_dict(attention.video_projection.state_dict())

        expected_weights = mask.to(frames.dtype)
        expected_weights /= expected_weights.sum(dim=1, keepdim=True)
        actual_weights = attention.attention_weights(frames, mask)
        torch.testing.assert_close(actual_weights, expected_weights, rtol=0.0, atol=0.0)
        self.assertTrue(torch.all(actual_weights[~mask] == 0))
        torch.testing.assert_close(
            attention.encode_video(frames, mask),
            mean.encode_video(frames, mask),
            rtol=0.0,
            atol=0.0,
        )

    def test_rejects_invalid_masks_nonfinite_values_and_long_timelines(self) -> None:
        model = TemporalVideoEncoder(
            4, hidden_dim=8, aggregator="temporal_attention", max_frames=3
        )
        with self.assertRaisesRegex(ValueError, "at least one unmasked"):
            model.encode_video(self.frames, torch.tensor([[0, 0, 0], [1, 1, 1]]))
        with self.assertRaisesRegex(ValueError, "zero/one"):
            model.encode_video(self.frames, torch.full((2, 3), 0.5))
        bad_frames = self.frames.clone()
        bad_frames[0, 0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            model.encode_video(bad_frames, self.mask)
        bad_text = self.text.clone()
        bad_text[0, 0] = torch.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            model.encode_text(bad_text)
        with self.assertRaisesRegex(ValueError, "exceeds max_frames"):
            model.encode_video(torch.ones(2, 4, 4))
        with self.assertRaisesRegex(ValueError, "rank-3"):
            model.encode_video(torch.ones(2, 4))
        with self.assertRaisesRegex(ValueError, "shape must match"):
            model.encode_video(self.frames, torch.ones(2, 2))

    def test_backward_gradients_are_finite(self) -> None:
        model = TemporalVideoEncoder(
            4,
            embedding_dim=4,
            hidden_dim=8,
            dropout=0.0,
            aggregator="temporal_attention",
            max_frames=3,
        )
        frames = self.frames.clone().requires_grad_(True)
        text = self.text.clone().requires_grad_(True)
        result = model(frames, text, self.mask)
        similarity = result["video"] @ result["text"].transpose(0, 1)
        loss = similarity.square().mean() / result["temperature"]
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(frames.grad)
        self.assertIsNotNone(text.grad)
        self.assertTrue(torch.isfinite(frames.grad).all())
        self.assertTrue(torch.isfinite(text.grad).all())
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
