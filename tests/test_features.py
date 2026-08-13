from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mosaic.features import load_feature_bundle, make_toy_feature_bundle


class FeatureTests(unittest.TestCase):
    def test_safe_npz_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            make_toy_feature_bundle(path, image_count=6, dim=8)
            metadata, arrays = load_feature_bundle(path)
            self.assertEqual(6, metadata["image_count"])
            self.assertEqual((12, 8), arrays["caption_features"].shape)

    def test_object_arrays_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez(path, image_ids=np.asarray([1], dtype=object))
            with self.assertRaises(ValueError):
                load_feature_bundle(path)


if __name__ == "__main__":
    unittest.main()

