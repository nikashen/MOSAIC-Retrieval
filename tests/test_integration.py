from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from mosaic.integrations.project4 import export_content_vectors
from mosaic.features import make_toy_feature_bundle


class IntegrationTests(unittest.TestCase):
    def test_identity_mapping_requires_explicit_smoke_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features.npz"
            make_toy_feature_bundle(features, image_count=4, dim=8)
            with self.assertRaises(ValueError):
                export_content_vectors(features, root / "out.npz")
            result = export_content_vectors(features, root / "out.npz", allow_identity_demo=True)
            self.assertIn("not_project4_catalog", result["scope"])

    def test_trained_item_vector_bundle_preserves_multimodal_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "items.npz"
            import numpy as np

            np.savez_compressed(
                source,
                content_id=np.asarray([7, 8], dtype=np.int64),
                content_vector=np.eye(2, dtype=np.float32),
                metadata_json=np.asarray(json.dumps({"vector_scope": "trained_full"})),
            )
            output = root / "out.npz"
            result = export_content_vectors(source, output, allow_identity_demo=True)
            with np.load(output, allow_pickle=False) as bundle:
                self.assertTrue(np.all(bundle["modality_mask"] == 3))
            self.assertEqual("trained_full", result["source_vector_scope"])


if __name__ == "__main__":
    unittest.main()
