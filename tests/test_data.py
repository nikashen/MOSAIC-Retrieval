from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mosaic.data import (
    assign_split,
    build_toy_manifest,
    caption_pairs,
    load_manifest,
    manifest_sha256,
)


class DataTests(unittest.TestCase):
    def test_split_is_deterministic(self) -> None:
        values = [assign_split(index, salt="x") for index in range(100)]
        self.assertEqual(values, [assign_split(index, salt="x") for index in range(100)])
        self.assertNotEqual(values, [assign_split(index, salt="y") for index in range(100)])

    def test_toy_manifest_digest_and_caption_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest = build_toy_manifest(manifest_path, image_root=root)
            loaded = load_manifest(manifest_path)
            self.assertEqual(manifest["manifest_sha256"], loaded["manifest_sha256"])
            self.assertEqual(manifest_sha256(loaded), loaded["manifest_sha256"])
            pairs = caption_pairs(loaded, "test")
            self.assertEqual({row["image_id"] for row in pairs}, {8, 9, 10, 11})
            self.assertEqual({row["split"] for row in pairs}, {"test"})

    def test_digest_rejects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            build_toy_manifest(path, image_root=Path(directory))
            payload = json.loads(path.read_text())
            payload["images"][0]["captions"][0] = "changed"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()

