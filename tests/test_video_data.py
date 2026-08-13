from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mosaic.video.data import _digest, build_msrvtt_manifests, load_video_manifest


def _row(index: int, captions: list[str] | str) -> dict[str, object]:
    return {
        "video_id": f"video{index}",
        "video": f"video{index}.mp4",
        "caption": captions,
        "category": index % 20,
        "id": index,
    }


class VideoManifestTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        video_root = root / "video"
        video_root.mkdir()
        train = [_row(index, [f"caption {index} a", f"caption {index} b"]) for index in range(6)]
        test = [_row(index, f"test caption {index}") for index in range(6, 9)]
        for row in train + test:
            (video_root / str(row["video"])).write_bytes(b"not-a-real-video")
        train_path = root / "train.json"
        test_path = root / "test.json"
        train_path.write_text(json.dumps(train), encoding="utf-8")
        test_path.write_text(json.dumps(test), encoding="utf-8")
        return train_path, test_path, video_root

    def test_build_is_deterministic_and_split_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path, test_path, video_root = self._fixture(root)
            first_train, first_test = build_msrvtt_manifests(
                train_path,
                test_path,
                video_root,
                root / "train_manifest.json",
                root / "test_manifest.json",
                dev_count=2,
                dev_salt="fixed-test-salt",
            )
            second_train, second_test = build_msrvtt_manifests(
                train_path,
                test_path,
                video_root,
                root / "train_manifest_2.json",
                root / "test_manifest_2.json",
                dev_count=2,
                dev_salt="fixed-test-salt",
            )
            self.assertEqual(first_train["manifest_sha256"], second_train["manifest_sha256"])
            self.assertEqual(first_test["manifest_sha256"], second_test["manifest_sha256"])
            self.assertEqual(first_train["selection"]["counts"], {"train": 4, "dev": 2})
            self.assertEqual(first_test["selection"]["counts"], {"test": 3})
            train_ids = {row["video_id"] for row in first_train["videos"]}
            test_ids = {row["video_id"] for row in first_test["videos"]}
            self.assertFalse(train_ids & test_ids)
            self.assertEqual(sum(row["split"] == "dev" for row in first_train["videos"]), 2)
            self.assertEqual(len(first_train["videos"][0]["captions"]), 2)
            self.assertEqual(len(first_test["videos"][0]["captions"]), 1)
            self.assertEqual(load_video_manifest(root / "train_manifest.json"), first_train)

    def test_digest_rejects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path, test_path, video_root = self._fixture(root)
            build_msrvtt_manifests(
                train_path,
                test_path,
                video_root,
                root / "train_manifest.json",
                root / "test_manifest.json",
                dev_count=2,
            )
            payload = json.loads((root / "train_manifest.json").read_text(encoding="utf-8"))
            payload["videos"][0]["captions"][0] = "tampered"
            (root / "train_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_video_manifest(root / "train_manifest.json")

    def test_missing_video_and_overlapping_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path, test_path, video_root = self._fixture(root)
            (video_root / "video0.mp4").unlink()
            with self.assertRaises(FileNotFoundError):
                build_msrvtt_manifests(
                    train_path,
                    test_path,
                    video_root,
                    root / "train_manifest.json",
                    root / "test_manifest.json",
                    dev_count=2,
                )

    def test_loader_rejects_rehashed_structural_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path, test_path, video_root = self._fixture(root)
            build_msrvtt_manifests(
                train_path,
                test_path,
                video_root,
                root / "train_manifest.json",
                root / "test_manifest.json",
                dev_count=2,
            )
            payload = json.loads((root / "train_manifest.json").read_text(encoding="utf-8"))
            payload["selection"]["counts"]["train"] += 1
            payload["manifest_sha256"] = _digest(payload)
            (root / "train_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split counts mismatch"):
                load_video_manifest(root / "train_manifest.json")
            test_rows = json.loads(test_path.read_text(encoding="utf-8"))
            test_rows[0]["video_id"] = "video0"
            test_path.write_text(json.dumps(test_rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                build_msrvtt_manifests(
                    train_path,
                    test_path,
                    video_root,
                    root / "train_manifest.json",
                    root / "test_manifest.json",
                    dev_count=2,
                    require_files=False,
                )


if __name__ == "__main__":
    unittest.main()
