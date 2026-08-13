from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from mosaic.video.data import _digest
from mosaic.video.experiment import (
    CHECKPOINT_CONFIG_NAME,
    CHECKPOINT_NAME,
    deterministic_epoch_caption_rows,
    evaluate_video_model,
    evaluate_video_suite,
    load_aligned_video_inputs,
    load_trained_video_model,
    strip_internal,
    train_video_encoder,
)
from mosaic.video.features import make_toy_video_feature_bundle
from mosaic.video.reporting import finalize_msrvtt, verify_final_audit


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(
    path: Path,
    splits: list[str],
    *,
    captions_per_video: int,
) -> dict[str, object]:
    rows = [
        {
            "video_id": f"video{index}",
            "numeric_id": index,
            "file_name": f"video{index}.mp4",
            "captions": [f"caption {index} slot {slot}" for slot in range(captions_per_video)],
            "category": index % 3,
            "split": split,
        }
        for index, split in enumerate(splits)
    ]
    payload: dict[str, object] = {
        "schema_version": "mosaic.video_manifest.v1",
        "dataset": "MSR-VTT-1K-A",
        "mirror": "unit-test",
        "video_root": str(path.parent),
        "protocol": "unit-test",
        "selection": {"counts": dict(Counter(splits))},
        "videos": rows,
    }
    payload["manifest_sha256"] = _digest(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _config() -> dict[str, object]:
    return {
        "model": {
            "embedding_dim": 8,
            "hidden_dim": 16,
            "dropout": 0.0,
            "temperature_init": 0.07,
            "hard_negative_weight": 0.1,
            "hard_negative_top_k": 2,
            "teacher_preservation_weight": 0.1,
        },
        "training": {
            "seed": 20260730,
            "epochs": 1,
            "batch_size": 3,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
        },
        "selection": {"directional_r10_max_drop": 0.002},
        "evaluation": {
            "ks": [1, 5, 10],
            "bootstrap_replicates": 100,
            "bootstrap_seed": 31,
        },
    }


class VideoExperimentTests(unittest.TestCase):
    def _bundle(
        self,
        root: Path,
        name: str,
        splits: list[str],
        captions_per_video: int,
    ) -> tuple[Path, Path]:
        manifest_path = root / f"{name}_manifest.json"
        manifest = _manifest(
            manifest_path, splits, captions_per_video=captions_per_video
        )
        feature_path = root / f"{name}_features.npz"
        make_toy_video_feature_bundle(
            feature_path,
            video_count=len(splits),
            frames_per_video=3,
            captions_per_video=captions_per_video,
            dim=8,
            manifest_sha256=str(manifest["manifest_sha256"]),
            dataset="MSR-VTT-1K-A",
        )
        return manifest_path, feature_path

    def test_caption_sampling_is_deterministic_and_grouped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, feature_path = self._bundle(
                Path(directory), "train", ["train"] * 6 + ["dev"] * 2, 3
            )
            manifest, _, arrays = load_aligned_video_inputs(manifest_path, feature_path)
            indices = np.asarray([0, 2, 5], dtype=np.int64)
            first = deterministic_epoch_caption_rows(
                manifest, arrays, indices, seed=7, epoch=1
            )
            second = deterministic_epoch_caption_rows(
                manifest, arrays, indices, seed=7, epoch=1
            )
            np.testing.assert_array_equal(first, second)
            self.assertTrue(all(index * 3 <= row < index * 3 + 3 for index, row in zip(indices, first)))
            broken = dict(arrays)
            broken["caption_video_index"] = arrays["caption_video_index"][::-1]
            with self.assertRaisesRegex(ValueError, "grouped"):
                deterministic_epoch_caption_rows(
                    manifest, broken, indices, seed=7, epoch=1
                )

    def test_train_checkpoint_and_independent_test_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_manifest, train_features = self._bundle(
                root, "train", ["train"] * 6 + ["dev"] * 2, 3
            )
            checkpoint = root / "checkpoint"
            summary = train_video_encoder(
                train_manifest,
                train_features,
                checkpoint,
                _config(),
                device="cpu",
            )
            self.assertTrue((checkpoint / CHECKPOINT_NAME).is_file())
            self.assertTrue((checkpoint / CHECKPOINT_CONFIG_NAME).is_file())
            self.assertEqual(summary["manifest_file_sha256"], _file_sha(train_manifest))
            self.assertEqual(summary["feature_npz_sha256"], _file_sha(train_features))
            payload = json.loads(
                (checkpoint / CHECKPOINT_CONFIG_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["training_feature_npz_sha256"], _file_sha(train_features))
            self.assertFalse(payload["test_labels_accessed"])
            model, loaded, device = load_trained_video_model(
                checkpoint, device="cpu", expected_input_dim=8, expected_frames=3
            )
            self.assertEqual(str(device), "cpu")
            self.assertEqual(loaded["epoch"], summary["best_epoch"])
            self.assertEqual(model.input_dim, 8)

            dev_result = evaluate_video_model(
                train_manifest,
                train_features,
                _config(),
                checkpoint_dir=checkpoint,
                device="cpu",
                split="dev",
                bootstrap_replicates=100,
            )
            self.assertEqual(dev_result["videos"], 2)
            self.assertIn("paired_video_cluster_bootstrap_vs_frozen_mean", dev_result)

            test_manifest, test_features = self._bundle(
                root, "test", ["test"] * 4, 1
            )
            test_result = evaluate_video_model(
                test_manifest,
                test_features,
                _config(),
                checkpoint_dir=checkpoint,
                device="cpu",
                split="test",
                bootstrap_replicates=100,
            )
            self.assertEqual(test_result["videos"], 4)
            self.assertEqual(test_result["captions"], 4)
            json.dumps(strip_internal(test_result))
            suite = evaluate_video_suite(
                test_manifest,
                test_features,
                _config(),
                checkpoint_dir=checkpoint,
                device="cpu",
                split="test",
                bootstrap_replicates=100,
            )
            self.assertEqual(set(suite["models"]), {
                "frozen_clip_mean_pool",
                "frozen_clip_max_pool",
                "mosaic_trained_temporal_attention",
            })
            config_path = root / "config.json"
            config_path.write_text(json.dumps(_config()), encoding="utf-8")
            report_path = root / "reports" / "final.json"
            markdown_path = root / "reports" / "final.md"
            audit_path = root / "reports" / "final.audit.json"
            formal = finalize_msrvtt(
                repo_root=root,
                manifest_path=test_manifest,
                feature_path=test_features,
                checkpoint_dir=checkpoint,
                config_path=config_path,
                report_path=report_path,
                markdown_path=markdown_path,
                audit_path=audit_path,
                device="cpu",
                expected_videos=4,
                require_clean=False,
            )
            self.assertEqual(formal["evaluation"]["videos"], 4)
            self.assertEqual(verify_final_audit(root, audit_path)["status"], "verified")
            with self.assertRaises(FileExistsError):
                finalize_msrvtt(
                    repo_root=root,
                    manifest_path=test_manifest,
                    feature_path=test_features,
                    checkpoint_dir=checkpoint,
                    config_path=config_path,
                    report_path=report_path,
                    markdown_path=markdown_path,
                    audit_path=audit_path,
                    device="cpu",
                    expected_videos=4,
                    require_clean=False,
                )

    def test_training_rejects_any_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, features = self._bundle(root, "test", ["test"] * 4, 1)
            with self.assertRaisesRegex(ValueError, "train/dev-only"):
                train_video_encoder(
                    manifest,
                    features,
                    root / "checkpoint",
                    _config(),
                    device="cpu",
                )

    def test_mean_projection_ablation_trains_and_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, features = self._bundle(
                root, "mean", ["train"] * 6 + ["dev"] * 2, 2
            )
            config = _config()
            config["model"]["aggregator"] = "mean"
            output = root / "mean_checkpoint"
            summary = train_video_encoder(
                manifest,
                features,
                output,
                config,
                device="cpu",
            )
            self.assertEqual(summary["architecture"]["aggregator"], "mean")
            model, payload, _ = load_trained_video_model(output, device="cpu")
            self.assertEqual(model.aggregator, "mean")
            self.assertEqual(payload["architecture"]["aggregator"], "mean")


if __name__ == "__main__":
    unittest.main()
