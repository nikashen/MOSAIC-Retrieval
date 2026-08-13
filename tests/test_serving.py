from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
)
from fastapi.testclient import TestClient
from PIL import Image

from mosaic.serving.app import create_app
from mosaic.serving.video_evidence import VideoEvidenceCatalog


def _canonical_digest(payload: dict[str, object]) -> str:
    clean = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(
        json.dumps(
            clean,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _direction(r1: float, r10: float) -> dict[str, object]:
    return {
        "recall_at": {"1": r1, "5": r1 + 0.1, "10": r10, "50": 0.9},
        "mrr": r1 + 0.12,
        "median_rank": 3.0,
        "mean_rank": 20.0,
        "queries": 1000,
    }


def _model(t2v_r1: float, t2v_r10: float, v2t_r1: float, v2t_r10: float) -> dict[str, object]:
    return {
        "text_to_video": _direction(t2v_r1, t2v_r10),
        "video_to_text": _direction(v2t_r1, v2t_r10),
        "rank_sha256": {"text_to_video": "a" * 64, "video_to_text": "b" * 64},
    }


def _delta(value: float, lower: float, upper: float) -> dict[str, object]:
    return {
        "delta": value,
        "lower": lower,
        "upper": upper,
        "alpha": 0.05,
        "replicates": 1000,
        "clusters": 1000,
        "seed": 7,
    }


class ServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        artifact = cls.root / "artifacts" / "mosaic_coco5k_v1"
        image_root = cls.root / "data" / "raw" / "val2017"
        video_root = cls.root / "data" / "raw" / "msrvtt" / "video"
        processed = cls.root / "data" / "processed"
        reports = cls.root / "reports"
        artifact.mkdir(parents=True)
        image_root.mkdir(parents=True)
        video_root.mkdir(parents=True)
        processed.mkdir(parents=True)
        reports.mkdir(parents=True)
        vectors = np.eye(2, dtype=np.float32)
        np.savez_compressed(
            artifact / "item_vectors.npz",
            content_id=np.asarray([7, 8], dtype=np.int64),
            content_vector=vectors,
            metadata_json=np.asarray(json.dumps({"safe_npz": "allow_pickle_false"})),
        )
        Image.new("RGB", (8, 8), (30, 80, 140)).save(image_root / "7.jpg")
        manifest = {
            "image_root": str(image_root),
            "images": [
                {"image_id": 7, "file_name": "7.jpg", "captions": ["blue square"]},
                {"image_id": 8, "file_name": "missing.jpg", "captions": ["missing"]},
            ],
        }
        (processed / "coco_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (video_root / "synthetic0001.mp4").write_bytes(b"allowlisted-mp4")
        (video_root / "synthetic9999.mp4").write_bytes(b"not-allowlisted")
        video_manifest: dict[str, object] = {
            "schema_version": "mosaic.video_manifest.v1",
            "dataset": "MSR-VTT-1K-A",
            "mirror": "fixture",
            "video_root": str(video_root),
            "protocol": "JSFusion 1K-A",
            "source_files": {"train_json_sha256": "c" * 64, "test_json_sha256": "d" * 64},
            "selection": {"test_protocol": "one caption", "counts": {"test": 2}},
            "videos": [
                {
                    "video_id": "synthetic0001",
                    "numeric_id": 1,
                    "file_name": "synthetic0001.mp4",
                    "captions": ["synthetic moving shapes fixture"],
                    "category": 3,
                    "split": "test",
                },
                {
                    "video_id": "synthetic9999",
                    "numeric_id": 9999,
                    "file_name": "synthetic9999.mp4",
                    "captions": ["synthetic hidden fixture"],
                    "category": 1,
                    "split": "test",
                },
            ],
        }
        video_manifest["manifest_sha256"] = _canonical_digest(video_manifest)
        (processed / "msrvtt_test_1ka_v1.json").write_text(
            json.dumps(video_manifest), encoding="utf-8"
        )
        allowlist_path = cls.root / "runtime-data" / "msrvtt_sample_allowlist.json"
        allowlist_path.parent.mkdir(parents=True)
        allowlist_path.write_text(
            json.dumps(
                [
                    {
                        "id": "synthetic-sample",
                        "video_id": "synthetic0001",
                        "label": "Synthetic fixture",
                    }
                ]
            ),
            encoding="utf-8",
        )
        bootstrap = {
            "text_to_video": {
                "recall@1": _delta(0.031, 0.007, 0.055),
                "recall@10": _delta(0.079, 0.056, 0.104),
                "mrr": _delta(0.048, 0.031, 0.066),
            },
            "video_to_text": {
                "recall@1": _delta(0.041, 0.014, 0.068),
                "recall@10": _delta(0.098, 0.073, 0.124),
                "mrr": _delta(0.062, 0.041, 0.083),
            },
        }
        video_report = {
            "schema_version": "mosaic.msrvtt_frozen_final.v1",
            "protocol": {
                "dataset": "MSR-VTT-1K-A",
                "query_policy": "one official JSFusion query caption per video",
                "selection": "all model/epoch decisions completed on deterministic Dev only",
                "bootstrap_cluster": "video_id",
            },
            "provenance": {"input_commit": "e" * 40, "training_input_commit": "f" * 40},
            "evaluation": {
                "videos": 1000,
                "captions": 1000,
                "checkpoint_epoch": 10,
                "models": {
                    "frozen_clip_mean_pool": _model(0.304, 0.631, 0.270, 0.610),
                    "frozen_clip_max_pool": _model(0.197, 0.506, 0.174, 0.479),
                    "mosaic_trained_temporal_attention": _model(0.335, 0.710, 0.311, 0.708),
                },
                "paired_video_cluster_bootstrap_trained_vs_mean": bootstrap,
            },
            "evaluation_sha256": "1" * 64,
        }
        report_path = reports / "mosaic_msrvtt_frozen_final_v1.json"
        report_path.write_text(json.dumps(video_report), encoding="utf-8")
        audit = {
            "schema_version": "mosaic.msrvtt_final_audit.v1",
            "status": "completed",
            "evaluation_sha256": "1" * 64,
            "report": {
                "json_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            },
        }
        (reports / "mosaic_msrvtt_frozen_final_v1.audit.json").write_text(
            json.dumps(audit), encoding="utf-8"
        )
        cls.app = create_app(cls.root, device="cpu")
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        cls.app.state.close_logging()
        cls.temporary.cleanup()

    def test_health_and_models_are_explicit(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(200, health.status_code)
        self.assertTrue(health.json()["index_ready"])
        self.assertTrue(health.json()["video_evidence_ready"])
        self.assertEqual(1, health.json()["video_samples"])
        models = self.client.get("/api/models").json()
        self.assertFalse(models["claim_boundary"]["online_ab_test"])

    def test_vector_search_and_image_contract(self) -> None:
        response = self.client.post(
            "/api/search/vector", json={"vector": [1.0, 0.0], "top_k": 1}
        )
        self.assertEqual(200, response.status_code)
        result = response.json()["results"][0]
        self.assertEqual(7, result["content_id"])
        self.assertEqual(200, self.client.get(result["image_url"]).status_code)

    def test_invalid_vector_is_rejected(self) -> None:
        response = self.client.post(
            "/api/search/vector", json={"vector": [0.0, 0.0], "top_k": 1}
        )
        self.assertEqual(400, response.status_code)
        self.assertEqual("invalid_query_vector", response.json()["error"]["code"])

    def test_video_evidence_projects_frozen_metrics_without_local_paths(self) -> None:
        response = self.client.get("/api/video/evidence")
        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("ready", payload["status"])
        self.assertFalse(payload["ranking"]["available"])
        self.assertIn("does not fabricate", payload["ranking"]["reason"])
        trained = next(model for model in payload["models"] if model["trainable"])
        self.assertEqual(0.335, trained["text_to_video"]["r1"])
        self.assertEqual(0.708, trained["video_to_text"]["r10"])
        self.assertEqual(0.079, payload["trained_vs_mean"]["text_to_video"]["r10"]["delta"])
        self.assertTrue(payload["evidence"]["audit_matches_report"])
        self.assertEqual(["synthetic-sample"], [row["id"] for row in payload["samples"]])
        self.assertNotIn(str(self.root), response.text)
        self.assertNotIn("file_name", response.text)

    def test_video_sample_endpoint_is_fixed_allowlist_only(self) -> None:
        response = self.client.get("/api/video/sample/synthetic-sample")
        self.assertEqual(200, response.status_code)
        self.assertEqual("video/mp4", response.headers["content-type"])
        self.assertEqual("nosniff", response.headers["x-content-type-options"])
        self.assertEqual(b"allowlisted-mp4", response.content)
        missing = self.client.get("/api/video/sample/not-allowlisted")
        self.assertEqual(404, missing.status_code)
        self.assertEqual("video_sample_not_found", missing.json()["error"]["code"])
        traversal = self.client.get("/api/video/sample/%2e%2e%2fsynthetic9999.mp4")
        self.assertEqual(404, traversal.status_code)

    def test_video_evidence_fails_closed_when_audit_binding_drifts(self) -> None:
        audit_path = self.root / "reports" / "mosaic_msrvtt_frozen_final_v1.audit.json"
        original = audit_path.read_text(encoding="utf-8")
        try:
            audit = json.loads(original)
            audit["evaluation_sha256"] = "0" * 64
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            catalog = VideoEvidenceCatalog(self.root)
            self.assertFalse(catalog.ready)
            payload = catalog.summary()
            self.assertEqual("not_ready", payload["status"])
            self.assertIn("evaluation digest mismatch", payload["error"])
            self.assertNotIn("models", payload)
        finally:
            audit_path.write_text(original, encoding="utf-8")

    def test_static_interface_declares_two_tracks_and_ranking_boundary(self) -> None:
        html = self.client.get("/").text
        self.assertIn("COCO 图文检索", html)
        self.assertIn("MSR-VTT 视频证据", html)
        self.assertIn("不会生成未保存的逐查询视频排名", html)


if __name__ == "__main__":
    unittest.main()
