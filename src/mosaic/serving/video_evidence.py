from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mosaic.video.data import load_video_manifest


VIDEO_REPORT_SCHEMA = "mosaic.msrvtt_frozen_final.v1"
VIDEO_AUDIT_SCHEMA = "mosaic.msrvtt_final_audit.v1"

_MODEL_SPECS = (
    ("frozen_clip_mean_pool", "Frozen CLIP mean", False),
    ("frozen_clip_max_pool", "Frozen CLIP max", False),
    ("mosaic_trained_temporal_attention", "MOSAIC temporal", True),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_projection(payload: dict[str, Any]) -> dict[str, Any]:
    recall = payload["recall_at"]
    return {
        "r1": float(recall["1"]),
        "r5": float(recall["5"]),
        "r10": float(recall["10"]),
        "r50": float(recall["50"]),
        "mrr": float(payload["mrr"]),
        "median_rank": float(payload["median_rank"]),
        "mean_rank": float(payload["mean_rank"]),
        "queries": int(payload["queries"]),
    }


def _delta_projection(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "delta": float(payload["delta"]),
        "lower": float(payload["lower"]),
        "upper": float(payload["upper"]),
        "replicates": int(payload["replicates"]),
        "clusters": int(payload["clusters"]),
    }


class VideoEvidenceCatalog:
    """Project the frozen MSR-VTT evidence and serve only fixed local samples."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.report_path = self.root / "reports" / "mosaic_msrvtt_frozen_final_v1.json"
        self.audit_path = self.root / "reports" / "mosaic_msrvtt_frozen_final_v1.audit.json"
        self.manifest_path = self.root / "data" / "processed" / "msrvtt_test_1ka_v1.json"
        self.video_root = (self.root / "data" / "raw" / "msrvtt" / "video").resolve()
        self.sample_allowlist_path = self.root / "runtime-data" / "msrvtt_sample_allowlist.json"
        self._report: dict[str, Any] | None = None
        self._audit: dict[str, Any] | None = None
        self._report_error: str | None = None
        self._samples: dict[str, dict[str, Any]] = {}
        self._sample_error: str | None = None
        self._load_report()
        self._load_samples()

    @property
    def ready(self) -> bool:
        return self._report is not None

    def _load_report(self) -> None:
        if not self.report_path.is_file():
            self._report_error = "frozen Final report is not available"
            return
        try:
            payload = json.loads(self.report_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != VIDEO_REPORT_SCHEMA:
                raise ValueError("MSR-VTT report schema mismatch")
            evaluation = payload["evaluation"]
            models = evaluation["models"]
            if set(models) != {spec[0] for spec in _MODEL_SPECS}:
                raise ValueError("MSR-VTT report model set mismatch")
            if int(evaluation["videos"]) != 1000 or int(evaluation["captions"]) != 1000:
                raise ValueError("MSR-VTT Final coverage mismatch")
            if not self.audit_path.is_file():
                raise ValueError("MSR-VTT Final audit is not available")
            audit = json.loads(self.audit_path.read_text(encoding="utf-8"))
            if audit.get("schema_version") != VIDEO_AUDIT_SCHEMA:
                raise ValueError("MSR-VTT audit schema mismatch")
            if audit.get("status") != "completed":
                raise ValueError("MSR-VTT audit is not completed")
            if audit.get("evaluation_sha256") != payload.get("evaluation_sha256"):
                raise ValueError("MSR-VTT evaluation digest mismatch")
            if audit.get("report", {}).get("json_sha256") != _sha256_file(
                self.report_path
            ):
                raise ValueError("MSR-VTT report digest mismatch")
            self._report = payload
            self._audit = audit
        except Exception as exc:
            self._report_error = f"{type(exc).__name__}: {exc}"

    def _load_samples(self) -> None:
        if (
            not self.manifest_path.is_file()
            or not self.video_root.is_dir()
            or not self.sample_allowlist_path.is_file()
        ):
            return
        try:
            configured = json.loads(self.sample_allowlist_path.read_text(encoding="utf-8"))
            if not isinstance(configured, list):
                raise ValueError("sample allowlist must be a JSON array")
            manifest = load_video_manifest(self.manifest_path)
            rows = {str(row["video_id"]): row for row in manifest["videos"]}
            samples: dict[str, dict[str, Any]] = {}
            for spec in configured:
                if not isinstance(spec, dict) or set(spec) != {"id", "video_id", "label"}:
                    raise ValueError("invalid sample allowlist row")
                row = rows.get(spec["video_id"])
                if row is None:
                    continue
                file_name = Path(str(row["file_name"])).name
                candidate = (self.video_root / file_name).resolve()
                if (
                    not file_name.lower().endswith(".mp4")
                    or not candidate.is_file()
                    or not candidate.is_relative_to(self.video_root)
                ):
                    continue
                captions = row.get("captions")
                if not isinstance(captions, list) or len(captions) != 1:
                    continue
                sample_id = str(spec["id"])
                samples[sample_id] = {
                    "id": sample_id,
                    "video_id": str(row["video_id"]),
                    "label": str(spec["label"]),
                    "caption": str(captions[0]),
                    "category": int(row["category"]),
                    "bytes": int(candidate.stat().st_size),
                    "stream_url": f"/api/video/sample/{sample_id}",
                    "path": candidate,
                }
            self._samples = samples
        except Exception as exc:
            self._sample_error = f"{type(exc).__name__}: {exc}"

    def sample_path(self, sample_id: str) -> Path:
        row = self._samples.get(str(sample_id))
        if row is None:
            raise KeyError(sample_id)
        candidate = Path(row["path"]).resolve()
        if not candidate.is_file() or not candidate.is_relative_to(self.video_root):
            raise KeyError(sample_id)
        return candidate

    def summary(self) -> dict[str, Any]:
        samples = [
            {key: value for key, value in row.items() if key != "path"}
            for row in self._samples.values()
        ]
        if self._report is None:
            return {
                "status": "not_ready",
                "scope": "frozen_final_aggregate_evidence",
                "error": self._report_error,
                "samples": samples,
                "sample_error": self._sample_error,
                "ranking": {
                    "available": False,
                    "reason": "No per-query ranked result artifact is published by this interface.",
                },
            }

        report = self._report
        evaluation = report["evaluation"]
        projected_models = []
        for model_id, label, trainable in _MODEL_SPECS:
            model = evaluation["models"][model_id]
            projected_models.append(
                {
                    "id": model_id,
                    "label": label,
                    "trainable": trainable,
                    "text_to_video": _metric_projection(model["text_to_video"]),
                    "video_to_text": _metric_projection(model["video_to_text"]),
                    "rank_sha256": {
                        "text_to_video": str(model["rank_sha256"]["text_to_video"]),
                        "video_to_text": str(model["rank_sha256"]["video_to_text"]),
                    },
                }
            )

        bootstrap = evaluation["paired_video_cluster_bootstrap_trained_vs_mean"]
        if self._audit is None:
            raise RuntimeError("validated video report is missing its audit binding")
        audit_status = str(self._audit["status"])
        audit_matches = True
        audit_file_sha256 = _sha256_file(self.audit_path)

        protocol = report["protocol"]
        return {
            "status": "ready",
            "scope": "frozen_final_aggregate_evidence",
            "dataset": str(protocol["dataset"]),
            "protocol": {
                "query_policy": str(protocol["query_policy"]),
                "selection": str(protocol["selection"]),
                "bootstrap_cluster": str(protocol["bootstrap_cluster"]),
                "frames_per_video": 12,
                "audio_or_ocr": False,
            },
            "coverage": {
                "videos": int(evaluation["videos"]),
                "captions": int(evaluation["captions"]),
                "checkpoint_epoch": int(evaluation["checkpoint_epoch"]),
            },
            "models": projected_models,
            "trained_vs_mean": {
                "text_to_video": {
                    "r1": _delta_projection(bootstrap["text_to_video"]["recall@1"]),
                    "r10": _delta_projection(bootstrap["text_to_video"]["recall@10"]),
                    "mrr": _delta_projection(bootstrap["text_to_video"]["mrr"]),
                },
                "video_to_text": {
                    "r1": _delta_projection(bootstrap["video_to_text"]["recall@1"]),
                    "r10": _delta_projection(bootstrap["video_to_text"]["recall@10"]),
                    "mrr": _delta_projection(bootstrap["video_to_text"]["mrr"]),
                },
            },
            "evidence": {
                "input_commit": str(report["provenance"]["input_commit"]),
                "training_input_commit": str(report["provenance"]["training_input_commit"]),
                "evaluation_sha256": str(report["evaluation_sha256"]),
                "report_sha256": _sha256_file(self.report_path),
                "audit_status": audit_status,
                "audit_matches_report": audit_matches,
                "audit_file_sha256": audit_file_sha256,
            },
            "claim_boundary": {
                "offline_public_data_only": True,
                "online_traffic": False,
                "sota_claimed": False,
                "audio_asr_ocr": False,
                "one_caption_protocol": True,
                "dataset_license_declared_by_mirror": False,
            },
            "ranking": {
                "available": False,
                "reason": (
                    "The frozen Final publishes aggregate metrics and rank digests, "
                    "not per-query ordered result rows. This browser does not fabricate "
                    "text-to-video rankings."
                ),
            },
            "samples": samples,
            "sample_error": self._sample_error,
        }


__all__ = [
    "VIDEO_AUDIT_SCHEMA",
    "VIDEO_REPORT_SCHEMA",
    "VideoEvidenceCatalog",
]
