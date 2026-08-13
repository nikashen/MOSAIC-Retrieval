from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from mosaic.features import ClipFeatureExtractor, load_feature_bundle, resolve_device


class SearchFailure(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = int(status_code)


class MosaicSearchEngine:
    def __init__(self, root: Path, *, device: str = "cpu") -> None:
        self.root = Path(root).resolve()
        self.device_name = device
        self._lock = threading.RLock()
        self._query_encoder: Any | None = None
        self._adapter: Any | None = None
        self._index: Any | None = None
        self._vectors: np.ndarray | None = None
        self._ids: np.ndarray | None = None
        self._metadata: dict[str, Any] = {}
        self._catalog: dict[int, dict[str, str]] = {}
        self._image_root: Path | None = None
        self._last_error: str | None = None
        self._ann_fallback_reason: str | None = None
        self._catalog_error: str | None = None
        self._load_index()
        self._load_catalog()

    def _load_catalog(self) -> None:
        manifest_path = self.root / "data" / "processed" / "coco_manifest.json"
        if not manifest_path.is_file():
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            image_root = Path(str(manifest["image_root"])).resolve()
            if not image_root.is_dir():
                image_root = (self.root / "data" / "raw" / "val2017").resolve()
            catalog: dict[int, dict[str, str]] = {}
            for row in manifest["images"]:
                content_id = int(row["image_id"])
                file_name = str(row["file_name"])
                caption = str(row["captions"][0]) if row.get("captions") else ""
                candidate = (image_root / file_name).resolve()
                if candidate.is_file() and candidate.is_relative_to(image_root):
                    catalog[content_id] = {"file_name": file_name, "caption": caption}
            self._catalog = catalog
            self._image_root = image_root
        except Exception as exc:
            self._catalog_error = f"{type(exc).__name__}: {exc}"

    def _load_index(self) -> None:
        path = self.root / "artifacts" / "mosaic_coco5k_v1" / "item_vectors.npz"
        if not path.is_file():
            self._last_error = f"index artifact not found: {path}"
            return
        try:
            with np.load(path, allow_pickle=False) as bundle:
                required = {"content_id", "content_vector", "metadata_json"}
                if set(bundle.files) != required:
                    raise ValueError("index NPZ key mismatch")
                self._ids = np.asarray(bundle["content_id"], dtype=np.int64)
                self._vectors = np.asarray(bundle["content_vector"], dtype=np.float32)
                self._metadata = json.loads(str(bundle["metadata_json"].item()))
            if self._vectors.ndim != 2 or self._ids.shape != (self._vectors.shape[0],):
                raise ValueError("index arrays are not aligned")
            norms = np.linalg.norm(self._vectors, axis=1)
            if not np.isfinite(self._vectors).all() or np.any(norms < 0.99) or np.any(norms > 1.01):
                raise ValueError("index vectors must be finite and L2-normalized")
            try:
                import faiss

                faiss_path = path.with_name("content.index")
                if faiss_path.is_file():
                    # FAISS' Windows C++ FileIO cannot reliably open a Unicode
                    # absolute path. The CLI changes into repository root, so
                    # an ASCII relative path is both portable and testable.
                    if Path.cwd().resolve() != self.root:
                        raise RuntimeError("process cwd must equal MOSAIC root for FAISS on Windows")
                    self._index = faiss.read_index(str(faiss_path.relative_to(self.root)))
                else:
                    index = faiss.IndexFlatIP(self._vectors.shape[1])
                    index.add(self._vectors)
                    self._index = index
            except Exception as exc:
                self._index = None
                self._ann_fallback_reason = f"{type(exc).__name__}: {exc}"
            self._last_error = None
        except Exception as exc:
            self._last_error = f"index load failed: {type(exc).__name__}: {exc}"

    @property
    def ready(self) -> bool:
        return self._vectors is not None and self._ids is not None

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.ready else "degraded",
            "service": "mosaic-retrieval",
            "scope": "offline_demo",
            "index_ready": self.ready,
            "index_backend": "faiss.IndexFlatIP" if self._index is not None else "numpy_exact",
            "ann_fallback_reason": self._ann_fallback_reason,
            "items": int(self._ids.size) if self._ids is not None else 0,
            "dimension": int(self._vectors.shape[1]) if self._vectors is not None else None,
            "catalog_items": len(self._catalog),
            "catalog_error": self._catalog_error,
            "device": self.device_name,
            "last_error": self._last_error,
        }

    def models(self) -> dict[str, Any]:
        return {
            "active_model": "mosaic-coco5k-v1",
            "evidence_stage": "external_final_frozen",
            "backbone": "openai/clip-vit-base-patch32",
            "trainable_scope": "projection_and_modality_gate_only",
            "display_scope": "frozen CLIP + adapter/gate",
            "index_ready": self.ready,
            "claim_boundary": {
                "online_traffic": False,
                "online_ab_test": False,
                "video_asr_ocr": False,
            },
        }

    def search_vector(self, vector: np.ndarray, top_k: int = 10) -> list[dict[str, Any]]:
        if not self.ready:
            raise SearchFailure("index_not_ready", "build the item index before searching", 503)
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self._vectors.shape[1] or not np.isfinite(vector).all():
            raise SearchFailure("invalid_query_vector", "query vector width or values are invalid")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            raise SearchFailure("invalid_query_vector", "query vector must be non-zero")
        vector = (vector / norm).reshape(1, -1)
        top_k = max(1, min(int(top_k), int(self._ids.size)))
        if self._index is not None:
            scores, positions = self._index.search(vector, top_k)
            scores, positions = scores[0], positions[0]
        else:
            values = (self._vectors @ vector[0]).astype(np.float32)
            positions = np.argsort(-values, kind="stable")[:top_k]
            scores = values[positions]
        rows: list[dict[str, Any]] = []
        for rank, (position, score) in enumerate(zip(positions.tolist(), scores.tolist())):
            if int(position) < 0:
                continue
            content_id = int(self._ids[position])
            row: dict[str, Any] = {"rank": rank + 1, "content_id": content_id, "score": float(score)}
            if content_id in self._catalog:
                row["preview_caption"] = self._catalog[content_id]["caption"]
                row["image_url"] = f"/api/content/{content_id}/image"
            rows.append(row)
        return rows

    def image_path(self, content_id: int) -> Path:
        if self._image_root is None or int(content_id) not in self._catalog:
            raise SearchFailure("content_not_found", "content image is not available", 404)
        candidate = (self._image_root / self._catalog[int(content_id)]["file_name"]).resolve()
        if not candidate.is_file() or not candidate.is_relative_to(self._image_root):
            raise SearchFailure("content_not_found", "content image is not available", 404)
        return candidate

    def _load_text_encoder(self) -> None:
        with self._lock:
            if self._query_encoder is not None:
                return
            from mosaic.experiment import load_trained_model

            feature_path = self.root / "artifacts" / "mosaic_coco5k_v1" / "clip_features.npz"
            config_path = self.root / "configs" / "coco5k_v1.json"
            checkpoint = self.root / "artifacts" / "mosaic_coco5k_v1"
            if not feature_path.is_file() or not (checkpoint / "adapter.safetensors").is_file():
                raise SearchFailure("model_not_ready", "feature bundle and trained adapter are required", 503)
            metadata, arrays = load_feature_bundle(feature_path)
            self._query_encoder = ClipFeatureExtractor(metadata["model_name"], device=self.device_name)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self._adapter, _ = load_trained_model(
                checkpoint,
                int(arrays["image_features"].shape[1]),
                config,
                device=self.device_name,
            )

    def search_text(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        query = " ".join(str(query).split())
        if not query:
            raise SearchFailure("invalid_query", "query must not be empty")
        self._load_text_encoder()
        encoded = self._query_encoder.encode_texts([query], batch_size=1)
        import torch

        with torch.inference_mode():
            vector = self._adapter.encode_query(torch.from_numpy(encoded).to(self._adapter.raw_temperature.device)).cpu().numpy()[0]
        return self.search_vector(vector, top_k)


__all__ = ["MosaicSearchEngine", "SearchFailure"]
