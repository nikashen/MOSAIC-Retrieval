from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from .data import load_manifest


def resolve_device(value: str = "auto") -> torch.device:
    value = str(value).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


class ClipFeatureExtractor:
    """Frozen CLIP feature extractor with bounded batches for 4GB GPUs."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        cache_dir: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        # HF_ENDPOINT/HF_HOME are read by huggingface_hub at model load time.
        from transformers import CLIPModel, CLIPProcessor

        self.model_name = str(model_name)
        self.device = resolve_device(device)
        kwargs: dict[str, Any] = {}
        if cache_dir:
            kwargs["cache_dir"] = str(cache_dir)
        if revision:
            kwargs["revision"] = str(revision)
        if local_files_only:
            kwargs["local_files_only"] = True
        try:
            self.processor = CLIPProcessor.from_pretrained(self.model_name, use_fast=True, **kwargs)
        except (TypeError, ValueError):
            # Older checkpoints may not ship a fast image processor.
            self.processor = CLIPProcessor.from_pretrained(self.model_name, **kwargs)
        self.model = CLIPModel.from_pretrained(self.model_name, **kwargs).to(self.device)
        self.revision = str(revision) if revision else None
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def output_dim(self) -> int:
        return int(self.model.config.projection_dim)

    def _autocast_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return torch.autocast(device_type="cpu", enabled=False)

    @torch.inference_mode()
    def encode_images(self, images: Iterable[Image.Image], *, batch_size: int = 32) -> np.ndarray:
        images = list(images)
        if not images:
            raise ValueError("images must not be empty")
        outputs: list[np.ndarray] = []
        for start in range(0, len(images), max(1, int(batch_size))):
            batch = images[start : start + max(1, int(batch_size))]
            encoded = self.processor(images=batch, return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(self.device)
            with self._autocast_context():
                features = self.model.get_image_features(pixel_values=pixel_values)
            features = torch.nn.functional.normalize(features.float(), dim=-1)
            outputs.append(features.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(outputs, axis=0)

    @torch.inference_mode()
    def encode_texts(self, texts: Iterable[str], *, batch_size: int = 64) -> np.ndarray:
        texts = [str(value) for value in texts]
        if not texts:
            raise ValueError("texts must not be empty")
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), max(1, int(batch_size))):
            batch = texts[start : start + max(1, int(batch_size))]
            encoded = self.processor(
                text=batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._autocast_context():
                features = self.model.get_text_features(**encoded)
            features = torch.nn.functional.normalize(features.float(), dim=-1)
            outputs.append(features.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(outputs, axis=0)


def _open_images(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> list[Image.Image]:
    configured = os.environ.get("MOSAIC_IMAGE_ROOT")
    root = Path(configured or manifest["image_root"]).resolve()
    if not root.is_dir() and not configured:
        portable = (Path.cwd() / "data" / "raw" / "val2017").resolve()
        if portable.is_dir():
            root = portable
    images: list[Image.Image] = []
    for row in rows:
        path = (root / Path(str(row["file_name"]))).resolve()
        if os.path.commonpath((str(path), str(root))) != str(root) or not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def _metadata_digest(metadata: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_feature_bundle(
    manifest_path: Path,
    output_path: Path,
    *,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "auto",
    cache_dir: str | None = None,
    model_revision: str | None = None,
    local_files_only: bool = False,
    image_batch_size: int = 16,
    text_batch_size: int = 64,
    max_images: int | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    rows = list(manifest["images"])
    if max_images is not None:
        if int(max_images) <= 0:
            raise ValueError("max_images must be positive")
        rows = rows[: int(max_images)]
    if not rows:
        raise ValueError("feature manifest is empty")
    extractor = ClipFeatureExtractor(
        model_name,
        device=device,
        cache_dir=cache_dir,
        revision=model_revision,
        local_files_only=local_files_only,
    )
    # Stream image batches instead of retaining all 5,000 decoded PIL images;
    # this is essential on the 16GB/4GB-GPU Windows machine used for evidence.
    image_chunks: list[np.ndarray] = []
    for start in range(0, len(rows), max(1, int(image_batch_size))):
        images = _open_images(manifest, rows[start : start + max(1, int(image_batch_size))])
        try:
            image_chunks.append(extractor.encode_images(images, batch_size=len(images)))
        finally:
            for image in images:
                image.close()
    image_features = np.concatenate(image_chunks, axis=0)
    captions = [str(caption) for row in rows for caption in row["captions"]]
    caption_features = extractor.encode_texts(captions, batch_size=text_batch_size)
    caption_image_index = np.concatenate(
        [np.full(len(row["captions"]), index, dtype=np.int32) for index, row in enumerate(rows)]
    )
    caption_index = np.concatenate(
        [np.arange(len(row["captions"]), dtype=np.int16) for row in rows]
    )
    image_ids = np.asarray([int(row["image_id"]) for row in rows], dtype=np.int64)
    metadata: dict[str, Any] = {
        "schema_version": "mosaic.features.v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "dataset": manifest["dataset"],
        "model_name": model_name,
        "model_revision": model_revision,
        "device_at_extraction": str(extractor.device),
        "image_count": int(image_features.shape[0]),
        "caption_count": int(caption_features.shape[0]),
        "embedding_dim": int(image_features.shape[1]),
        "image_dtype": str(image_features.dtype),
        "safe_npz": "allow_pickle_false",
    }
    metadata["metadata_sha256"] = _metadata_digest(metadata)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        image_ids=image_ids,
        image_features=image_features,
        caption_features=caption_features,
        caption_image_index=caption_image_index,
        caption_index=caption_index,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return metadata


def make_toy_feature_bundle(output_path: Path, *, image_count: int = 12, dim: int = 16) -> dict[str, Any]:
    """Deterministic feature bundle for tests/smoke only; never formal evidence."""

    if image_count < 3 or dim < 2:
        raise ValueError("toy feature dimensions are too small")
    rng = np.random.default_rng(20260723)
    image = rng.normal(size=(image_count, dim)).astype(np.float32)
    image /= np.linalg.norm(image, axis=1, keepdims=True)
    captions = np.repeat(image, 2, axis=0) + rng.normal(0, 0.03, size=(image_count * 2, dim)).astype(np.float32)
    captions /= np.linalg.norm(captions, axis=1, keepdims=True)
    metadata = {
        "schema_version": "mosaic.features.v1",
        "manifest_sha256": "toy",
        "dataset": "MOSAIC-toy-not-for-evaluation",
        "model_name": "synthetic",
        "device_at_extraction": "cpu",
        "image_count": image_count,
        "caption_count": image_count * 2,
        "embedding_dim": dim,
        "safe_npz": "allow_pickle_false",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        image_ids=np.arange(image_count, dtype=np.int64),
        image_features=image,
        caption_features=captions,
        caption_image_index=np.repeat(np.arange(image_count, dtype=np.int32), 2),
        caption_index=np.tile(np.arange(2, dtype=np.int16), image_count),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return metadata


def load_feature_bundle(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(Path(path), allow_pickle=False) as bundle:
        required = {
            "image_ids",
            "image_features",
            "caption_features",
            "caption_image_index",
            "caption_index",
            "metadata_json",
        }
        if set(bundle.files) != required:
            raise ValueError(f"feature bundle keys mismatch: {bundle.files}")
        metadata = json.loads(str(bundle["metadata_json"].item()))
        arrays = {key: np.asarray(bundle[key]) for key in required if key != "metadata_json"}
    if arrays["image_features"].ndim != 2 or arrays["caption_features"].ndim != 2:
        raise ValueError("feature matrices must be rank 2")
    if arrays["image_features"].shape[1] != arrays["caption_features"].shape[1]:
        raise ValueError("image/text feature widths differ")
    if arrays["caption_features"].shape[0] != arrays["caption_image_index"].size:
        raise ValueError("caption index alignment failed")
    if np.any(arrays["caption_image_index"] < 0) or np.any(
        arrays["caption_image_index"] >= arrays["image_features"].shape[0]
    ):
        raise ValueError("caption image index out of range")
    return metadata, arrays


__all__ = [
    "ClipFeatureExtractor",
    "build_feature_bundle",
    "load_feature_bundle",
    "make_toy_feature_bundle",
    "resolve_device",
]
