from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from ..features import ClipFeatureExtractor
from .data import load_video_manifest


SCHEMA = "mosaic.video_features.v1"


def _metadata_digest(metadata: dict[str, Any]) -> str:
    clean = {key: value for key, value in metadata.items() if key != "metadata_sha256"}
    return hashlib.sha256(
        json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def uniform_midpoint_indices(frame_count: int, samples: int) -> np.ndarray:
    """Return deterministic midpoint samples over equal frame-count bins."""

    frame_count = int(frame_count)
    samples = int(samples)
    if frame_count <= 0 or samples <= 0:
        raise ValueError("frame_count and samples must be positive")
    positions = np.floor((np.arange(samples, dtype=np.float64) + 0.5) * frame_count / samples)
    return np.clip(positions.astype(np.int64), 0, frame_count - 1)


def decode_uniform_frames(path: Path, *, frames_per_video: int = 12) -> tuple[list[Image.Image], dict[str, Any]]:
    """Decode bounded RGB samples with imageio's bundled FFmpeg binary.

    MSR-VTT clips expose duration and FPS in the first FFmpeg metadata record.
    Sampling indices are computed from that declared duration and decoded in a
    single streaming pass. If a malformed container ends slightly early, the
    final decoded frame is repeated and the fallback is recorded explicitly.
    """

    import imageio_ffmpeg

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    frames_per_video = int(frames_per_video)
    if frames_per_video <= 0:
        raise ValueError("frames_per_video must be positive")
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    images: list[Image.Image] = []
    last_image: Image.Image | None = None
    decoded = 0
    repeated = 0
    try:
        metadata = next(reader)
        width, height = (int(value) for value in metadata["size"])
        fps = float(metadata.get("fps") or 0.0)
        duration = float(metadata.get("duration") or 0.0)
        if width <= 0 or height <= 0 or fps <= 0 or duration <= 0:
            raise ValueError(f"invalid video metadata for {path.name}: {metadata}")
        estimated_frames = max(1, int(round(fps * duration)))
        targets = uniform_midpoint_indices(estimated_frames, frames_per_video)
        target_cursor = 0
        for frame_index, frame in enumerate(reader):
            decoded = frame_index + 1
            if target_cursor >= targets.size:
                break
            if frame_index < int(targets[target_cursor]):
                continue
            current = Image.frombytes("RGB", (width, height), frame)
            if last_image is not None:
                last_image.close()
            last_image = current
            while target_cursor < targets.size and int(targets[target_cursor]) <= frame_index:
                images.append(current.copy())
                target_cursor += 1
            if target_cursor >= targets.size:
                break
        if last_image is None:
            raise ValueError(f"video contains no decodable frames: {path}")
        while len(images) < frames_per_video:
            images.append(last_image.copy())
            repeated += 1
    except Exception:
        for image in images:
            image.close()
        raise
    finally:
        if last_image is not None:
            last_image.close()
        reader.close()
    diagnostics = {
        "fps": fps,
        "duration_seconds": duration,
        "source_width": width,
        "source_height": height,
        "estimated_frames": estimated_frames,
        "decoded_through_frame": decoded,
        "repeated_tail_samples": repeated,
    }
    return images, diagnostics


def _resolve_video_root(manifest: dict[str, Any]) -> Path:
    configured = os.environ.get("MOSAIC_VIDEO_ROOT")
    root = Path(configured or manifest["video_root"]).resolve()
    if not root.is_dir() and not configured:
        portable = (Path.cwd() / "data" / "raw" / "msrvtt" / "video").resolve()
        if portable.is_dir():
            root = portable
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def _video_path(root: Path, row: dict[str, Any]) -> Path:
    resolved_root = Path(root).resolve()
    path = (resolved_root / Path(str(row["file_name"]))).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError:
        raise FileNotFoundError(path) from None
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _caption_layout(rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    captions = [str(caption) for row in rows for caption in row["captions"]]
    video_index = np.concatenate(
        [np.full(len(row["captions"]), index, dtype=np.int32) for index, row in enumerate(rows)]
    )
    caption_index = np.concatenate(
        [np.arange(len(row["captions"]), dtype=np.int16) for row in rows]
    )
    return captions, video_index, caption_index


def _decode_video_batch(
    rows: list[dict[str, Any]],
    root: Path,
    *,
    frames_per_video: int,
    decode_workers: int,
    decoder: Callable[..., tuple[list[Image.Image], dict[str, Any]]] = decode_uniform_frames,
) -> list[tuple[list[Image.Image], dict[str, Any]]]:
    """Decode concurrently while preserving manifest order and closing failures."""

    if not rows:
        return []
    workers = min(max(1, int(decode_workers)), len(rows))
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mosaic-video")
    futures: dict[Future[tuple[list[Image.Image], dict[str, Any]]], int] = {
        executor.submit(
            decoder,
            _video_path(root, row),
            frames_per_video=int(frames_per_video),
        ): index
        for index, row in enumerate(rows)
    }
    ordered: list[tuple[list[Image.Image], dict[str, Any]] | None] = [None] * len(rows)
    try:
        for future in as_completed(futures):
            ordered[futures[future]] = future.result()
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        # Futures that finished before or during shutdown may own live PIL
        # objects even if their result was never consumed by as_completed.
        for future in futures:
            if future.cancelled() or not future.done():
                continue
            try:
                frames, _ = future.result()
            except BaseException:
                continue
            for image in frames:
                image.close()
        raise
    else:
        executor.shutdown(wait=True)
    if any(value is None for value in ordered):  # pragma: no cover - defensive
        raise RuntimeError("video decoder batch completed without every result")
    return [value for value in ordered if value is not None]


def build_video_feature_bundle(
    manifest_path: Path,
    output_path: Path,
    *,
    model_name: str = "openai/clip-vit-base-patch32",
    model_revision: str | None = None,
    device: str = "auto",
    cache_dir: str | None = None,
    local_files_only: bool = False,
    frames_per_video: int = 12,
    video_batch_size: int = 4,
    decode_workers: int = 4,
    image_batch_size: int = 32,
    text_batch_size: int = 256,
    max_videos: int | None = None,
) -> dict[str, Any]:
    """Stream video/frame features into a resumable, pickle-free NPZ bundle."""

    manifest = load_video_manifest(manifest_path)
    rows = list(manifest["videos"])
    if max_videos is not None:
        if int(max_videos) <= 0:
            raise ValueError("max_videos must be positive")
        rows = rows[: int(max_videos)]
    if not rows:
        raise ValueError("video feature manifest is empty")
    if min(
        int(frames_per_video),
        int(video_batch_size),
        int(decode_workers),
        int(image_batch_size),
        int(text_batch_size),
    ) <= 0:
        raise ValueError("feature extraction sizes must be positive")

    extractor = ClipFeatureExtractor(
        model_name,
        device=device,
        cache_dir=cache_dir,
        revision=model_revision,
        local_files_only=local_files_only,
    )
    embedding_dim = extractor.output_dim
    root = _resolve_video_root(manifest)
    captions, caption_video_index, caption_index = _caption_layout(rows)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work = output_path.with_suffix(output_path.suffix + ".work")
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / "progress.json"
    expected_state: dict[str, Any] = {
        "schema_version": "mosaic.video_feature_progress.v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "model_name": str(model_name),
        "model_revision": str(model_revision) if model_revision else None,
        "video_count": len(rows),
        "caption_count": len(captions),
        "frames_per_video": int(frames_per_video),
        "embedding_dim": int(embedding_dim),
        "storage_dtype": "float16",
    }
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for key, value in expected_state.items():
            if state.get(key) != value:
                raise ValueError(f"incompatible resumable feature state for {key}")
    else:
        state = {
            **expected_state,
            "video_completed": 0,
            "caption_completed": 0,
            "videos_with_repeated_tail": 0,
            "repeated_tail_samples": 0,
        }
        np.lib.format.open_memmap(
            work / "frame_features.npy",
            mode="w+",
            dtype=np.float16,
            shape=(len(rows), int(frames_per_video), embedding_dim),
        ).flush()
        np.lib.format.open_memmap(
            work / "caption_features.npy",
            mode="w+",
            dtype=np.float16,
            shape=(len(captions), embedding_dim),
        ).flush()
        _atomic_json(state_path, state)

    frame_store = np.lib.format.open_memmap(work / "frame_features.npy", mode="r+")
    caption_store = np.lib.format.open_memmap(work / "caption_features.npy", mode="r+")
    video_start = int(state.get("video_completed", 0))
    for start in range(video_start, len(rows), int(video_batch_size)):
        stop = min(len(rows), start + int(video_batch_size))
        decoded: list[Image.Image] = []
        diagnostics: list[dict[str, Any]] = []
        try:
            groups = _decode_video_batch(
                rows[start:stop],
                root,
                frames_per_video=int(frames_per_video),
                decode_workers=int(decode_workers),
            )
            for frames, detail in groups:
                decoded.extend(frames)
                diagnostics.append(detail)
            features = extractor.encode_images(decoded, batch_size=int(image_batch_size))
            frame_store[start:stop] = features.reshape(
                stop - start, int(frames_per_video), embedding_dim
            ).astype(np.float16)
            frame_store.flush()
        finally:
            for image in decoded:
                image.close()
        state["video_completed"] = stop
        state["videos_with_repeated_tail"] = int(
            state.get("videos_with_repeated_tail", 0)
        ) + sum(int(detail["repeated_tail_samples"]) > 0 for detail in diagnostics)
        state["repeated_tail_samples"] = int(state.get("repeated_tail_samples", 0)) + sum(
            int(detail["repeated_tail_samples"]) for detail in diagnostics
        )
        _atomic_json(state_path, state)

    caption_start = int(state.get("caption_completed", 0))
    for start in range(caption_start, len(captions), int(text_batch_size)):
        stop = min(len(captions), start + int(text_batch_size))
        caption_store[start:stop] = extractor.encode_texts(
            captions[start:stop], batch_size=int(text_batch_size)
        ).astype(np.float16)
        caption_store.flush()
        state["caption_completed"] = stop
        _atomic_json(state_path, state)

    frame_features = np.load(work / "frame_features.npy", mmap_mode="r", allow_pickle=False)
    caption_features = np.load(work / "caption_features.npy", mmap_mode="r", allow_pickle=False)
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "dataset": manifest["dataset"],
        "model_name": str(model_name),
        "model_revision": str(model_revision) if model_revision else None,
        "device_at_extraction": str(extractor.device),
        "video_count": len(rows),
        "caption_count": len(captions),
        "frames_per_video": int(frames_per_video),
        "embedding_dim": int(embedding_dim),
        "sampling": "uniform midpoint of equal-duration bins",
        "decode_backend": "imageio-ffmpeg bundled binary",
        "storage_dtype": "float16",
        "safe_npz": "allow_pickle_false",
        "decode_audit": {
            "videos_with_repeated_tail": int(state.get("videos_with_repeated_tail", 0)),
            "repeated_tail_samples": int(state.get("repeated_tail_samples", 0)),
        },
    }
    metadata["metadata_sha256"] = _metadata_digest(metadata)
    np.savez_compressed(
        output_path,
        video_ids=np.asarray([int(row["numeric_id"]) for row in rows], dtype=np.int64),
        frame_features=frame_features,
        frame_mask=np.ones((len(rows), int(frames_per_video)), dtype=np.uint8),
        caption_features=caption_features,
        caption_video_index=caption_video_index,
        caption_index=caption_index,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return metadata


def make_toy_video_feature_bundle(
    output_path: Path,
    *,
    video_count: int = 12,
    frames_per_video: int = 4,
    captions_per_video: int = 2,
    dim: int = 16,
    manifest_sha256: str = "toy",
    dataset: str = "MOSAIC-video-toy-not-for-evaluation",
) -> dict[str, Any]:
    if video_count < 3 or frames_per_video < 2 or captions_per_video < 1 or dim < 2:
        raise ValueError("toy video feature dimensions are too small")
    rng = np.random.default_rng(20260730)
    centers = rng.normal(size=(video_count, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    frames = centers[:, None, :] + rng.normal(
        0, 0.025, size=(video_count, frames_per_video, dim)
    ).astype(np.float32)
    frames /= np.linalg.norm(frames, axis=-1, keepdims=True)
    captions = np.repeat(centers, captions_per_video, axis=0) + rng.normal(
        0, 0.025, size=(video_count * captions_per_video, dim)
    ).astype(np.float32)
    captions /= np.linalg.norm(captions, axis=1, keepdims=True)
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA,
        "manifest_sha256": str(manifest_sha256),
        "dataset": str(dataset),
        "model_name": "synthetic",
        "model_revision": None,
        "device_at_extraction": "cpu",
        "video_count": video_count,
        "caption_count": video_count * captions_per_video,
        "frames_per_video": frames_per_video,
        "embedding_dim": dim,
        "sampling": "synthetic",
        "decode_backend": "none",
        "storage_dtype": "float32",
        "safe_npz": "allow_pickle_false",
    }
    metadata["metadata_sha256"] = _metadata_digest(metadata)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        video_ids=np.arange(video_count, dtype=np.int64),
        frame_features=frames,
        frame_mask=np.ones((video_count, frames_per_video), dtype=np.uint8),
        caption_features=captions,
        caption_video_index=np.repeat(np.arange(video_count, dtype=np.int32), captions_per_video),
        caption_index=np.tile(np.arange(captions_per_video, dtype=np.int16), video_count),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return metadata


def load_video_feature_bundle(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    required = {
        "video_ids",
        "frame_features",
        "frame_mask",
        "caption_features",
        "caption_video_index",
        "caption_index",
        "metadata_json",
    }
    with np.load(Path(path), allow_pickle=False) as bundle:
        if set(bundle.files) != required:
            raise ValueError(f"video feature bundle keys mismatch: {bundle.files}")
        metadata = json.loads(str(bundle["metadata_json"].item()))
        arrays = {key: np.asarray(bundle[key]) for key in required if key != "metadata_json"}
    if metadata.get("schema_version") != SCHEMA:
        raise ValueError("unsupported video feature schema")
    if metadata.get("metadata_sha256") != _metadata_digest(metadata):
        raise ValueError("video feature metadata digest mismatch")
    frames = arrays["frame_features"]
    mask = arrays["frame_mask"]
    captions = arrays["caption_features"]
    if frames.ndim != 3 or captions.ndim != 2 or frames.shape[2] != captions.shape[1]:
        raise ValueError("video frame/text feature shapes are invalid")
    if mask.shape != frames.shape[:2] or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("frame mask is invalid")
    if np.any(mask.sum(axis=1) <= 0):
        raise ValueError("every video requires at least one frame")
    if arrays["video_ids"].shape != (frames.shape[0],):
        raise ValueError("video id alignment failed")
    if captions.shape[0] != arrays["caption_video_index"].size or captions.shape[0] != arrays["caption_index"].size:
        raise ValueError("caption alignment failed")
    if np.any(arrays["caption_video_index"] < 0) or np.any(
        arrays["caption_video_index"] >= frames.shape[0]
    ):
        raise ValueError("caption video index out of range")
    if not np.isfinite(frames).all() or not np.isfinite(captions).all():
        raise ValueError("video features must be finite")
    return metadata, arrays


__all__ = [
    "SCHEMA",
    "build_video_feature_bundle",
    "decode_uniform_frames",
    "load_video_feature_bundle",
    "make_toy_video_feature_bundle",
    "uniform_midpoint_indices",
]
