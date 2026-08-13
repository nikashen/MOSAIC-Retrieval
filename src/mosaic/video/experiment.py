from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from ..features import resolve_device
from ..metrics import (
    build_caption_relevance,
    build_image_relevance,
    evaluate_direction,
    metric_vectors_from_ranks,
    paired_bootstrap_delta_ci,
)
from ..models import (
    hard_negative_margin_loss,
    symmetric_contrastive_loss,
    trainable_parameter_count,
)
from .data import load_video_manifest
from .features import load_video_feature_bundle
from .models import TemporalVideoEncoder


CHECKPOINT_NAME = "video_encoder.safetensors"
CHECKPOINT_CONFIG_NAME = "video_encoder_config.json"
SUMMARY_NAME = "training_summary.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    resolved = Path(path).resolve()
    root = next(
        (candidate for candidate in (resolved.parent, *resolved.parents) if (candidate / ".git").exists()),
        None,
    )
    if root is None:
        return None
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def split_video_indices(manifest: dict[str, Any], split: str) -> np.ndarray:
    split = str(split)
    return np.asarray(
        [index for index, row in enumerate(manifest["videos"]) if row["split"] == split],
        dtype=np.int64,
    )


def _caption_rows_for_videos(
    caption_video_index: np.ndarray, video_indices: np.ndarray, video_count: int
) -> np.ndarray:
    selected = np.zeros(int(video_count), dtype=np.bool_)
    selected[np.asarray(video_indices, dtype=np.int64)] = True
    return np.flatnonzero(selected[np.asarray(caption_video_index, dtype=np.int64)]).astype(
        np.int64
    )


def _validate_alignment(
    manifest: dict[str, Any], metadata: dict[str, Any], arrays: dict[str, np.ndarray]
) -> None:
    rows = list(manifest["videos"])
    if metadata.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("feature/manifest digest mismatch")
    if metadata.get("dataset") != manifest.get("dataset"):
        raise ValueError("feature/manifest dataset mismatch")
    frames = arrays["frame_features"]
    captions = arrays["caption_features"]
    if int(metadata.get("video_count", -1)) != len(rows) or frames.shape[0] != len(rows):
        raise ValueError("feature/manifest video count mismatch")
    expected_caption_count = sum(len(row.get("captions", [])) for row in rows)
    if (
        int(metadata.get("caption_count", -1)) != expected_caption_count
        or captions.shape[0] != expected_caption_count
    ):
        raise ValueError("feature/manifest caption count mismatch")
    if int(metadata.get("frames_per_video", -1)) != frames.shape[1]:
        raise ValueError("feature metadata frame count mismatch")
    if int(metadata.get("embedding_dim", -1)) != frames.shape[2]:
        raise ValueError("feature metadata embedding width mismatch")

    expected_ids: list[int] = []
    expected_video_index: list[int] = []
    expected_caption_index: list[int] = []
    split_counts: Counter[str] = Counter()
    video_ids: set[str] = set()
    numeric_ids: set[int] = set()
    for video_index, row in enumerate(rows):
        video_id = str(row.get("video_id", ""))
        captions_for_video = row.get("captions")
        split = str(row.get("split", ""))
        if not video_id or video_id in video_ids:
            raise ValueError("manifest video ids must be non-empty and unique")
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"unsupported video split: {split}")
        if not isinstance(captions_for_video, list) or not captions_for_video:
            raise ValueError("every manifest video must contain captions")
        if any(not isinstance(value, str) or not value.strip() for value in captions_for_video):
            raise ValueError("manifest captions must be non-empty strings")
        numeric_id = int(row["numeric_id"])
        if numeric_id in numeric_ids:
            raise ValueError("manifest numeric ids must be unique")
        video_ids.add(video_id)
        numeric_ids.add(numeric_id)
        expected_ids.append(numeric_id)
        expected_video_index.extend([video_index] * len(captions_for_video))
        expected_caption_index.extend(range(len(captions_for_video)))
        split_counts[split] += 1

    declared_counts = {
        str(key): int(value)
        for key, value in manifest.get("selection", {}).get("counts", {}).items()
    }
    if declared_counts != dict(split_counts):
        raise ValueError("manifest declared split counts do not match its rows")
    if not np.array_equal(arrays["video_ids"], np.asarray(expected_ids, dtype=np.int64)):
        raise ValueError("feature/manifest video id order mismatch")
    if not np.array_equal(
        arrays["caption_video_index"], np.asarray(expected_video_index, dtype=np.int32)
    ):
        raise ValueError("feature/manifest caption-video alignment mismatch")
    if not np.array_equal(
        arrays["caption_index"], np.asarray(expected_caption_index, dtype=np.int16)
    ):
        raise ValueError("feature/manifest caption slot alignment mismatch")


def load_aligned_video_inputs(
    manifest_path: Path, feature_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    manifest = load_video_manifest(manifest_path)
    metadata, arrays = load_video_feature_bundle(feature_path)
    _validate_alignment(manifest, metadata, arrays)
    return manifest, metadata, arrays


def deterministic_epoch_caption_rows(
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    video_indices: np.ndarray,
    *,
    seed: int,
    epoch: int,
) -> np.ndarray:
    """Choose one caption per video from a stable seed/epoch/video-id hash."""

    if int(epoch) <= 0:
        raise ValueError("epoch must be positive")
    caption_video_index = np.asarray(arrays["caption_video_index"], dtype=np.int64)
    video_count = len(manifest["videos"])
    if caption_video_index.ndim != 1 or caption_video_index.size == 0:
        raise ValueError("caption-video index must be a non-empty vector")
    if np.any(caption_video_index < 0) or np.any(caption_video_index >= video_count):
        raise ValueError("caption-video index is out of range")
    if np.any(caption_video_index[1:] < caption_video_index[:-1]):
        raise ValueError("caption-video index must be grouped in manifest order")
    counts = np.bincount(caption_video_index, minlength=video_count)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(counts)))
    chosen: list[int] = []
    for video_index in np.asarray(video_indices, dtype=np.int64).tolist():
        if not 0 <= int(video_index) < video_count:
            raise ValueError("selected video index is out of range")
        start = int(offsets[video_index])
        count = int(counts[video_index])
        if count == 0:
            raise ValueError(f"video row {video_index} has no caption features")
        token = f"{int(seed)}:{int(epoch)}:{manifest['videos'][video_index]['video_id']}"
        value = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
        chosen.append(start + value % count)
    return np.asarray(chosen, dtype=np.int64)


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 1e-8):
        raise ValueError("retrieval embeddings must be finite and non-zero")
    return (values / norms).astype(np.float32)


def _frozen_video_embeddings(
    arrays: dict[str, np.ndarray], video_indices: np.ndarray, aggregator: str
) -> np.ndarray:
    frames = np.asarray(arrays["frame_features"][video_indices], dtype=np.float32)
    mask = np.asarray(arrays["frame_mask"][video_indices], dtype=np.bool_)
    if aggregator == "mean":
        pooled = (frames * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
    elif aggregator == "max":
        pooled = np.where(mask[..., None], frames, -np.inf).max(axis=1)
    else:
        raise ValueError("frozen aggregator must be 'mean' or 'max'")
    return _normalise(pooled)


def _caption_selection(
    arrays: dict[str, np.ndarray], video_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    caption_rows = _caption_rows_for_videos(
        arrays["caption_video_index"], video_indices, arrays["frame_features"].shape[0]
    )
    local = {int(global_index): index for index, global_index in enumerate(video_indices.tolist())}
    caption_video_local = np.asarray(
        [local[int(value)] for value in arrays["caption_video_index"][caption_rows]],
        dtype=np.int64,
    )
    return caption_rows, caption_video_local


def _evaluate_selected(
    video_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    caption_video_local: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    ks: Iterable[int],
) -> dict[str, Any]:
    ks_tuple = tuple(sorted({int(value) for value in ks}))
    if not ks_tuple or ks_tuple[0] <= 0:
        raise ValueError("ks must contain positive integers")
    text_to_video, text_ranks, _ = evaluate_direction(
        text_embeddings,
        video_embeddings,
        build_caption_relevance(caption_video_local, video_embeddings.shape[0]),
        ks=ks_tuple,
    )
    video_to_text, video_ranks, _ = evaluate_direction(
        video_embeddings,
        text_embeddings,
        build_image_relevance(caption_video_local, video_embeddings.shape[0]),
        ks=ks_tuple,
    )
    text_clusters = cluster_ids[caption_video_local]
    return {
        "text_to_video": text_to_video.to_dict(),
        "video_to_text": video_to_text.to_dict(),
        "ranks": {
            "text_to_video": text_ranks.astype(np.int64).tolist(),
            "video_to_text": video_ranks.astype(np.int64).tolist(),
        },
        "_ranks": {"text_to_video": text_ranks, "video_to_text": video_ranks},
        "_clusters": {"text_to_video": text_clusters, "video_to_text": cluster_ids},
    }


def _selection_score(result: dict[str, Any]) -> float:
    return float(
        np.mean(
            [
                result["text_to_video"]["recall_at"]["1"],
                result["text_to_video"]["recall_at"]["10"],
                result["video_to_text"]["recall_at"]["1"],
                result["video_to_text"]["recall_at"]["10"],
            ]
        )
    )


def _model_architecture(
    input_dim: int, frames_per_video: int, config: dict[str, Any]
) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    embedding_dim = int(model_cfg.get("embedding_dim", input_dim))
    if embedding_dim != int(input_dim):
        raise ValueError(
            "video training requires embedding_dim == input_dim for the exact mean-pool fallback"
        )
    aggregator = str(model_cfg.get("aggregator", "temporal_attention"))
    if aggregator not in {"mean", "temporal_attention"}:
        raise ValueError("trainable video aggregator must be mean or temporal_attention")
    return {
        "input_dim": int(input_dim),
        "embedding_dim": embedding_dim,
        "hidden_dim": int(model_cfg.get("hidden_dim", 256)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "temperature_init": float(model_cfg.get("temperature_init", 0.07)),
        "aggregator": aggregator,
        "max_frames": int(frames_per_video),
    }


def _build_from_architecture(
    architecture: dict[str, Any], *, device: str
) -> tuple[TemporalVideoEncoder, torch.device]:
    required = {
        "input_dim",
        "embedding_dim",
        "hidden_dim",
        "dropout",
        "temperature_init",
        "aggregator",
        "max_frames",
    }
    if set(architecture) != required or architecture["aggregator"] not in {
        "mean",
        "temporal_attention",
    }:
        raise ValueError("invalid temporal video checkpoint architecture")
    target = resolve_device(device)
    return TemporalVideoEncoder(**architecture).to(target), target


def _encode_trained(
    model: TemporalVideoEncoder,
    arrays: dict[str, np.ndarray],
    video_indices: np.ndarray,
    caption_rows: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if int(batch_size) <= 0:
        raise ValueError("encoding batch_size must be positive")
    model.eval()
    videos: list[np.ndarray] = []
    texts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, video_indices.size, int(batch_size)):
            rows = video_indices[start : start + int(batch_size)]
            frames = torch.from_numpy(
                np.asarray(arrays["frame_features"][rows], dtype=np.float32)
            ).to(device)
            mask = torch.from_numpy(np.asarray(arrays["frame_mask"][rows])).to(device)
            videos.append(model.encode_video(frames, mask).cpu().numpy())
        for start in range(0, caption_rows.size, int(batch_size)):
            rows = caption_rows[start : start + int(batch_size)]
            text = torch.from_numpy(
                np.asarray(arrays["caption_features"][rows], dtype=np.float32)
            ).to(device)
            texts.append(model.encode_text(text).cpu().numpy())
    return np.concatenate(videos).astype(np.float32), np.concatenate(texts).astype(np.float32)


def _checkpoint_payload(
    architecture: dict[str, Any],
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    *,
    epoch: int,
    score: float,
    fallback: bool,
    manifest_file_sha256: str,
    feature_npz_sha256: str,
    input_commit: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "mosaic.video_encoder.v1",
        "architecture": architecture,
        "training_manifest_sha256": manifest["manifest_sha256"],
        "training_feature_metadata_sha256": metadata["metadata_sha256"],
        "training_manifest_file_sha256": str(manifest_file_sha256),
        "training_feature_npz_sha256": str(feature_npz_sha256),
        "training_input_commit": input_commit,
        "epoch": int(epoch),
        "dev_joint_selection_score": float(score),
        "selection_split": "dev",
        "test_labels_accessed": False,
        "fallback_reason": (
            "exact_frozen_mean_pool_initialization_if_no_dev_candidate_improves"
            if fallback
            else None
        ),
    }


def _save_checkpoint(
    model: TemporalVideoEncoder, output_dir: Path, payload: dict[str, Any]
) -> None:
    tensors = {
        key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()
    }
    save_file(tensors, str(output_dir / CHECKPOINT_NAME))
    (output_dir / CHECKPOINT_CONFIG_NAME).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def load_trained_video_model(
    checkpoint_dir: Path,
    *,
    device: str = "auto",
    expected_input_dim: int | None = None,
    expected_frames: int | None = None,
) -> tuple[TemporalVideoEncoder, dict[str, Any], torch.device]:
    checkpoint_dir = Path(checkpoint_dir)
    payload = json.loads(
        (checkpoint_dir / CHECKPOINT_CONFIG_NAME).read_text(encoding="utf-8")
    )
    if payload.get("schema_version") != "mosaic.video_encoder.v1":
        raise ValueError("unsupported temporal video checkpoint schema")
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("video checkpoint is missing its architecture")
    if expected_input_dim is not None and int(architecture.get("input_dim", -1)) != int(
        expected_input_dim
    ):
        raise ValueError("checkpoint/feature embedding width mismatch")
    if expected_frames is not None and int(architecture.get("max_frames", -1)) < int(
        expected_frames
    ):
        raise ValueError("checkpoint cannot encode the feature timeline length")
    model, target = _build_from_architecture(architecture, device=device)
    state = load_file(str(checkpoint_dir / CHECKPOINT_NAME), device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload, target


def _balanced_batches(order: np.ndarray, batch_size: int) -> list[np.ndarray]:
    # InfoNCE cannot consume a singleton. Merge a potential final singleton into
    # its predecessor while keeping every sampled video in the epoch.
    batches = [order[start : start + batch_size] for start in range(0, order.size, batch_size)]
    if len(batches) > 1 and batches[-1].size == 1:
        batches[-2] = np.concatenate((batches[-2], batches[-1]))
        batches.pop()
    return batches


def train_video_encoder(
    manifest_path: Path,
    feature_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    *,
    device: str = "auto",
    epochs: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    manifest, metadata, arrays = load_aligned_video_inputs(manifest_path, feature_path)
    manifest_file_sha256 = _sha256_file(manifest_path)
    feature_npz_sha256 = _sha256_file(feature_path)
    input_commit = _git_commit(manifest_path)
    observed_splits = {str(row["split"]) for row in manifest["videos"]}
    if observed_splits != {"train", "dev"}:
        raise ValueError("training requires a train/dev-only manifest and must not read Test rows")
    train_indices = split_video_indices(manifest, "train")
    dev_indices = split_video_indices(manifest, "dev")
    if train_indices.size < 2 or dev_indices.size < 2:
        raise ValueError("train/dev split is too small")

    training_cfg = config.get("training", {})
    epochs = int(epochs if epochs is not None else training_cfg.get("epochs", 10))
    batch_size = int(
        batch_size if batch_size is not None else training_cfg.get("batch_size", 64)
    )
    run_seed = int(seed if seed is not None else training_cfg.get("seed", 20260730))
    if epochs <= 0 or batch_size < 2:
        raise ValueError("epochs must be positive and batch_size must be at least two")
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    architecture = _model_architecture(
        arrays["frame_features"].shape[2], arrays["frame_features"].shape[1], config
    )
    model, target = _build_from_architecture(architecture, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    model_cfg = config.get("model", {})
    hard_weight = float(model_cfg.get("hard_negative_weight", 0.1))
    hard_top_k = int(model_cfg.get("hard_negative_top_k", 8))
    teacher_weight = float(model_cfg.get("teacher_preservation_weight", 0.1))
    if min(hard_weight, teacher_weight) < 0 or hard_top_k <= 0:
        raise ValueError("loss weights must be non-negative and hard-negative top-k positive")

    dev_caption_rows, dev_caption_local = _caption_selection(arrays, dev_indices)
    dev_clusters = np.asarray(arrays["video_ids"][dev_indices], dtype=np.int64)
    baseline = _evaluate_selected(
        _frozen_video_embeddings(arrays, dev_indices, "mean"),
        _normalise(arrays["caption_features"][dev_caption_rows]),
        dev_caption_local,
        dev_clusters,
        ks=(1, 10),
    )
    baseline_score = _selection_score(baseline)
    threshold = float(config.get("selection", {}).get("directional_r10_max_drop", 0.002))
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("directional_r10_max_drop must be finite and non-negative")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_epoch = 0
    best_score = baseline_score
    _save_checkpoint(
        model,
        output_dir,
        _checkpoint_payload(
            architecture,
            manifest,
            metadata,
            epoch=0,
            score=baseline_score,
            fallback=True,
            manifest_file_sha256=manifest_file_sha256,
            feature_npz_sha256=feature_npz_sha256,
            input_commit=input_commit,
        ),
    )

    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        caption_rows = deterministic_epoch_caption_rows(
            manifest, arrays, train_indices, seed=run_seed, epoch=epoch
        )
        rng = np.random.default_rng(run_seed + epoch)
        order = rng.permutation(train_indices.size)
        model.train()
        totals = {"loss": 0.0, "contrastive": 0.0, "hard_negative": 0.0, "teacher": 0.0}
        steps = 0
        for batch_order in _balanced_batches(order, batch_size):
            video_rows = train_indices[batch_order]
            text_rows = caption_rows[batch_order]
            frames = torch.from_numpy(
                np.asarray(arrays["frame_features"][video_rows], dtype=np.float32)
            ).to(target)
            mask = torch.from_numpy(np.asarray(arrays["frame_mask"][video_rows])).to(target)
            raw_text = torch.from_numpy(
                np.asarray(arrays["caption_features"][text_rows], dtype=np.float32)
            ).to(target)
            valid = mask.to(dtype=frames.dtype)
            teacher_video = F.normalize(
                (frames * valid.unsqueeze(-1)).sum(dim=1)
                / valid.sum(dim=1, keepdim=True),
                dim=-1,
            )
            teacher_text = F.normalize(raw_text, dim=-1)

            result = model(frames, raw_text, mask)
            contrastive = symmetric_contrastive_loss(
                result["video"], result["text"], result["temperature"]
            )
            hard = hard_negative_margin_loss(
                result["video"], result["text"], top_k=hard_top_k
            )
            teacher = (
                (1.0 - F.cosine_similarity(result["video"], teacher_video, dim=-1)).mean()
                + (1.0 - F.cosine_similarity(result["text"], teacher_text, dim=-1)).mean()
            )
            loss = contrastive + hard_weight * hard + teacher_weight * teacher
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("video training produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("video training produced non-finite gradients")
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["contrastive"] += float(contrastive.detach().cpu())
            totals["hard_negative"] += float(hard.detach().cpu())
            totals["teacher"] += float(teacher.detach().cpu())
            steps += 1

        dev_video, dev_text = _encode_trained(
            model,
            arrays,
            dev_indices,
            dev_caption_rows,
            device=target,
            batch_size=max(1, batch_size),
        )
        dev_result = _evaluate_selected(
            dev_video, dev_text, dev_caption_local, dev_clusters, ks=(1, 10)
        )
        score = _selection_score(dev_result)
        passes_gate = bool(
            dev_result["text_to_video"]["recall_at"]["10"]
            >= baseline["text_to_video"]["recall_at"]["10"] - threshold
            and dev_result["video_to_text"]["recall_at"]["10"]
            >= baseline["video_to_text"]["recall_at"]["10"] - threshold
        )
        entry = {
            "epoch": epoch,
            "sampled_caption_rows_sha256": hashlib.sha256(
                caption_rows.tobytes(order="C")
            ).hexdigest(),
            "sampled_captions": int(caption_rows.size),
            **{f"train_{key}": value / max(1, steps) for key, value in totals.items()},
            "dev_text_to_video_recall@1": dev_result["text_to_video"]["recall_at"]["1"],
            "dev_text_to_video_recall@10": dev_result["text_to_video"]["recall_at"]["10"],
            "dev_video_to_text_recall@1": dev_result["video_to_text"]["recall_at"]["1"],
            "dev_video_to_text_recall@10": dev_result["video_to_text"]["recall_at"]["10"],
            "dev_joint_selection_score": score,
            "passes_directional_gate": passes_gate,
        }
        history.append(entry)
        if passes_gate and score > best_score:
            best_epoch = epoch
            best_score = score
            _save_checkpoint(
                model,
                output_dir,
                _checkpoint_payload(
                    architecture,
                    manifest,
                    metadata,
                    epoch=epoch,
                    score=score,
                    fallback=False,
                    manifest_file_sha256=manifest_file_sha256,
                    feature_npz_sha256=feature_npz_sha256,
                    input_commit=input_commit,
                ),
            )

    summary = {
        "schema_version": "mosaic.video_training.v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "feature_metadata_sha256": metadata["metadata_sha256"],
        "manifest_file_sha256": manifest_file_sha256,
        "feature_npz_sha256": feature_npz_sha256,
        "input_commit": input_commit,
        "train_videos": int(train_indices.size),
        "dev_videos": int(dev_indices.size),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "selection": {
            "split": "dev",
            "metric": "mean(text_to_video R@1/R@10, video_to_text R@1/R@10)",
            "baseline": "frozen_clip_mean_pool",
            "baseline_score": baseline_score,
            "best_score": best_score,
            "baseline_text_to_video": baseline["text_to_video"],
            "baseline_video_to_text": baseline["video_to_text"],
            "directional_r10_max_drop": threshold,
            "test_labels_accessed": False,
        },
        "loss": {
            "symmetric_info_nce": True,
            "hard_negative_weight": hard_weight,
            "hard_negative_top_k": hard_top_k,
            "teacher_preservation_weight": teacher_weight,
        },
        "caption_sampling": "one deterministic sha256(seed:epoch:video_id) caption per video per epoch",
        "architecture": architecture,
        "trainable_parameters": trainable_parameter_count(model),
        "device": str(target),
        "seed": run_seed,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }
    (output_dir / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _paired_vs_mean(
    trained: dict[str, Any],
    baseline: dict[str, Any],
    *,
    ks: tuple[int, ...],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates < 100:
        raise ValueError("at least 100 paired bootstrap replicates are required")
    output: dict[str, Any] = {}
    offset = 0
    for direction in ("text_to_video", "video_to_text"):
        left = metric_vectors_from_ranks(trained["_ranks"][direction], ks)
        right = metric_vectors_from_ranks(baseline["_ranks"][direction], ks)
        if not np.array_equal(
            trained["_clusters"][direction], baseline["_clusters"][direction]
        ):
            raise ValueError("trained/baseline video clusters are not paired")
        output[direction] = {}
        for key in [*(f"recall@{value}" for value in ks), "mrr"]:
            output[direction][key] = paired_bootstrap_delta_ci(
                left[key],
                right[key],
                trained["_clusters"][direction],
                replicates=replicates,
                seed=seed + offset,
            )
            offset += 1
    return output


def evaluate_video_model(
    manifest_path: Path,
    feature_path: Path,
    config: dict[str, Any],
    *,
    checkpoint_dir: Path | None = None,
    aggregator: str = "mean",
    device: str = "auto",
    split: str = "test",
    batch_size: int = 512,
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    manifest, metadata, arrays = load_aligned_video_inputs(manifest_path, feature_path)
    video_indices = split_video_indices(manifest, split)
    if video_indices.size == 0:
        raise ValueError(f"split {split} is empty")
    caption_rows, caption_video_local = _caption_selection(arrays, video_indices)
    cluster_ids = np.asarray(arrays["video_ids"][video_indices], dtype=np.int64)
    ks = tuple(int(value) for value in config.get("evaluation", {}).get("ks", (1, 5, 10, 50)))

    checkpoint_payload: dict[str, Any] | None = None
    if checkpoint_dir is None:
        if aggregator not in {"mean", "max"}:
            raise ValueError("frozen evaluation supports only mean or max aggregation")
        video_embeddings = _frozen_video_embeddings(arrays, video_indices, aggregator)
        text_embeddings = _normalise(arrays["caption_features"][caption_rows])
        model_name = f"frozen_clip_{aggregator}_pool"
    else:
        model, checkpoint_payload, target = load_trained_video_model(
            checkpoint_dir,
            device=device,
            expected_input_dim=arrays["frame_features"].shape[2],
            expected_frames=arrays["frame_features"].shape[1],
        )
        video_embeddings, text_embeddings = _encode_trained(
            model,
            arrays,
            video_indices,
            caption_rows,
            device=target,
            batch_size=batch_size,
        )
        model_name = f"mosaic_trained_{checkpoint_payload['architecture']['aggregator']}"

    result = _evaluate_selected(
        video_embeddings, text_embeddings, caption_video_local, cluster_ids, ks=ks
    )
    result.update(
        {
            "schema_version": "mosaic.video_evaluation.v1",
            "model_name": model_name,
            "split": str(split),
            "catalog_scope": "full selected split catalog",
            "manifest_sha256": manifest["manifest_sha256"],
            "feature_metadata_sha256": metadata["metadata_sha256"],
            "videos": int(video_indices.size),
            "captions": int(caption_rows.size),
        }
    )
    if checkpoint_payload is not None:
        mean_baseline = _evaluate_selected(
            _frozen_video_embeddings(arrays, video_indices, "mean"),
            _normalise(arrays["caption_features"][caption_rows]),
            caption_video_local,
            cluster_ids,
            ks=ks,
        )
        evaluation_cfg = config.get("evaluation", {})
        replicates = int(
            bootstrap_replicates
            if bootstrap_replicates is not None
            else evaluation_cfg.get("bootstrap_replicates", 1000)
        )
        result["checkpoint_epoch"] = int(checkpoint_payload["epoch"])
        result["frozen_mean_reference"] = {
            "text_to_video": mean_baseline["text_to_video"],
            "video_to_text": mean_baseline["video_to_text"],
        }
        result["paired_video_cluster_bootstrap_vs_frozen_mean"] = _paired_vs_mean(
            result,
            mean_baseline,
            ks=ks,
            replicates=replicates,
            seed=int(evaluation_cfg.get("bootstrap_seed", 20260731)),
        )
    return result


def evaluate_video_suite(
    manifest_path: Path,
    feature_path: Path,
    config: dict[str, Any],
    *,
    checkpoint_dir: Path,
    device: str = "auto",
    split: str = "test",
    batch_size: int = 512,
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    """Load one split once and compare mean/max/trained under one catalog."""

    manifest, metadata, arrays = load_aligned_video_inputs(manifest_path, feature_path)
    video_indices = split_video_indices(manifest, split)
    if video_indices.size == 0:
        raise ValueError(f"split {split} is empty")
    caption_rows, caption_video_local = _caption_selection(arrays, video_indices)
    cluster_ids = np.asarray(arrays["video_ids"][video_indices], dtype=np.int64)
    ks = tuple(int(value) for value in config.get("evaluation", {}).get("ks", (1, 5, 10, 50)))
    raw_text = _normalise(arrays["caption_features"][caption_rows])
    mean_result = _evaluate_selected(
        _frozen_video_embeddings(arrays, video_indices, "mean"),
        raw_text,
        caption_video_local,
        cluster_ids,
        ks=ks,
    )
    max_result = _evaluate_selected(
        _frozen_video_embeddings(arrays, video_indices, "max"),
        raw_text,
        caption_video_local,
        cluster_ids,
        ks=ks,
    )
    model, checkpoint_payload, target = load_trained_video_model(
        checkpoint_dir,
        device=device,
        expected_input_dim=arrays["frame_features"].shape[2],
        expected_frames=arrays["frame_features"].shape[1],
    )
    trained_video, trained_text = _encode_trained(
        model,
        arrays,
        video_indices,
        caption_rows,
        device=target,
        batch_size=batch_size,
    )
    trained_result = _evaluate_selected(
        trained_video,
        trained_text,
        caption_video_local,
        cluster_ids,
        ks=ks,
    )
    evaluation_cfg = config.get("evaluation", {})
    replicates = int(
        bootstrap_replicates
        if bootstrap_replicates is not None
        else evaluation_cfg.get("bootstrap_replicates", 1000)
    )
    paired = _paired_vs_mean(
        trained_result,
        mean_result,
        ks=ks,
        replicates=replicates,
        seed=int(evaluation_cfg.get("bootstrap_seed", 20260731)),
    )

    def public(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "text_to_video": result["text_to_video"],
            "video_to_text": result["video_to_text"],
            "rank_sha256": {
                direction: hashlib.sha256(
                    np.asarray(result["_ranks"][direction], dtype="<i8").tobytes(order="C")
                ).hexdigest()
                for direction in ("text_to_video", "video_to_text")
            },
        }

    return {
        "schema_version": "mosaic.video_evaluation_suite.v1",
        "split": str(split),
        "catalog_scope": "full selected split catalog",
        "manifest_sha256": manifest["manifest_sha256"],
        "feature_metadata_sha256": metadata["metadata_sha256"],
        "videos": int(video_indices.size),
        "captions": int(caption_rows.size),
        "checkpoint_epoch": int(checkpoint_payload["epoch"]),
        "models": {
            "frozen_clip_mean_pool": public(mean_result),
            "frozen_clip_max_pool": public(max_result),
            "mosaic_trained_temporal_attention": public(trained_result),
        },
        "paired_video_cluster_bootstrap_trained_vs_mean": paired,
    }


def strip_internal(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("_")}


__all__ = [
    "CHECKPOINT_CONFIG_NAME",
    "CHECKPOINT_NAME",
    "SUMMARY_NAME",
    "deterministic_epoch_caption_rows",
    "evaluate_video_model",
    "evaluate_video_suite",
    "load_aligned_video_inputs",
    "load_trained_video_model",
    "split_video_indices",
    "strip_internal",
    "train_video_encoder",
]
