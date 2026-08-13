from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .data import load_manifest
from .features import load_feature_bundle, resolve_device
from .metrics import (
    aggregate_ranks,
    bootstrap_ci,
    build_caption_relevance,
    build_image_relevance,
    evaluate_direction,
    metric_vectors_from_ranks,
    paired_bootstrap_delta_ci,
)
from .models import (
    MosaicDualEncoder,
    hard_negative_margin_loss,
    modality_dropout,
    symmetric_contrastive_loss,
    trainable_parameter_count,
)


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _split_indices(manifest: dict[str, Any], split: str) -> np.ndarray:
    return np.asarray(
        [index for index, row in enumerate(manifest["images"]) if row["split"] == split],
        dtype=np.int64,
    )


def _caption_indices_for_images(caption_image_index: np.ndarray, image_indices: np.ndarray) -> np.ndarray:
    lookup = np.zeros(int(caption_image_index.max()) + 1, dtype=np.bool_)
    lookup[image_indices] = True
    return np.flatnonzero(lookup[np.asarray(caption_image_index, dtype=np.int64)]).astype(np.int64)


def _caption_groups(
    caption_image_index: np.ndarray, image_indices: np.ndarray
) -> dict[int, np.ndarray]:
    selected = _caption_indices_for_images(caption_image_index, image_indices)
    groups: dict[int, np.ndarray] = {}
    for image_index in image_indices.tolist():
        groups[int(image_index)] = selected[caption_image_index[selected] == int(image_index)]
    return groups


def _mean_caption_features(
    caption_features: np.ndarray,
    caption_image_index: np.ndarray,
    image_count: int,
    *,
    excluded_caption_index: int | None = None,
    caption_index: np.ndarray | None = None,
) -> np.ndarray:
    output = np.zeros((int(image_count), caption_features.shape[1]), dtype=np.float32)
    for image_index in range(int(image_count)):
        mask = caption_image_index == image_index
        if excluded_caption_index is not None:
            if caption_index is None:
                raise ValueError("caption_index is required for leave-one-caption-out")
            mask &= caption_index != int(excluded_caption_index)
        if not np.any(mask):
            raise ValueError(f"image {image_index} has no metadata captions")
        value = np.asarray(caption_features[mask], dtype=np.float32).mean(axis=0)
        norm = float(np.linalg.norm(value))
        if norm <= 1e-8:
            raise ValueError("caption mean has zero norm")
        output[image_index] = value / norm
    return output


def _encode_model(
    model: MosaicDualEncoder,
    arrays: dict[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    model.eval()
    image_raw = torch.from_numpy(np.asarray(arrays["image_features"], dtype=np.float32))
    text_raw = torch.from_numpy(np.asarray(arrays["caption_features"], dtype=np.float32))
    image_out: list[np.ndarray] = []
    text_out: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, image_raw.shape[0], max(1, int(batch_size))):
            image_out.append(
                model.encode_image(image_raw[start : start + batch_size].to(device)).cpu().numpy()
            )
        for start in range(0, text_raw.shape[0], max(1, int(batch_size))):
            text_out.append(
                model.encode_text(text_raw[start : start + batch_size].to(device)).cpu().numpy()
            )
    image = np.concatenate(image_out, axis=0).astype(np.float32)
    text = np.concatenate(text_out, axis=0).astype(np.float32)
    return {"image": image, "text": text}


def evaluate_caption_folds(
    encoded: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    image_indices: np.ndarray,
    *,
    ks: Iterable[int] = (1, 5, 10, 50),
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Evaluate standard and modality-ablation text→image retrieval."""

    image_indices = np.asarray(image_indices, dtype=np.int64)
    selected_caption_global = _caption_indices_for_images(arrays["caption_image_index"], image_indices)
    # Local target positions are compact and stable across the selected split.
    local_image = {int(global_index): local for local, global_index in enumerate(image_indices.tolist())}
    target_image = encoded["image"][image_indices]
    target_caption = encoded["text"][selected_caption_global]
    target_caption_image = np.asarray(
        [local_image[int(value)] for value in arrays["caption_image_index"][selected_caption_global]],
        dtype=np.int64,
    )
    selected_slots = arrays["caption_index"][selected_caption_global]
    # Standard image -> caption direction uses all five captions and no fused
    # target; it is the directly comparable CLIP/dual-encoder benchmark.
    image_relevance = build_image_relevance(target_caption_image, len(image_indices))
    image_metrics, image_ranks, image_ndcg = evaluate_direction(
        encoded["image"][image_indices],
        target_caption,
        image_relevance,
        ks=ks,
        chunk_size=chunk_size,
    )

    rank_by_mode: dict[str, list[np.ndarray]] = {"image_only": [], "text_only": [], "full": []}
    ndcg_by_mode: dict[str, list[np.ndarray]] = {key: [] for key in rank_by_mode}
    clusters_by_mode: dict[str, list[np.ndarray]] = {key: [] for key in rank_by_mode}
    fold_ids = sorted({int(value) for value in selected_slots.tolist()})
    for slot in fold_ids:
        query_mask = selected_slots == slot
        query_global = selected_caption_global[query_mask]
        query = encoded["text"][query_global]
        # Build metadata from raw CLIP captions excluding the query slot.
        metadata_raw = _mean_caption_features(
            arrays["caption_features"],
            arrays["caption_image_index"],
            arrays["image_features"].shape[0],
            excluded_caption_index=slot,
            caption_index=arrays["caption_index"],
        )
        # The caller may supply an encoded model, so metadata must be projected
        # by using the same model outside this function. A preprojected override
        # is attached by evaluate_model below.
        if "metadata_by_slot" not in encoded:
            raise ValueError("encoded metadata_by_slot is required for fused evaluation")
        metadata_projected = encoded["metadata_by_slot"][int(slot)][image_indices]
        for mode, target in (
            ("image_only", target_image),
            ("text_only", metadata_projected),
            (
                "full",
                encoded["full_by_slot"][int(slot)][image_indices],
            ),
        ):
            relevance = build_caption_relevance(
                target_image_indices := np.asarray(
                    [local_image[int(value)] for value in arrays["caption_image_index"][query_global]],
                    dtype=np.int64,
                ),
                len(image_indices),
            )
            metrics, ranks, ndcg = evaluate_direction(
                query,
                target,
                relevance,
                ks=ks,
                chunk_size=chunk_size,
            )
            rank_by_mode[mode].append(ranks)
            ndcg_by_mode[mode].append(ndcg)
            clusters_by_mode[mode].append(target_image_indices)

    result: dict[str, Any] = {
        "image_to_text": image_metrics.to_dict(),
        "image_to_text_ndcg@1000": float(image_ndcg.mean()),
        "text_to_image": {},
        "_ranks": {},
        "_clusters": {},
        "_ndcg": {},
    }
    for mode in rank_by_mode:
        ranks = np.concatenate(rank_by_mode[mode])
        clusters = np.concatenate(clusters_by_mode[mode])
        ndcg = np.concatenate(ndcg_by_mode[mode])
        result["text_to_image"][mode] = aggregate_ranks(ranks, ks).to_dict()
        result["text_to_image"][mode]["ndcg@1000"] = float(ndcg.mean())
        result["_ranks"][mode] = ranks
        result["_clusters"][mode] = clusters
        result["_ndcg"][mode] = ndcg
    result["_ranks"]["image_to_text"] = image_ranks
    result["_clusters"]["image_to_text"] = np.arange(len(image_ranks), dtype=np.int64)
    return result


def build_model(input_dim: int, config: dict[str, Any], *, device: str = "auto") -> tuple[MosaicDualEncoder, torch.device]:
    model_cfg = config.get("model", config)
    model = MosaicDualEncoder(
        input_dim=int(input_dim),
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        hidden_dim=int(model_cfg.get("projection_hidden_dim", 512)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        temperature_init=float(model_cfg.get("temperature_init", 0.07)),
    )
    target = resolve_device(device)
    return model.to(target), target


def _sample_training_pairs(
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    train_indices: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    groups = _caption_groups(arrays["caption_image_index"], train_indices)
    image_rows: list[int] = []
    caption_rows: list[int] = []
    for image_index in train_indices.tolist():
        choices = groups[int(image_index)]
        if choices.size == 0:
            raise ValueError(f"training image {image_index} has no captions")
        image_rows.append(int(image_index))
        caption_rows.append(int(rng.choice(choices)))
    return np.asarray(image_rows, dtype=np.int64), np.asarray(caption_rows, dtype=np.int64)


def train_adapter(
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
    manifest = load_manifest(manifest_path)
    metadata, arrays = load_feature_bundle(feature_path)
    if int(metadata["image_count"]) != len(manifest["images"]):
        raise ValueError("feature/manifest image count mismatch")
    train_indices = _split_indices(manifest, "train")
    dev_indices = _split_indices(manifest, "dev")
    if train_indices.size < 2 or dev_indices.size < 2:
        raise ValueError("train/dev split is too small")
    train_cfg = config.get("training", {})
    epochs = int(epochs if epochs is not None else train_cfg.get("epochs", 8))
    batch_size = int(batch_size if batch_size is not None else train_cfg.get("batch_size", 32))
    accumulation = int(train_cfg.get("gradient_accumulation", 1))
    if epochs <= 0 or batch_size <= 0 or accumulation <= 0:
        raise ValueError("epochs, batch_size and accumulation must be positive")
    run_seed = int(seed if seed is not None else train_cfg.get("seed", 20260723))
    np_rng = np.random.default_rng(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    model, target = build_model(arrays["image_features"].shape[1], config, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    amp = bool(train_cfg.get("amp", True)) and target.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    model_cfg = config.get("model", {})
    modality_probability = float(model_cfg.get("modality_dropout", 0.15))
    hard_top_k = int(model_cfg.get("hard_negative_top_k", 8))
    hard_weight = float(model_cfg.get("hard_negative_weight", 0.10))
    teacher_weight = float(model_cfg.get("teacher_preservation_weight", 0.10))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Dev selection uses both retrieval directions at R@1 and R@10. Selecting
    # only text->image R@10 was found to hide a large image->text regression.
    dev_caps = _caption_indices_for_images(arrays["caption_image_index"], dev_indices)
    dev_local = {int(global_index): idx for idx, global_index in enumerate(dev_indices.tolist())}
    dev_caption_local = np.asarray(
        [dev_local[int(value)] for value in arrays["caption_image_index"][dev_caps]], dtype=np.int64
    )
    dev_t2i_relevance = build_caption_relevance(dev_caption_local, len(dev_indices))
    dev_i2t_relevance = build_image_relevance(dev_caption_local, len(dev_indices))
    baseline_t2i, _, _ = evaluate_direction(
        arrays["caption_features"][dev_caps],
        arrays["image_features"][dev_indices],
        dev_t2i_relevance,
        ks=(1, 10),
    )
    baseline_i2t, _, _ = evaluate_direction(
        arrays["image_features"][dev_indices],
        arrays["caption_features"][dev_caps],
        dev_i2t_relevance,
        ks=(1, 10),
    )
    baseline_score = float(
        np.mean(
            [
                baseline_t2i.recall_at["1"],
                baseline_t2i.recall_at["10"],
                baseline_i2t.recall_at["1"],
                baseline_i2t.recall_at["10"],
            ]
        )
    )
    best_score = baseline_score
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    raw_image = torch.from_numpy(arrays["image_features"].astype(np.float32, copy=False))
    raw_text = torch.from_numpy(arrays["caption_features"].astype(np.float32, copy=False))
    # Epoch 0 is an exact identity dual encoder and is a valid fallback if no
    # trained checkpoint passes the dev quality gates.
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
        str(output_dir / "adapter.safetensors"),
    )
    (output_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "schema_version": "mosaic.adapter.v1",
                "input_dim": int(arrays["image_features"].shape[1]),
                "model": config.get("model", {}),
                "manifest_sha256": manifest["manifest_sha256"],
                "feature_metadata_sha256": metadata.get("metadata_sha256"),
                "epoch": 0,
                "dev_joint_selection_score": baseline_score,
                "fallback_reason": "identity_initialization_if_no_dev_candidate_passes",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for epoch in range(epochs):
        model.train()
        image_rows, caption_rows = _sample_training_pairs(manifest, arrays, train_indices, np_rng)
        order = np_rng.permutation(image_rows.size)
        total_loss = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_start in range(0, order.size, batch_size):
            batch_order = order[batch_start : batch_start + batch_size]
            image_batch = raw_image[image_rows[batch_order]].to(target)
            text_batch = raw_text[caption_rows[batch_order]].to(target)
            if image_batch.shape[0] < 2:
                continue
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                image_proj = model.encode_image(image_batch)
                text_proj = model.encode_text(text_batch)
                dropped = modality_dropout(image_proj, text_proj, modality_probability)
                full_item = model.gate(image_proj, text_proj, torch.ones_like(dropped.mask))
                full_item = torch.nn.functional.normalize(
                    full_item[:, 0:1] * image_proj + full_item[:, 1:2] * text_proj, dim=-1
                )
                drop_item = model.gate(dropped.image, dropped.text, dropped.mask)
                drop_item = torch.nn.functional.normalize(
                    drop_item[:, 0:1] * dropped.image + drop_item[:, 1:2] * dropped.text,
                    dim=-1,
                )
                loss = (
                    symmetric_contrastive_loss(image_proj, text_proj, model.temperature)
                    + 0.5 * symmetric_contrastive_loss(full_item, text_proj, model.temperature)
                    + 0.5 * symmetric_contrastive_loss(drop_item, text_proj, model.temperature)
                    + hard_weight * hard_negative_margin_loss(image_proj, text_proj, top_k=hard_top_k)
                    + teacher_weight
                    * (
                        (1.0 - torch.nn.functional.cosine_similarity(image_proj, image_batch, dim=-1)).mean()
                        + (1.0 - torch.nn.functional.cosine_similarity(text_proj, text_batch, dim=-1)).mean()
                    )
                ) / float(accumulation)
            scaler.scale(loss).backward()
            if (steps + 1) % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu()) * float(accumulation)
            steps += 1
        if steps and steps % accumulation:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        encoded = _encode_model(model, arrays, device=target)
        dev_t2i, _, _ = evaluate_direction(
            encoded["text"][dev_caps],
            encoded["image"][dev_indices],
            dev_t2i_relevance,
            ks=(1, 10),
        )
        dev_i2t, _, _ = evaluate_direction(
            encoded["image"][dev_indices],
            encoded["text"][dev_caps],
            dev_i2t_relevance,
            ks=(1, 10),
        )
        score = float(
            np.mean(
                [
                    dev_t2i.recall_at["1"],
                    dev_t2i.recall_at["10"],
                    dev_i2t.recall_at["1"],
                    dev_i2t.recall_at["10"],
                ]
            )
        )
        passes_gate = bool(
            dev_t2i.recall_at["10"] >= baseline_t2i.recall_at["10"] - 0.002
            and dev_i2t.recall_at["10"] >= baseline_i2t.recall_at["10"] - 0.002
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(1, steps),
                "dev_text_to_image_recall@1": dev_t2i.recall_at["1"],
                "dev_text_to_image_recall@10": dev_t2i.recall_at["10"],
                "dev_image_to_text_recall@1": dev_i2t.recall_at["1"],
                "dev_image_to_text_recall@10": dev_i2t.recall_at["10"],
                "dev_joint_selection_score": score,
                "passes_directional_gate": passes_gate,
            }
        )
        if passes_gate and score > best_score:
            best_score = score
            best_epoch = epoch + 1
            tensors = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
            save_file(tensors, str(output_dir / "adapter.safetensors"))
            (output_dir / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "schema_version": "mosaic.adapter.v1",
                        "input_dim": int(arrays["image_features"].shape[1]),
                        "model": config.get("model", {}),
                        "manifest_sha256": manifest["manifest_sha256"],
                        "feature_metadata_sha256": metadata.get("metadata_sha256"),
                        "epoch": best_epoch,
                        "dev_joint_selection_score": best_score,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    elapsed = time.perf_counter() - started
    # Resolve the repository root from the manifest location rather than from
    # the current working directory; this keeps provenance correct when the
    # script is launched through a wrapper.
    manifest_root = Path(manifest_path).resolve()
    for candidate in (manifest_root.parent, *manifest_root.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").is_file():
            manifest_root = candidate
            break
    result = {
        "schema_version": "mosaic.training.v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "feature_metadata_sha256": metadata.get("metadata_sha256"),
        "train_images": int(train_indices.size),
        "dev_images": int(dev_indices.size),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "selection": {
            "metric": "mean(text_to_image R@1/R@10, image_to_text R@1/R@10)",
            "baseline_score": baseline_score,
            "best_score": best_score,
            "baseline_text_to_image": baseline_t2i.to_dict(),
            "baseline_image_to_text": baseline_i2t.to_dict(),
            "directional_r10_max_drop": 0.002,
        },
        "trainable_parameters": trainable_parameter_count(model),
        "device": str(target),
        "elapsed_seconds": elapsed,
        "history": history,
        "seed": run_seed,
        "input_commit": _git_commit(manifest_root),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def load_trained_model(
    checkpoint_dir: Path,
    input_dim: int,
    config: dict[str, Any],
    *,
    device: str = "auto",
) -> tuple[MosaicDualEncoder, torch.device]:
    checkpoint_dir = Path(checkpoint_dir)
    model, target = build_model(input_dim, config, device=device)
    state = load_file(str(checkpoint_dir / "adapter.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, target


def evaluate_model(
    manifest_path: Path,
    feature_path: Path,
    config: dict[str, Any],
    *,
    checkpoint_dir: Path | None = None,
    device: str = "auto",
    split: str = "test",
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    metadata, arrays = load_feature_bundle(feature_path)
    if checkpoint_dir is None:
        # Zero-shot baseline uses raw, normalized CLIP projections.
        encoded = {"image": arrays["image_features"], "text": arrays["caption_features"]}
        encoded["metadata_by_slot"] = {}
        encoded["full_by_slot"] = {}
        for slot in sorted({int(value) for value in arrays["caption_index"].tolist()}):
            metadata_raw = _mean_caption_features(
                arrays["caption_features"],
                arrays["caption_image_index"],
                arrays["image_features"].shape[0],
                excluded_caption_index=slot,
                caption_index=arrays["caption_index"],
            )
            full = arrays["image_features"] * 0.5 + metadata_raw * 0.5
            full /= np.linalg.norm(full, axis=1, keepdims=True).clip(min=1e-8)
            encoded["metadata_by_slot"][slot] = metadata_raw.astype(np.float32)
            encoded["full_by_slot"][slot] = full.astype(np.float32)
        model_name = metadata.get("model_name", "unknown")
    else:
        model, target = load_trained_model(
            checkpoint_dir,
            arrays["image_features"].shape[1],
            config,
            device=device,
        )
        encoded = _encode_model(model, arrays, device=target)
        # Build projected leave-one-caption-out metadata and fused targets for
        # all observed caption slots. This is computed before reading Test labels.
        slots = sorted({int(value) for value in arrays["caption_index"].tolist()})
        encoded["metadata_by_slot"] = {}
        encoded["full_by_slot"] = {}
        with torch.inference_mode():
            for slot in slots:
                metadata_raw = _mean_caption_features(
                    arrays["caption_features"],
                    arrays["caption_image_index"],
                    arrays["image_features"].shape[0],
                    excluded_caption_index=slot,
                    caption_index=arrays["caption_index"],
                )
                image_tensor = torch.from_numpy(arrays["image_features"].astype(np.float32)).to(target)
                metadata_tensor = torch.from_numpy(metadata_raw).to(target)
                image_proj = model.encode_image(image_tensor)
                metadata_proj = model.encode_text(metadata_tensor)
                mask = torch.ones((image_proj.shape[0], 2), device=target)
                weights = model.gate(image_proj, metadata_proj, mask)
                full = torch.nn.functional.normalize(
                    weights[:, 0:1] * image_proj + weights[:, 1:2] * metadata_proj, dim=-1
                )
                encoded["metadata_by_slot"][slot] = metadata_proj.cpu().numpy().astype(np.float32)
                encoded["full_by_slot"][slot] = full.cpu().numpy().astype(np.float32)
        model_name = "mosaic-trained-adapter"
    image_indices = _split_indices(manifest, split)
    if image_indices.size == 0:
        raise ValueError(f"split {split} is empty")
    eval_result = evaluate_caption_folds(
        encoded,
        arrays,
        image_indices,
        ks=config.get("evaluation", {}).get("ks", (1, 5, 10, 50)),
    )
    # Add cluster bootstrap intervals for the main text→image R@10 and MRR.
    reps = int(
        bootstrap_replicates
        if bootstrap_replicates is not None
        else config.get("evaluation", {}).get("bootstrap_replicates", 1000)
    )
    seed = int(config.get("evaluation", {}).get("bootstrap_seed", 20260724))
    for mode in ("image_only", "text_only", "full"):
        ranks = eval_result["_ranks"][mode]
        clusters = eval_result["_clusters"][mode]
        vectors = metric_vectors_from_ranks(ranks, (10,))
        eval_result["text_to_image"][mode]["bootstrap"] = {
            key: bootstrap_ci(value, clusters, replicates=reps, seed=seed + idx)
            for idx, (key, value) in enumerate(vectors.items())
        }
    eval_result["model_name"] = model_name
    eval_result["split"] = split
    eval_result["feature_metadata_sha256"] = metadata.get("metadata_sha256")
    # Internal arrays are useful for paired comparisons but are removed before
    # JSON serialization by callers.
    return eval_result


def strip_internal(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("_")}


__all__ = [
    "build_model",
    "evaluate_caption_folds",
    "evaluate_model",
    "load_trained_model",
    "strip_internal",
    "train_adapter",
]
