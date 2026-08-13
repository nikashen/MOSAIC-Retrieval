from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .data import load_manifest
from .experiment import _caption_indices_for_images, _encode_model, _split_indices, load_trained_model
from .features import load_feature_bundle, resolve_device
from .metrics import aggregate_ranks, metric_vectors_from_ranks, paired_bootstrap_delta_ci
from .models import CrossEncoderReranker, trainable_parameter_count


def mine_hard_negatives(
    query: np.ndarray,
    items: np.ndarray,
    positive_indices: np.ndarray,
    *,
    top_k: int = 5,
    chunk_size: int = 512,
) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    items = np.asarray(items, dtype=np.float32)
    positive_indices = np.asarray(positive_indices, dtype=np.int64).reshape(-1)
    if query.ndim != 2 or items.ndim != 2 or query.shape[1] != items.shape[1]:
        raise ValueError("query/item matrices are incompatible")
    if positive_indices.shape != (query.shape[0],):
        raise ValueError("positive indices must align with queries")
    k = min(max(1, int(top_k)), items.shape[0] - 1)
    output = np.empty((query.shape[0], k), dtype=np.int64)
    for start in range(0, query.shape[0], max(1, int(chunk_size))):
        stop = min(query.shape[0], start + max(1, int(chunk_size)))
        scores = query[start:stop] @ items.T
        scores[np.arange(stop - start), positive_indices[start:stop]] = -np.inf
        candidates = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(-candidate_scores, axis=1, kind="stable")
        output[start:stop] = np.take_along_axis(candidates, order, axis=1)
    return output


@torch.inference_mode()
def candidate_scores(
    model: CrossEncoderReranker,
    query: np.ndarray,
    items: np.ndarray,
    *,
    candidate_k: int = 50,
    batch_size: int = 4096,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.float32)
    items = np.asarray(items, dtype=np.float32)
    base = query @ items.T
    candidate_k = min(max(1, int(candidate_k)), items.shape[0])
    order = np.argsort(-base, axis=1, kind="stable")
    candidates = order[:, :candidate_k]
    base_candidate = np.take_along_axis(base, candidates, axis=1)
    flat_query = np.repeat(query, candidate_k, axis=0)
    flat_items = items[candidates.reshape(-1)]
    outputs: list[np.ndarray] = []
    model.eval()
    for start in range(0, flat_query.shape[0], max(1, int(batch_size))):
        q = torch.from_numpy(flat_query[start : start + batch_size]).to(device)
        i = torch.from_numpy(flat_items[start : start + batch_size]).to(device)
        outputs.append(model(q, i).float().cpu().numpy())
    interaction = np.concatenate(outputs).reshape(query.shape[0], candidate_k)
    return order, candidates, base_candidate, interaction


def reranked_ranks(
    full_base_order: np.ndarray,
    candidates: np.ndarray,
    base_scores: np.ndarray,
    interaction_scores: np.ndarray,
    positive_indices: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    positive_indices = np.asarray(positive_indices, dtype=np.int64).reshape(-1)
    if full_base_order.shape[0] != positive_indices.size:
        raise ValueError("positive indices do not align with candidate rows")
    combined = base_scores + float(alpha) * interaction_scores
    local_order = np.argsort(-combined, axis=1, kind="stable")
    reranked = np.take_along_axis(candidates, local_order, axis=1)
    ranks = np.empty(positive_indices.size, dtype=np.float64)
    for row, positive in enumerate(positive_indices.tolist()):
        inside = np.flatnonzero(reranked[row] == positive)
        if inside.size:
            ranks[row] = float(inside[0] + 1)
        else:
            ranks[row] = float(np.flatnonzero(full_base_order[row] == positive)[0] + 1)
    return ranks


def select_alpha(
    full_base_order: np.ndarray,
    candidates: np.ndarray,
    base_scores: np.ndarray,
    interaction_scores: np.ndarray,
    positive_indices: np.ndarray,
    alphas: Iterable[float] = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5),
) -> tuple[float, float, dict[str, Any]]:
    best_alpha = 0.0
    best_score = -float("inf")
    evidence: dict[str, Any] = {}
    for alpha in alphas:
        ranks = reranked_ranks(
            full_base_order,
            candidates,
            base_scores,
            interaction_scores,
            positive_indices,
            alpha=float(alpha),
        )
        metrics = aggregate_ranks(ranks, (1, 10))
        score = float((metrics.recall_at["1"] + metrics.recall_at["10"]) / 2)
        evidence[str(float(alpha))] = {"score": score, "metrics": metrics.to_dict()}
        if score > best_score + 1e-12:
            best_alpha, best_score = float(alpha), score
    return best_alpha, best_score, evidence


def train_reranker(
    manifest_path: Path,
    feature_path: Path,
    adapter_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    *,
    device: str = "auto",
    epochs: int = 4,
    negative_k: int = 5,
    candidate_k: int = 50,
    seed: int = 20260725,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    _, arrays = load_feature_bundle(feature_path)
    adapter, adapter_device = load_trained_model(
        adapter_dir,
        int(arrays["image_features"].shape[1]),
        config,
        device=device,
    )
    encoded = _encode_model(adapter, arrays, device=adapter_device)
    train_images = _split_indices(manifest, "train")
    dev_images = _split_indices(manifest, "dev")

    def split_arrays(image_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        captions = _caption_indices_for_images(arrays["caption_image_index"], image_indices)
        local = {int(global_id): index for index, global_id in enumerate(image_indices.tolist())}
        positives = np.asarray(
            [local[int(value)] for value in arrays["caption_image_index"][captions]], dtype=np.int64
        )
        return encoded["text"][captions], encoded["image"][image_indices], positives

    train_query, train_items, train_positive = split_arrays(train_images)
    dev_query, dev_items, dev_positive = split_arrays(dev_images)
    negatives = mine_hard_negatives(train_query, train_items, train_positive, top_k=negative_k)
    target = resolve_device(device)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    model = CrossEncoderReranker(
        int(train_query.shape[1]),
        hidden_dim=int(config.get("model", {}).get("reranker_hidden_dim", 256)),
        dropout=float(config.get("model", {}).get("dropout", 0.1)),
    ).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    rng = np.random.default_rng(int(seed))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = -1
    selected_alpha = 0.0
    started = time.perf_counter()
    pair_query = np.repeat(np.arange(train_query.shape[0], dtype=np.int64), negative_k)
    pair_negative = negatives.reshape(-1)
    pair_positive = np.repeat(train_positive, negative_k)
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(pair_query.size)
        losses: list[float] = []
        for start in range(0, order.size, 512):
            batch = order[start : start + 512]
            q = torch.from_numpy(train_query[pair_query[batch]]).to(target)
            positive = torch.from_numpy(train_items[pair_positive[batch]]).to(target)
            negative = torch.from_numpy(train_items[pair_negative[batch]]).to(target)
            positive_score = model(q, positive)
            negative_score = model(q, negative)
            loss = torch.nn.functional.softplus(-(positive_score - negative_score)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        full_order, candidates, base_score, interaction = candidate_scores(
            model, dev_query, dev_items, candidate_k=candidate_k, device=target
        )
        alpha, score, alpha_evidence = select_alpha(
            full_order, candidates, base_score, interaction, dev_positive
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_pairwise_loss": float(np.mean(losses)),
                "dev_selection_score": score,
                "selected_alpha": alpha,
                "alpha_grid": alpha_evidence,
            }
        )
        if score > best_score:
            best_score, best_epoch, selected_alpha = score, epoch + 1, alpha
            save_file(
                {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
                str(output_dir / "reranker.safetensors"),
            )
    summary = {
        "schema_version": "mosaic.reranker.v1",
        "train_queries": int(train_query.shape[0]),
        "dev_queries": int(dev_query.shape[0]),
        "hard_negatives_per_query": int(negative_k),
        "candidate_k": int(candidate_k),
        "best_epoch": int(best_epoch),
        "selected_alpha": float(selected_alpha),
        "best_dev_score": float(best_score),
        "trainable_parameters": trainable_parameter_count(model),
        "seed": int(seed),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }
    (output_dir / "reranker_config.json").write_text(
        json.dumps(
            {
                "schema_version": "mosaic.reranker.config.v1",
                "embedding_dim": int(train_query.shape[1]),
                "hidden_dim": int(config.get("model", {}).get("reranker_hidden_dim", 256)),
                "dropout": float(config.get("model", {}).get("dropout", 0.1)),
                "candidate_k": int(candidate_k),
                "selected_alpha": float(selected_alpha),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "reranker_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def load_reranker(path: Path, *, device: str = "cpu") -> tuple[CrossEncoderReranker, dict[str, Any], torch.device]:
    path = Path(path)
    config = json.loads((path / "reranker_config.json").read_text(encoding="utf-8"))
    target = resolve_device(device)
    model = CrossEncoderReranker(
        int(config["embedding_dim"]), int(config["hidden_dim"]), float(config["dropout"])
    ).to(target)
    model.load_state_dict(load_file(str(path / "reranker.safetensors"), device="cpu"), strict=True)
    model.eval()
    return model, config, target


def evaluate_reranker(
    manifest_path: Path,
    feature_path: Path,
    adapter_dir: Path,
    reranker_dir: Path,
    config: dict[str, Any],
    *,
    split: str,
    device: str = "cpu",
    bootstrap_replicates: int = 1000,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    metadata, arrays = load_feature_bundle(feature_path)
    adapter, adapter_device = load_trained_model(
        adapter_dir,
        int(arrays["image_features"].shape[1]),
        config,
        device=device,
    )
    encoded = _encode_model(adapter, arrays, device=adapter_device)
    image_indices = _split_indices(manifest, split)
    captions = _caption_indices_for_images(arrays["caption_image_index"], image_indices)
    local = {int(global_id): index for index, global_id in enumerate(image_indices.tolist())}
    positives = np.asarray(
        [local[int(value)] for value in arrays["caption_image_index"][captions]], dtype=np.int64
    )
    query, items = encoded["text"][captions], encoded["image"][image_indices]
    model, reranker_config, target = load_reranker(reranker_dir, device=device)
    full_order, candidates, base_score, interaction = candidate_scores(
        model,
        query,
        items,
        candidate_k=int(reranker_config["candidate_k"]),
        device=target,
    )
    baseline_ranks = reranked_ranks(
        full_order, candidates, base_score, interaction, positives, alpha=0.0
    )
    reranker_ranks = reranked_ranks(
        full_order,
        candidates,
        base_score,
        interaction,
        positives,
        alpha=float(reranker_config["selected_alpha"]),
    )
    ks = config.get("evaluation", {}).get("ks", (1, 5, 10, 50))
    baseline_metrics = aggregate_ranks(baseline_ranks, ks)
    reranker_metrics = aggregate_ranks(reranker_ranks, ks)
    seed = int(config.get("evaluation", {}).get("bootstrap_seed", 20260724)) + 100
    paired: dict[str, Any] = {}
    left = metric_vectors_from_ranks(reranker_ranks, (1, 10))
    right = metric_vectors_from_ranks(baseline_ranks, (1, 10))
    for index, key in enumerate(("recall@1", "recall@10", "mrr")):
        paired[key] = paired_bootstrap_delta_ci(
            left[key],
            right[key],
            positives,
            replicates=int(bootstrap_replicates),
            seed=seed + index,
        )
    return {
        "schema_version": "mosaic.reranker.evaluation.v1",
        "split": split,
        "queries": int(query.shape[0]),
        "items": int(items.shape[0]),
        "candidate_k": int(reranker_config["candidate_k"]),
        "selected_alpha": float(reranker_config["selected_alpha"]),
        "baseline": baseline_metrics.to_dict(),
        "reranked": reranker_metrics.to_dict(),
        "paired_cluster_bootstrap": paired,
        "feature_metadata_sha256": metadata.get("metadata_sha256"),
    }


__all__ = [
    "candidate_scores",
    "evaluate_reranker",
    "load_reranker",
    "mine_hard_negatives",
    "reranked_ranks",
    "select_alpha",
    "train_reranker",
]
