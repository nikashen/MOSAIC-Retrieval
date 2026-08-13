from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at: dict[str, float]
    mrr: float
    median_rank: float
    mean_rank: float
    queries: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_embeddings(query: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if query.ndim != 2 or targets.ndim != 2 or query.shape[1] != targets.shape[1]:
        raise ValueError("query and targets must be [n, d] and [m, d]")
    if not np.isfinite(query).all() or not np.isfinite(targets).all():
        raise ValueError("embeddings must be finite")
    if query.shape[0] == 0 or targets.shape[0] == 0:
        raise ValueError("retrieval inputs must be non-empty")
    return query, targets


def _ranks_for_relevance(
    query: np.ndarray,
    targets: np.ndarray,
    relevant: Sequence[np.ndarray],
    *,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    query, targets = _validate_embeddings(query, targets)
    if len(relevant) != query.shape[0]:
        raise ValueError("relevance list length must equal query count")
    ranks: list[int] = []
    ndcgs: list[float] = []
    for start in range(0, query.shape[0], max(1, int(chunk_size))):
        stop = min(query.shape[0], start + max(1, int(chunk_size)))
        scores = query[start:stop] @ targets.T
        order = np.argsort(-scores, axis=1, kind="stable")
        for local, indices in enumerate(order):
            rel = np.asarray(relevant[start + local], dtype=np.int64).reshape(-1)
            if rel.size == 0 or np.any(rel < 0) or np.any(rel >= targets.shape[0]):
                raise ValueError("relevance indices are invalid")
            rel = np.unique(rel)
            positions = np.flatnonzero(np.isin(indices, rel))
            if positions.size == 0:
                # Every finite full-catalog ranking contains a relevant item.
                raise AssertionError("relevant item was absent from full ranking")
            best = int(positions[0]) + 1
            ranks.append(best)
            top_rel = np.isin(indices[: min(len(indices), 1000)], rel).astype(np.float64)
            discounts = 1.0 / np.log2(np.arange(2, top_rel.size + 2))
            dcg = float((top_rel * discounts).sum())
            ideal_count = min(rel.size, top_rel.size)
            ideal = float((1.0 / np.log2(np.arange(2, ideal_count + 2))).sum())
            ndcgs.append(dcg / ideal if ideal else 0.0)
    return np.asarray(ranks, dtype=np.float64), np.asarray(ndcgs, dtype=np.float64)


def aggregate_ranks(ranks: np.ndarray, ks: Iterable[int]) -> RetrievalMetrics:
    ranks = np.asarray(ranks, dtype=np.float64).reshape(-1)
    if ranks.size == 0 or not np.isfinite(ranks).all() or np.any(ranks < 1):
        raise ValueError("ranks must be finite positive values")
    ks_tuple = tuple(sorted({int(k) for k in ks}))
    if not ks_tuple or ks_tuple[0] <= 0:
        raise ValueError("ks must contain positive integers")
    return RetrievalMetrics(
        recall_at={str(k): float(np.mean(ranks <= k)) for k in ks_tuple},
        mrr=float(np.mean(1.0 / ranks)),
        median_rank=float(np.median(ranks)),
        mean_rank=float(np.mean(ranks)),
        queries=int(ranks.size),
    )


def evaluate_direction(
    query_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    relevant: Sequence[Sequence[int] | np.ndarray],
    *,
    ks: Iterable[int] = (1, 5, 10, 50),
    chunk_size: int = 256,
) -> tuple[RetrievalMetrics, np.ndarray, np.ndarray]:
    ranks, ndcgs = _ranks_for_relevance(
        query_embeddings,
        target_embeddings,
        [np.asarray(row, dtype=np.int64) for row in relevant],
        chunk_size=chunk_size,
    )
    metrics = aggregate_ranks(ranks, ks)
    # NDCG is kept as an explicit additional metric rather than silently mixing
    # it into Recall; callers can report the mean and bootstrap it separately.
    return metrics, ranks, ndcgs


def build_caption_relevance(
    caption_image_indices: Sequence[int], image_count: int
) -> list[np.ndarray]:
    indices = np.asarray(caption_image_indices, dtype=np.int64).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= int(image_count)):
        raise ValueError("caption image index out of range")
    return [np.asarray([int(index)], dtype=np.int64) for index in indices]


def build_image_relevance(
    caption_image_indices: Sequence[int], image_count: int
) -> list[np.ndarray]:
    indices = np.asarray(caption_image_indices, dtype=np.int64).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= int(image_count)):
        raise ValueError("caption image index out of range")
    return [np.flatnonzero(indices == image_index).astype(np.int64) for image_index in range(image_count)]


def cluster_metric_values(values: np.ndarray, clusters: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    clusters = np.asarray(clusters, dtype=np.int64).reshape(-1)
    if values.shape != clusters.shape or values.size == 0:
        raise ValueError("values and clusters must be aligned and non-empty")
    unique = np.unique(clusters)
    means = np.asarray([values[clusters == cluster].mean() for cluster in unique], dtype=np.float64)
    return unique, means


def bootstrap_ci(
    values: np.ndarray,
    clusters: Sequence[int],
    *,
    replicates: int = 1000,
    seed: int = 20260724,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Cluster bootstrap for a single per-query metric."""

    _, cluster_means = cluster_metric_values(values, clusters)
    if int(replicates) < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must be in (0, 1)")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, cluster_means.size, size=(int(replicates), cluster_means.size))
    estimates = cluster_means[draws].mean(axis=1)
    lower, upper = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": float(cluster_means.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "alpha": float(alpha),
        "replicates": int(replicates),
        "clusters": int(cluster_means.size),
        "seed": int(seed),
    }


def paired_bootstrap_delta_ci(
    left_values: np.ndarray,
    right_values: np.ndarray,
    clusters: Sequence[int],
    *,
    replicates: int = 1000,
    seed: int = 20260724,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    left_values = np.asarray(left_values, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right_values, dtype=np.float64).reshape(-1)
    clusters = np.asarray(clusters, dtype=np.int64).reshape(-1)
    if left_values.shape != right_values.shape or left_values.shape != clusters.shape:
        raise ValueError("paired values and clusters must be aligned")
    unique = np.unique(clusters)
    left_cluster = np.asarray([left_values[clusters == c].mean() for c in unique])
    right_cluster = np.asarray([right_values[clusters == c].mean() for c in unique])
    delta = left_cluster - right_cluster
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, delta.size, size=(int(replicates), delta.size))
    estimates = delta[draws].mean(axis=1)
    lower, upper = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return {
        "delta": float(delta.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "alpha": float(alpha),
        "replicates": int(replicates),
        "clusters": int(unique.size),
        "seed": int(seed),
    }


def metric_vectors_from_ranks(ranks: np.ndarray, ks: Iterable[int]) -> dict[str, np.ndarray]:
    ranks = np.asarray(ranks, dtype=np.float64).reshape(-1)
    values: dict[str, np.ndarray] = {f"recall@{int(k)}": (ranks <= int(k)).astype(np.float64) for k in ks}
    values["mrr"] = 1.0 / ranks
    return values


def deterministic_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes(order="C")).hexdigest()


__all__ = [
    "RetrievalMetrics",
    "aggregate_ranks",
    "bootstrap_ci",
    "build_caption_relevance",
    "build_image_relevance",
    "cluster_metric_values",
    "deterministic_digest",
    "evaluate_direction",
    "metric_vectors_from_ranks",
    "paired_bootstrap_delta_ci",
]

