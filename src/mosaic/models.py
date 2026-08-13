from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _check_matrix(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")
    return value


class ProjectionHead(nn.Module):
    """Small trainable adapter on top of a frozen CLIP embedding."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if min(int(input_dim), int(output_dim), int(hidden_dim)) <= 0:
            raise ValueError("projection dimensions must be positive")
        if not 0 <= float(dropout) < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.residual = self.input_dim == self.output_dim
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.output_dim),
        )
        if self.residual:
            # Start near the frozen CLIP geometry. A random replacement head can
            # easily destroy a strong pretrained space on a 3.4k-image train set.
            self.raw_residual_scale = nn.Parameter(torch.tensor(-4.5951198501))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            self.register_parameter("raw_residual_scale", None)

    def forward(self, features: Tensor) -> Tensor:
        _check_matrix(features, "features")
        if features.shape[-1] != self.input_dim:
            raise ValueError("projection input width mismatch")
        adapted = self.net(features)
        if self.residual:
            adapted = features + torch.sigmoid(self.raw_residual_scale) * adapted
        return F.normalize(adapted, dim=-1)


class ModalityGate(nn.Module):
    """Reliability gate that is explicitly conditioned on missing-modality mask."""

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim * 2 + 2, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )
        # Equal fusion is the frozen starting policy; learning must justify any
        # deviation on Dev rather than inheriting a random modality preference.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, image: Tensor, text: Tensor, mask: Tensor) -> Tensor:
        _check_matrix(image, "image")
        _check_matrix(text, "text")
        _check_matrix(mask, "mask")
        if image.shape != text.shape or image.shape[0] != mask.shape[0] or mask.shape[1] != 2:
            raise ValueError("gate inputs have incompatible shapes")
        if torch.any(mask.sum(dim=1) <= 0):
            raise ValueError("at least one modality must be present per row")
        mask = mask.to(dtype=image.dtype, device=image.device).clamp(0, 1)
        logits = self.mlp(torch.cat((image, text, mask), dim=-1))
        logits = logits.masked_fill(mask <= 0, -1e4)
        return torch.softmax(logits, dim=-1)


@dataclass(frozen=True)
class ModalityBatch:
    image: Tensor
    text: Tensor
    mask: Tensor


def modality_dropout(
    image: Tensor,
    text: Tensor,
    probability: float,
    *,
    generator: torch.Generator | None = None,
) -> ModalityBatch:
    """Randomly hide one modality while never producing an all-missing row."""

    _check_matrix(image, "image")
    _check_matrix(text, "text")
    if image.shape != text.shape:
        raise ValueError("image and text features must have the same shape")
    probability = float(probability)
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    batch = image.shape[0]
    mask = torch.ones((batch, 2), dtype=image.dtype, device=image.device)
    if probability:
        random = torch.rand((batch,), generator=generator, device=image.device)
        hide = random < probability
        choose_image = torch.rand((batch,), generator=generator, device=image.device) < 0.5
        mask[hide & choose_image, 0] = 0
        mask[hide & ~choose_image, 1] = 0
    return ModalityBatch(image * mask[:, 0:1], text * mask[:, 1:2], mask)


class MosaicDualEncoder(nn.Module):
    """Trainable projection/fusion heads over frozen CLIP features."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        temperature_init: float = 0.07,
    ) -> None:
        super().__init__()
        if float(temperature_init) <= 0:
            raise ValueError("temperature_init must be positive")
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.image_projection = ProjectionHead(input_dim, embedding_dim, hidden_dim, dropout)
        self.text_projection = ProjectionHead(input_dim, embedding_dim, hidden_dim, dropout)
        self.gate = ModalityGate(embedding_dim, hidden_dim // 2 or 1)
        # log(exp(t)-1) keeps temperature positive while allowing optimization.
        self.raw_temperature = nn.Parameter(
            torch.tensor(math.log(math.expm1(float(temperature_init))))
        )

    @property
    def temperature(self) -> Tensor:
        return F.softplus(self.raw_temperature).clamp(0.005, 1.0)

    def encode_image(self, image_features: Tensor) -> Tensor:
        return self.image_projection(image_features)

    def encode_text(self, text_features: Tensor) -> Tensor:
        return self.text_projection(text_features)

    def encode_item(
        self,
        image_features: Tensor,
        text_features: Tensor,
        modality_mask: Tensor | None = None,
    ) -> Tensor:
        image = self.encode_image(image_features)
        text = self.encode_text(text_features)
        if modality_mask is None:
            modality_mask = torch.ones(
                (image.shape[0], 2), dtype=image.dtype, device=image.device
            )
        weights = self.gate(image, text, modality_mask)
        return F.normalize(weights[:, 0:1] * image + weights[:, 1:2] * text, dim=-1)

    def encode_query(self, text_features: Tensor) -> Tensor:
        return self.encode_text(text_features)

    def forward(self, image_features: Tensor, text_features: Tensor) -> dict[str, Tensor]:
        image = self.encode_image(image_features)
        text = self.encode_text(text_features)
        item = self.encode_item(image_features, text_features)
        return {"image": image, "text": text, "item": item, "temperature": self.temperature}


def symmetric_contrastive_loss(
    image_embeddings: Tensor,
    text_embeddings: Tensor,
    temperature: Tensor | float,
) -> Tensor:
    """CLIP-style symmetric InfoNCE with in-batch negatives."""

    image_embeddings = F.normalize(_check_matrix(image_embeddings, "image_embeddings"), dim=-1)
    text_embeddings = F.normalize(_check_matrix(text_embeddings, "text_embeddings"), dim=-1)
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError("image and text embedding matrices must have the same shape")
    if image_embeddings.shape[0] < 2:
        raise ValueError("contrastive loss requires at least two pairs")
    temp = torch.as_tensor(temperature, dtype=image_embeddings.dtype, device=image_embeddings.device)
    temp = temp.clamp(0.005, 1.0)
    logits = image_embeddings @ text_embeddings.transpose(0, 1) / temp
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels)) / 2


def hard_negative_margin_loss(
    image_embeddings: Tensor,
    text_embeddings: Tensor,
    *,
    top_k: int = 8,
    margin: float = 0.10,
) -> Tensor:
    """Margin loss against top-scoring non-matching pairs in the current batch."""

    image_embeddings = F.normalize(_check_matrix(image_embeddings, "image_embeddings"), dim=-1)
    text_embeddings = F.normalize(_check_matrix(text_embeddings, "text_embeddings"), dim=-1)
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError("image and text embedding matrices must have the same shape")
    n = image_embeddings.shape[0]
    if n < 2:
        return image_embeddings.sum() * 0
    similarity = image_embeddings @ text_embeddings.transpose(0, 1)
    diagonal = similarity.diagonal()
    mask = torch.eye(n, dtype=torch.bool, device=similarity.device)
    negatives = similarity.masked_fill(mask, -1e4)
    k = max(1, min(int(top_k), n - 1))
    hard = negatives.topk(k=k, dim=1).values
    image_to_text = F.relu(float(margin) + hard - diagonal[:, None]).mean()
    text_to_image = F.relu(
        float(margin)
        + negatives.transpose(0, 1).topk(k=k, dim=1).values
        - diagonal[:, None]
    ).mean()
    return (image_to_text + text_to_image) / 2


class CrossEncoderReranker(nn.Module):
    """Pair-interaction MLP used after dev freezes the candidate model.

    The historical class name is kept for artifact compatibility; this is not
    a token-level Transformer cross-encoder and reports must not call it one.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        width = int(embedding_dim) * 4
        self.network = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    @staticmethod
    def pair_features(query: Tensor, item: Tensor) -> Tensor:
        _check_matrix(query, "query")
        _check_matrix(item, "item")
        if query.shape != item.shape:
            raise ValueError("query and item shapes must match")
        return torch.cat((query, item, torch.abs(query - item), query * item), dim=-1)

    def forward(self, query: Tensor, item: Tensor) -> Tensor:
        return self.network(self.pair_features(query, item)).squeeze(-1)


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


__all__ = [
    "CrossEncoderReranker",
    "ModalityBatch",
    "ModalityGate",
    "MosaicDualEncoder",
    "ProjectionHead",
    "hard_negative_margin_loss",
    "modality_dropout",
    "symmetric_contrastive_loss",
    "trainable_parameter_count",
]
