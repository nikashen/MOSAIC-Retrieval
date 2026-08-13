from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..models import ProjectionHead


_AGGREGATORS = frozenset({"mean", "max", "temporal_attention"})


class TemporalVideoEncoder(nn.Module):
    """Project pooled frame features and text features into one retrieval space.

    The encoder consumes already-extracted frame embeddings. Temporal attention
    is deliberately initialized as uniform pooling: its final scoring layer is
    zero, so the learned positional embeddings cannot perturb the epoch-0 CLIP
    mean-pooling baseline.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int | None = None,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        temperature_init: float = 0.07,
        aggregator: str = "mean",
        max_frames: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = self.input_dim if embedding_dim is None else int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_frames = int(max_frames)
        self.aggregator = str(aggregator)

        if min(self.input_dim, self.embedding_dim, self.hidden_dim, self.max_frames) <= 0:
            raise ValueError("model dimensions and max_frames must be positive")
        if self.aggregator not in _AGGREGATORS:
            allowed = ", ".join(sorted(_AGGREGATORS))
            raise ValueError(f"aggregator must be one of: {allowed}")
        if not 0 <= float(dropout) < 1:
            raise ValueError("dropout must be in [0, 1)")
        temperature_init = float(temperature_init)
        if not math.isfinite(temperature_init) or temperature_init <= 0:
            raise ValueError("temperature_init must be finite and positive")

        self.video_projection = ProjectionHead(
            self.input_dim, self.embedding_dim, self.hidden_dim, float(dropout)
        )
        self.text_projection = ProjectionHead(
            self.input_dim, self.embedding_dim, self.hidden_dim, float(dropout)
        )

        if self.aggregator == "temporal_attention":
            self.position_embedding = nn.Parameter(torch.empty(self.max_frames, self.input_dim))
            nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
            self.attention_scorer: nn.Sequential | None = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, 1),
            )
            nn.init.zeros_(self.attention_scorer[-1].weight)
            nn.init.zeros_(self.attention_scorer[-1].bias)
        else:
            self.register_parameter("position_embedding", None)
            self.attention_scorer = None

        # A stable inverse-softplus avoids overflow for unusually large, but
        # still valid, audit configurations.
        raw_temperature = (
            math.log(math.expm1(temperature_init))
            if temperature_init < 20.0
            else temperature_init
        )
        self.raw_temperature = nn.Parameter(torch.tensor(raw_temperature))

    @property
    def temperature(self) -> Tensor:
        # Match the bounded positive temperature contract used by the image
        # dual encoder, preventing a learned zero-temperature singularity.
        return F.softplus(self.raw_temperature).clamp(0.005, 1.0)

    def _validate_frames(
        self, frame_features: Tensor, frame_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(frame_features, Tensor) or frame_features.ndim != 3:
            raise ValueError("frame_features must be a rank-3 [B, T, D] tensor")
        if not frame_features.is_floating_point():
            raise ValueError("frame_features must use a floating-point dtype")
        batch, frames, width = frame_features.shape
        if batch <= 0 or frames <= 0:
            raise ValueError("frame_features must contain a non-empty batch and timeline")
        if width != self.input_dim:
            raise ValueError("frame feature width mismatch")
        if frames > self.max_frames:
            raise ValueError(
                f"frame timeline length {frames} exceeds max_frames={self.max_frames}"
            )
        if not bool(torch.isfinite(frame_features).all()):
            raise ValueError("frame_features must contain finite values")

        if frame_mask is None:
            valid = torch.ones((batch, frames), dtype=torch.bool, device=frame_features.device)
        else:
            if not isinstance(frame_mask, Tensor) or frame_mask.ndim != 2:
                raise ValueError("frame_mask must be a rank-2 [B, T] tensor")
            if tuple(frame_mask.shape) != (batch, frames):
                raise ValueError("frame_mask shape must match the frame batch and timeline")
            if frame_mask.is_complex():
                raise ValueError("frame_mask must be boolean or binary numeric values")
            if not bool(torch.isfinite(frame_mask).all()):
                raise ValueError("frame_mask must contain finite values")
            if bool(((frame_mask != 0) & (frame_mask != 1)).any()):
                raise ValueError("frame_mask must contain only zero/one values")
            valid = frame_mask.to(device=frame_features.device, dtype=torch.bool)
        if bool((~valid.any(dim=1)).any()):
            raise ValueError("every video must contain at least one unmasked frame")
        return frame_features, valid

    def _validate_text(self, text_features: Tensor) -> Tensor:
        if not isinstance(text_features, Tensor) or text_features.ndim != 2:
            raise ValueError("text_features must be a rank-2 [B, D] tensor")
        if not text_features.is_floating_point():
            raise ValueError("text_features must use a floating-point dtype")
        if text_features.shape[0] <= 0 or text_features.shape[1] != self.input_dim:
            raise ValueError("text_features have an incompatible shape")
        if not bool(torch.isfinite(text_features).all()):
            raise ValueError("text_features must contain finite values")
        return text_features

    @staticmethod
    def _uniform_weights(valid: Tensor, dtype: torch.dtype) -> Tensor:
        weights = valid.to(dtype=dtype)
        return weights / weights.sum(dim=1, keepdim=True)

    def attention_weights(
        self, frame_features: Tensor, frame_mask: Tensor | None = None
    ) -> Tensor:
        """Return masked temporal-attention probabilities with shape ``[B, T]``."""

        if self.aggregator != "temporal_attention":
            raise ValueError("attention_weights requires aggregator='temporal_attention'")
        frames, valid = self._validate_frames(frame_features, frame_mask)
        if self.position_embedding is None or self.attention_scorer is None:  # pragma: no cover
            raise RuntimeError("temporal attention modules are not initialized")
        positioned = frames + self.position_embedding[: frames.shape[1]].unsqueeze(0)
        logits = self.attention_scorer(positioned).squeeze(-1)
        logits = logits.masked_fill(~valid, -torch.inf)
        weights = torch.softmax(logits, dim=1)
        if not bool(torch.isfinite(weights).all()):  # defensive against corrupted parameters
            raise ValueError("temporal attention produced non-finite weights")
        return weights

    def _aggregate(self, frame_features: Tensor, frame_mask: Tensor | None) -> Tensor:
        frames, valid = self._validate_frames(frame_features, frame_mask)
        if self.aggregator == "mean":
            weights = self._uniform_weights(valid, frames.dtype)
            return (frames * weights.unsqueeze(-1)).sum(dim=1)
        if self.aggregator == "max":
            return frames.masked_fill(~valid.unsqueeze(-1), -torch.inf).max(dim=1).values
        weights = self.attention_weights(frames, valid)
        return (frames * weights.unsqueeze(-1)).sum(dim=1)

    def encode_video(
        self, frame_features: Tensor, frame_mask: Tensor | None = None
    ) -> Tensor:
        return self.video_projection(self._aggregate(frame_features, frame_mask))

    def encode_text(self, text_features: Tensor) -> Tensor:
        return self.text_projection(self._validate_text(text_features))

    def forward(
        self,
        frame_features: Tensor,
        text_features: Tensor,
        frame_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        video = self.encode_video(frame_features, frame_mask)
        text = self.encode_text(text_features)
        if video.shape[0] != text.shape[0]:
            raise ValueError("video and text batches must contain the same number of rows")
        return {"video": video, "text": text, "temperature": self.temperature}


__all__ = ["TemporalVideoEncoder"]
