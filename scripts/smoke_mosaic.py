from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from mosaic.experiment import build_model, evaluate_model, strip_internal
from mosaic.features import load_feature_bundle
from mosaic.models import hard_negative_margin_loss, modality_dropout, symmetric_contrastive_loss


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "data" / "processed" / "toy_manifest.json"
    features = root / "artifacts" / "toy_features.npz"
    config = json.loads((root / "configs" / "coco5k_v1.json").read_text(encoding="utf-8"))
    metadata, arrays = load_feature_bundle(features)
    model, device = build_model(arrays["image_features"].shape[1], config, device="cpu")
    image = torch.from_numpy(arrays["image_features"][:4])
    text = torch.from_numpy(arrays["caption_features"][:4])
    out = model(image, text)
    assert out["item"].shape == (4, int(config["model"]["embedding_dim"]))
    assert torch.isfinite(symmetric_contrastive_loss(out["image"], out["text"], out["temperature"]))
    assert torch.isfinite(hard_negative_margin_loss(out["image"], out["text"]))
    dropped = modality_dropout(out["image"], out["text"], 1.0)
    assert torch.all(dropped.mask.sum(dim=1) == 1)
    print(json.dumps({"status": "ok", "feature_metadata": metadata, "device": str(device)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
