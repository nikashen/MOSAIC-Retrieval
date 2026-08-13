from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mosaic.experiment import _mean_caption_features, load_trained_model
from mosaic.features import load_feature_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a safe exact/FAISS content index")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/coco5k_v1.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    metadata, arrays = load_feature_bundle(args.features)
    vectors = np.asarray(arrays["image_features"], dtype=np.float32)
    vector_scope = "zero_shot_image_only"
    if args.checkpoint_dir is not None:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        model, device = load_trained_model(
            args.checkpoint_dir,
            int(vectors.shape[1]),
            config,
            device=args.device,
        )
        with torch.inference_mode():
            image_tensor = torch.from_numpy(vectors).to(device)
            image_projected = model.encode_image(image_tensor).cpu().numpy().astype(np.float32)
            metadata_raw = _mean_caption_features(
                arrays["caption_features"],
                arrays["caption_image_index"],
                vectors.shape[0],
            )
            metadata_projected = model.encode_text(torch.from_numpy(metadata_raw).to(device)).cpu().numpy().astype(np.float32)
            mask = torch.ones((vectors.shape[0], 2), device=device)
            weights = model.gate(torch.from_numpy(image_projected).to(device), torch.from_numpy(metadata_projected).to(device), mask)
            vectors = torch.nn.functional.normalize(
                weights[:, 0:1] * torch.from_numpy(image_projected).to(device)
                + weights[:, 1:2] * torch.from_numpy(metadata_projected).to(device), dim=-1
            ).cpu().numpy().astype(np.float32)
        vector_scope = "mosaic_trained_full_all_metadata"
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "item_vectors.npz",
        content_id=arrays["image_ids"].astype(np.int64),
        content_vector=vectors,
        metadata_json=np.asarray(json.dumps({"feature_metadata_sha256": metadata.get("metadata_sha256"), "vector_scope": vector_scope, "safe_npz": "allow_pickle_false"})),
    )
    try:
        import faiss

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(args.output_dir / "content.index"))
        backend = "faiss.IndexFlatIP"
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        (args.output_dir / "faiss_unavailable.txt").write_text(type(exc).__name__ + ": " + str(exc), encoding="utf-8")
        backend = "numpy_exact_fallback"
    digest = hashlib.sha256(vectors.tobytes(order="C")).hexdigest()
    summary = {"backend": backend, "vector_scope": vector_scope, "items": int(vectors.shape[0]), "dimension": int(vectors.shape[1]), "vectors_sha256": digest}
    (args.output_dir / "index_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
