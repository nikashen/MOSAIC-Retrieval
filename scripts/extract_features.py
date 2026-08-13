from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mosaic.features import build_feature_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()
    metadata = build_feature_bundle(
        args.manifest,
        args.output,
        model_name=args.model,
        device=args.device,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        image_batch_size=args.image_batch_size,
        text_batch_size=args.text_batch_size,
        max_images=args.max_images,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

