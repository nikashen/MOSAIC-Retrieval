from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic.reranking import train_reranker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/coco5k_v1.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = train_reranker(
        args.manifest,
        args.features,
        args.adapter_dir,
        args.output_dir,
        config,
        device=args.device,
        epochs=args.epochs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

