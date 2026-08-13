from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic.reranking import evaluate_reranker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--reranker-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/coco5k_v1.json"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = evaluate_reranker(
        args.manifest,
        args.features,
        args.adapter_dir,
        args.reranker_dir,
        config,
        split=args.split,
        device=args.device,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

