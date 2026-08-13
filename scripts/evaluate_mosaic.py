from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mosaic.experiment import evaluate_model, strip_internal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--config", type=Path, default=Path("configs/coco5k_v1.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = evaluate_model(
        args.manifest,
        args.features,
        config,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    clean = strip_internal(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "mosaic_coco5k_v1" if args.checkpoint_dir else "mosaic_coco5k_zero_shot_v1"
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(clean, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

