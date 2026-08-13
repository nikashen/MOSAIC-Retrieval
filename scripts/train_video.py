from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic.video.experiment import train_video_encoder


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Dev-selected MSR-VTT temporal encoder")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/msrvtt_1ka_v1.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = train_video_encoder(
        args.manifest,
        args.features,
        args.output_dir,
        config,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
