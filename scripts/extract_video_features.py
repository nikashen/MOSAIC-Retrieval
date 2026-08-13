from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic.video.features import build_video_feature_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract resumable frozen-CLIP MSR-VTT features")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--frames-per-video", type=int, default=12)
    parser.add_argument("--video-batch-size", type=int, default=4)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()
    metadata = build_video_feature_bundle(
        args.manifest,
        args.output,
        model_name=args.model,
        model_revision=args.revision,
        device=args.device,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        local_files_only=args.local_files_only,
        frames_per_video=args.frames_per_video,
        video_batch_size=args.video_batch_size,
        decode_workers=args.decode_workers,
        image_batch_size=args.image_batch_size,
        text_batch_size=args.text_batch_size,
        max_videos=args.max_videos,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
