from __future__ import annotations

import argparse
from pathlib import Path

from mosaic.data import build_coco_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the immutable MOSAIC COCO manifest")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/coco_manifest.json"))
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    root = args.data_root.resolve()
    manifest = build_coco_manifest(
        root / "annotations" / "captions_val2017.json",
        root / "val2017",
        args.output,
        require_files=not args.allow_missing,
    )
    print({
        "output": str(args.output.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "counts": manifest["split"]["counts"],
        "images": len(manifest["images"]),
        "captions": sum(len(row["captions"]) for row in manifest["images"]),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
