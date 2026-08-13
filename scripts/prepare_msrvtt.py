from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mosaic.video.data import build_msrvtt_manifests


SOURCE = "friedrichor/MSR-VTT@c1af215a96934854f42683c19c51391aaee6f962"
ARCHIVE_SHA256 = "be4935000f7f9470ff9852d833b2ec808fcfaeb5beb3b2c08d25894b59352196"


def _video_root(raw_root: Path, marker: dict[str, object]) -> Path:
    configured = os.environ.get("MOSAIC_VIDEO_ROOT")
    if configured:
        root = Path(configured).resolve()
    else:
        recorded = Path(str(marker["video_root"]))
        root = recorded.resolve() if recorded.is_absolute() else (raw_root / recorded).resolve()
        if not root.is_dir():
            root = (raw_root / "video").resolve()
        if os.path.commonpath((str(root), str(raw_root.resolve()))) != str(raw_root.resolve()):
            raise ValueError("video root must remain inside raw root unless MOSAIC_VIDEO_ROOT is set")
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/msrvtt"))
    parser.add_argument("--train-output", type=Path, default=Path("data/processed/msrvtt_train_dev_v1.json"))
    parser.add_argument("--test-output", type=Path, default=Path("data/processed/msrvtt_test_1ka_v1.json"))
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    raw_root = args.raw_root.resolve()
    marker = json.loads((raw_root / ".mosaic_msrvtt_extracted.json").read_text(encoding="utf-8"))
    audit = json.loads((raw_root / "download_audit.json").read_text(encoding="utf-8"))
    if audit.get("schema_version") != "mosaic.msrvtt_download.v1" or audit.get("source") != SOURCE:
        raise ValueError("MSR-VTT download audit source mismatch")
    if marker.get("archive_sha256") != ARCHIVE_SHA256 or int(marker.get("videos", -1)) != 10_000:
        raise ValueError("MSR-VTT extraction marker is not pinned or complete")
    if marker.get("crc32_verified") is not True:
        raise ValueError("MSR-VTT extracted files have not passed full CRC verification")
    train, test = build_msrvtt_manifests(
        raw_root / "msrvtt_train_9k.json",
        raw_root / "msrvtt_test_1k.json",
        _video_root(raw_root, marker),
        args.train_output,
        args.test_output,
        require_files=not args.allow_missing,
    )
    print(
        json.dumps(
            {
                "train_dev_manifest_sha256": train["manifest_sha256"],
                "test_manifest_sha256": test["manifest_sha256"],
                "train_dev_counts": train["selection"]["counts"],
                "test_counts": test["selection"]["counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
