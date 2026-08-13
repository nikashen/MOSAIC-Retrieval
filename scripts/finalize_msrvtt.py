from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic.video.reporting import finalize_msrvtt


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot frozen MSR-VTT-1K-A finalizer")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/msrvtt_test_1ka_v1.json"))
    parser.add_argument("--features", type=Path, default=Path("artifacts/mosaic_msrvtt_1ka_v1/test_clip_features.npz"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/mosaic_msrvtt_1ka_v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/msrvtt_1ka_v1.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/mosaic_msrvtt_frozen_final_v1.json"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/mosaic_msrvtt_frozen_final_v1.md"))
    parser.add_argument("--audit", type=Path, default=Path("reports/mosaic_msrvtt_frozen_final_v1.audit.json"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = finalize_msrvtt(
        repo_root=root,
        manifest_path=args.manifest,
        feature_path=args.features,
        checkpoint_dir=args.checkpoint_dir,
        config_path=args.config,
        report_path=args.report,
        markdown_path=args.markdown,
        audit_path=args.audit,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "report": str(args.report.resolve()),
                "audit": str(args.audit.resolve()),
                "videos": report["evaluation"]["videos"],
                "captions": report["evaluation"]["captions"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
