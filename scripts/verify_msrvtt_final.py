from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic.video.reporting import verify_final_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify frozen MSR-VTT evidence without re-evaluation")
    parser.add_argument("--audit", type=Path, default=Path("reports/mosaic_msrvtt_frozen_final_v1.audit.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = verify_final_audit(root, args.audit)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
