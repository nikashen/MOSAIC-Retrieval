from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic.integrations.project4 import export_content_vectors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--features", dest="input_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, default=None)
    parser.add_argument("--allow-identity-demo", action="store_true")
    parser.add_argument("--encoder-version", default="mosaic-coco5k-v1")
    args = parser.parse_args()
    result = export_content_vectors(
        args.input_path,
        args.output,
        mapping=args.mapping,
        allow_identity_demo=args.allow_identity_demo,
        encoder_version=args.encoder_version,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
