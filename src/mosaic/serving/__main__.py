from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from .app import create_app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    os.chdir(args.root.resolve())
    uvicorn.run(create_app(args.root, device=args.device), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
