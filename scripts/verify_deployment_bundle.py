from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = (
    "Dockerfile",
    "compose.yaml",
    "run_project.sh",
    "pyproject.toml",
    "src/mosaic/serving/app.py",
    "src/mosaic/serving/engine.py",
    "src/mosaic/serving/static/index.html",
    "src/mosaic/serving/static/app.js",
    "src/mosaic/serving/static/style.css",
    "docs/DEPLOY_OTHER_MACHINE.md",
    "configs/msrvtt_1ka_v1.json",
    "configs/msrvtt_dev_ablation_v1.json",
    "docs/MSRVTT_PROTOCOL.md",
    "src/mosaic/video/data.py",
    "src/mosaic/video/features.py",
    "src/mosaic/video/models.py",
    "src/mosaic/video/experiment.py",
    "src/mosaic/video/reporting.py",
    "scripts/finalize_msrvtt.py",
    "scripts/verify_msrvtt_final.py",
    "scripts/run_video_dev_ablations.py",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    missing = [path for path in REQUIRED if not (root / path).is_file()]
    static = (root / "src/mosaic/serving/static/index.html").read_text(encoding="utf-8")
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    compose = (root / "compose.yaml").read_text(encoding="utf-8")
    docker_contract = all(
        value in dockerfile
        for value in (
            "COPY src ./src", "COPY reports ./reports", "LICENSE THIRD_PARTY_NOTICES.md",
            '"--port", "8050"',
        )
    ) and all(value in compose for value in ("8050:8050", "/app/artifacts", "/app/data"))
    report = {
        "schema_version": "mosaic.deployment_bundle.v1",
        "required_files": len(REQUIRED),
        "missing": missing,
        "static_routes_declared": all(value in static for value in ("/static/style.css", "/static/app.js")),
        "docker_runtime_contract": docker_contract,
        "passed": not missing and docker_contract,
        "docker_build_executed": False,
        "docker_build_boundary": "Docker daemon may be unavailable; this check validates reproducible bundle structure only.",
    }
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
