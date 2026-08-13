from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

from mosaic.data import load_manifest
from mosaic.features import load_feature_bundle


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text:
            self.parts.append(text)


def _visible_html_text(path: Path) -> str:
    parser = _VisibleTextParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return " ".join(parser.parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict MOSAIC repository/evidence audit")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, details: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    manifest_path = root / "data" / "processed" / "coco_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = load_manifest(manifest_path)
            check("manifest_digest", True, manifest["manifest_sha256"])
            counts = manifest["split"]["counts"]
            check("cluster_split_nonempty", all(int(counts[s]) > 0 for s in ("train", "dev", "test")), str(counts))
            check("caption_cluster_count", all(len(row["captions"]) >= 1 for row in manifest["images"]), "all images have captions")
        except Exception as exc:
            check("manifest_digest", False, f"{type(exc).__name__}: {exc}")
    else:
        check("manifest_present", False, "run data preparation first")

    feature_path = root / "artifacts" / "mosaic_coco5k_v1" / "clip_features.npz"
    if feature_path.is_file():
        try:
            metadata, arrays = load_feature_bundle(feature_path)
            check("safe_feature_bundle", True, f"{arrays['image_features'].shape}")
            check("feature_norms", bool(np.allclose(np.linalg.norm(arrays["image_features"], axis=1), 1, atol=2e-2)), "L2")
            check("feature_manifest_link", bool(manifest_path.is_file() and metadata.get("manifest_sha256") == manifest["manifest_sha256"]), "hash link")
        except Exception as exc:
            check("safe_feature_bundle", False, f"{type(exc).__name__}: {exc}")
    else:
        check("feature_bundle_present", False, "run feature extraction first")

    report = root / "reports" / "mosaic_external_final_v1.json"
    if report.is_file():
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
            required = {"schema_version", "dataset", "model", "metrics", "claim_boundary", "provenance"}
            check("report_schema", required.issubset(payload), str(sorted(set(payload) & required)))
            boundary = payload.get("claim_boundary", {})
            forbidden_true = [
                key
                for key in ("sota_claimed", "production_traffic", "online_ab_test")
                if boundary.get(key) is not False
            ]
            check("claim_boundary_false", not forbidden_true, ",".join(forbidden_true))
            ui = _visible_html_text(
                root / "src" / "mosaic" / "serving" / "static" / "index.html"
            )
            stats = payload["statistics"]["adapter_vs_zero_shot_image_only"]
            expected_ui = {
                f"+{stats['recall@1']['delta'] * 100:.2f} pp",
                f"+{stats['recall@10']['delta'] * 100:.2f} pp",
                f"95% CI [{stats['recall@1']['lower'] * 100:+.2f}, {stats['recall@1']['upper'] * 100:+.2f}]",
                f"95% CI [{stats['recall@10']['lower'] * 100:+.2f}, {stats['recall@10']['upper'] * 100:+.2f}]",
            }
            missing_ui = sorted(value for value in expected_ui if value not in ui)
            check("ui_evidence_matches_report", not missing_ui, ",".join(missing_ui))
            audit_path = report.with_suffix(".audit.json")
            if audit_path.is_file():
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                check("final_audit_complete", audit.get("status") == "complete", str(audit.get("status")))
                check("final_report_hash", audit.get("report_sha256") == _sha256(report), str(audit.get("report_sha256")))
                correction = audit.get("metadata_correction")
                if correction:
                    check(
                        "metadata_correction_preserved_metrics",
                        correction.get("external_metrics_or_statistics_reread") is False
                        and bool(correction.get("core_metrics_statistics_sha256")),
                        str(correction.get("core_metrics_statistics_sha256")),
                    )
            else:
                check("final_audit_present", False, "external final report has no audit")
        except Exception as exc:
            check("report_schema", False, f"{type(exc).__name__}: {exc}")
    else:
        check("formal_report_present", False, "formal evaluation has not run")

    # Run the unit suite as part of the audit, but do not hide its output.
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-q"],
        cwd=str(root),
        env={**__import__("os").environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )
    check("unit_tests", completed.returncode == 0, (completed.stdout + completed.stderr).strip()[-500:])
    summary = {"schema_version": "mosaic.audit.v1", "checks": checks, "passed": all(item["passed"] for item in checks)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
