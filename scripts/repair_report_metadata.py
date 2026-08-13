from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a non-metric final-report metadata correction once")
    parser.add_argument("--report", type=Path, default=Path("reports/mosaic_external_final_v1.json"))
    parser.add_argument("--ablation", type=Path, default=Path("reports/mosaic_dev_ablation_v1.json"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    audit_path = args.report.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "complete" or audit.get("metadata_correction"):
        raise RuntimeError("audit is not eligible for a one-time metadata correction")
    core_before = digest({"metrics": report["metrics"], "statistics": report["statistics"]})
    previous_hash = file_hash(args.report)
    report["dev_ablation"] = json.loads(args.ablation.read_text(encoding="utf-8"))
    report.setdefault("provenance", {})["post_evaluation_metadata_correction"] = {
        "reason": "replace pre-determinism Dev ablation payload with rerun deterministic Dev-only payload",
        "external_metrics_or_statistics_reread": False,
        "core_metrics_statistics_sha256_before": core_before,
        "previous_report_sha256": previous_hash,
        "corrected_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core_after = digest({"metrics": report["metrics"], "statistics": report["statistics"]})
    if core_after != core_before:
        raise AssertionError("metadata repair changed final metrics or statistics")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit["metadata_correction"] = {
        "core_metrics_statistics_sha256": core_after,
        "previous_report_sha256": previous_hash,
        "new_report_sha256": file_hash(args.report),
        "ablation_sha256": file_hash(args.ablation),
        "external_metrics_or_statistics_reread": False,
    }
    audit["report_sha256"] = file_hash(args.report)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit["metadata_correction"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

