"""Verify MOSAIC publication hygiene and frozen aggregate boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_NAME = "nika"
PUBLIC_EMAIL = "129031523+nikashen@users.noreply.github.com"
REPOSITORY_URL = "https://github.com/nikashen/MOSAIC-Retrieval"
PAGES_URL = "https://nikashen.github.io/MOSAIC-Retrieval/"
MAX_FILE_BYTES = 5 * 1024 * 1024

REQUIRED_FILES = {
    ".github/workflows/ci.yml", ".github/workflows/pages.yml", "LICENSE",
    "README.md", "THIRD_PARTY_NOTICES.md", "docs/index.html", "docs/app.js",
    "docs/styles.css", "evidence/public_metrics.json", "reports/evidence_card.md",
    "reports/mosaic_external_final_v1.json", "reports/mosaic_external_final_v1.audit.json",
    "reports/mosaic_external_final_v1.public_wording_correction.json",
    "reports/mosaic_msrvtt_frozen_final_v1.json", "reports/mosaic_msrvtt_frozen_final_v1.audit.json",
    "pyproject.toml",
}
FORBIDDEN_SUFFIXES = {
    ".bin", ".ckpt", ".csv", ".db", ".gif", ".index", ".jpeg", ".jpg",
    ".jsonl", ".mp4", ".npy", ".npz", ".parquet", ".png", ".pt", ".pth",
    ".safetensors", ".sqlite", ".whl", ".zip",
}
FORBIDDEN_DIRECTORIES = {
    ".pytest_cache", ".ruff_cache", "__pycache__", "artifacts", "build", "data",
    "datasets", "dist", "logs", "models", "outputs", "runtime-data",
}
SECRET_PATTERNS = (
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-{5}BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-{5}"),
)
WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9+.-])([A-Za-z]:[\\/](?![\\/])[^\s\"'`<>|]*)")
REAL_VIDEO_SAMPLE = re.compile(r"\bvideo7\d{3}\b", re.IGNORECASE)


class ReleaseError(RuntimeError):
    pass


def _run_git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments], check=check, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )


def _paths() -> list[str]:
    if (ROOT / ".git").exists():
        tracked = _run_git("ls-files", "--cached", "-z").stdout.split("\0")
        staged = _run_git("diff", "--cached", "--name-only", "--diff-filter=D", "-z").stdout.split("\0")
        untracked = _run_git("ls-files", "--others", "--exclude-standard", "-z").stdout.split("\0")
        deleted = set(value for value in staged if value)
        return sorted(
            set(value for value in (*tracked, *untracked) if value and value not in deleted)
        )
    output = []
    for directory, names, files in os.walk(ROOT):
        names[:] = [name for name in names if name not in {".git", ".venv", "__pycache__"}]
        for name in files:
            output.append((Path(directory) / name).relative_to(ROOT).as_posix())
    return sorted(output)


def _text(path: Path) -> str | None:
    payload = path.read_bytes()
    if b"\0" in payload:
        return None
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(relative: str) -> dict[str, Any]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReleaseError(f"JSON root must be an object: {relative}")
    return value


def _check_paths(paths: list[str]) -> int:
    missing = sorted(REQUIRED_FILES - set(paths))
    if missing:
        raise ReleaseError(f"required files are missing: {missing}")
    violations: list[str] = []
    total_bytes = 0
    for relative in paths:
        path = ROOT / relative
        parts = {part.lower() for part in Path(relative).parts}
        if not path.is_file() or path.is_symlink():
            violations.append(f"non-regular release path: {relative}")
            continue
        total_bytes += path.stat().st_size
        if path.stat().st_size > MAX_FILE_BYTES:
            violations.append(f"file exceeds 5 MiB: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"private/media/build artifact suffix: {relative}")
        if parts & FORBIDDEN_DIRECTORIES or any(part.endswith(".egg-info") for part in parts):
            violations.append(f"private/build directory: {relative}")
    if violations:
        raise ReleaseError("\n".join(violations))
    return total_bytes


def _check_text(paths: list[str]) -> None:
    violations: list[str] = []
    for relative in paths:
        text = _text(ROOT / relative)
        if text is None:
            continue
        if WINDOWS_PATH.search(text):
            violations.append(f"machine-local path in {relative}")
        if REAL_VIDEO_SAMPLE.search(text):
            violations.append(f"real MSR-VTT sample identifier in {relative}")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            violations.append(f"secret-shaped value in {relative}")
    if violations:
        raise ReleaseError("\n".join(violations))


def _check_claims() -> None:
    public = _load_json("evidence/public_metrics.json")
    coco = public["coco_external_final"]
    if coco["reported_adapter_view"] != "text-to-image image-only full-catalog retrieval":
        raise ReleaseError("COCO image-only metric scope was broadened")
    if coco["adapter_minus_zero_shot"]["t2i_r1"]["delta"] != 0.0289:
        raise ReleaseError("COCO R@1 evidence changed")
    video = public["msrvtt_frozen_final"]
    if video["temporal_minus_mean"]["t2v_r10"]["delta"] != 0.079:
        raise ReleaseError("MSR-VTT R@10 evidence changed")
    attribution = public["dev_only_attribution"]
    if attribution["hard_negative_independent_support"] or attribution["test_accessed"]:
        raise ReleaseError("Dev-only attribution boundary was broadened")
    boundary = public["publication_boundary"]
    expected_false = {
        "raw_images_published", "raw_videos_published", "captions_or_queries_published",
        "query_level_rankings_published", "model_or_adapter_weights_published",
        "derived_features_or_indexes_published", "online_ab_test", "production_sla", "sota_claim",
    }
    if any(boundary[key] is not False for key in expected_false):
        raise ReleaseError("publication or claim boundary was broadened")

    coco_report = _load_json("reports/mosaic_external_final_v1.json")
    if coco_report["statistics"]["adapter_vs_zero_shot_image_only"]["recall@1"]["delta"] != 0.0289:
        raise ReleaseError("public COCO summary differs from frozen report")
    coco_audit = _load_json("reports/mosaic_external_final_v1.audit.json")
    if coco_audit["status"] != "complete" or coco_audit["report_sha256"] != _sha256(ROOT / "reports/mosaic_external_final_v1.json"):
        raise ReleaseError("COCO report/audit binding failed")
    wording = _load_json("reports/mosaic_external_final_v1.public_wording_correction.json")
    if (
        wording["original_frozen_markdown_sha256"] != coco_audit["report_markdown_sha256"]
        or wording["corrected_public_markdown_sha256"] != _sha256(ROOT / "reports/mosaic_external_final_v1.md")
        or wording["frozen_json_sha256"] != coco_audit["report_sha256"]
        or wording["metrics_or_statistics_changed"] is not False
        or wording["external_final_reread"] is not False
        or wording["historical_audit_modified"] is not False
    ):
        raise ReleaseError("COCO public wording-correction binding failed")
    video_report = _load_json("reports/mosaic_msrvtt_frozen_final_v1.json")
    video_audit = _load_json("reports/mosaic_msrvtt_frozen_final_v1.audit.json")
    if video_audit["status"] != "completed" or video_audit["evaluation_sha256"] != video_report["evaluation_sha256"]:
        raise ReleaseError("MSR-VTT evaluation/audit binding failed")
    if video_audit["report"]["json_sha256"] != _sha256(ROOT / "reports/mosaic_msrvtt_frozen_final_v1.json"):
        raise ReleaseError("MSR-VTT report hash binding failed")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for marker in (REPOSITORY_URL, PAGES_URL, "image-only", "0.304/0.631", "0.335/0.710"):
        if marker not in readme:
            raise ReleaseError(f"README missing boundary marker: {marker}")
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for marker in ("does not grant rights", "No COCO image", "does not declare a dataset license"):
        if marker not in notices:
            raise ReleaseError(f"third-party boundary missing: {marker}")


def _check_history() -> str:
    if not (ROOT / ".git").exists():
        return "source-export"
    head = _run_git("rev-parse", "--verify", "HEAD", check=False)
    if head.returncode != 0:
        return "no-commit"
    roots = [value for value in _run_git("rev-list", "HEAD", "--max-parents=0").stdout.splitlines() if value]
    if len(roots) != 1:
        raise ReleaseError(f"HEAD history must have exactly one root commit, found {len(roots)}")
    fields = [value.strip() for value in _run_git("log", "HEAD", "--format=%an%x00%ae%x00%cn%x00%ce%x00").stdout.split("\0") if value.strip()]
    identities = [tuple(fields[index:index + 4]) for index in range(0, len(fields), 4)]
    expected = (PUBLIC_NAME, PUBLIC_EMAIL, PUBLIC_NAME, PUBLIC_EMAIL)
    if not identities or any(identity != expected for identity in identities):
        raise ReleaseError(f"unexpected Git identity: {identities}")
    return "single-root-public-identity"


def verify_public_release() -> dict[str, Any]:
    paths = _paths()
    total_bytes = _check_paths(paths)
    _check_text(paths)
    _check_claims()
    return {
        "status": "pass", "scope": "sanitized-public-snapshot",
        "checked_paths": len(paths), "checked_bytes": total_bytes,
        "git_history": _check_history(), "raw_images": 0, "raw_videos": 0,
        "captions_or_queries": 0, "query_level_rankings": 0,
        "model_or_adapter_weights": 0, "benchmark_reproduced": False,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(verify_public_release(), ensure_ascii=False, indent=2))
    except ReleaseError as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
