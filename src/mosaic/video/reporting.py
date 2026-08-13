from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .experiment import CHECKPOINT_CONFIG_NAME, SUMMARY_NAME, evaluate_video_suite


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _reserve(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _git_state(root: Path) -> tuple[str | None, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None, "git-unavailable"
    return commit or None, status


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"formal evidence path must be inside repository: {path}") from error


def _metric_table(suite: dict[str, Any], direction: str) -> list[str]:
    labels = {
        "frozen_clip_mean_pool": "Frozen CLIP mean",
        "frozen_clip_max_pool": "Frozen CLIP max",
        "mosaic_trained_temporal_attention": "MOSAIC temporal",
    }
    lines = ["| Model | R@1 | R@5 | R@10 | MRR |", "|---|---:|---:|---:|---:|"]
    for key, label in labels.items():
        metric = suite["models"][key][direction]
        recall = metric["recall_at"]
        lines.append(
            f"| {label} | {recall['1']:.4f} | {recall['5']:.4f} | "
            f"{recall['10']:.4f} | {metric['mrr']:.4f} |"
        )
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    suite = report["evaluation"]
    paired = suite["paired_video_cluster_bootstrap_trained_vs_mean"]
    lines = [
        "# MOSAIC MSR-VTT-1K-A Frozen Final",
        "",
        f"- Input commit: `{report['provenance']['input_commit']}`",
        f"- Videos / query captions: `{suite['videos']} / {suite['captions']}`",
        "- Protocol: JSFusion 1K-A, one official query caption per video.",
        "",
        "## Text-to-video",
        "",
        *_metric_table(suite, "text_to_video"),
        "",
        "## Video-to-text",
        "",
        *_metric_table(suite, "video_to_text"),
        "",
        "## Paired video-cluster bootstrap: trained minus frozen mean",
        "",
        "| Direction / metric | Delta | 95% CI | Interpretation |",
        "|---|---:|---:|---|",
    ]
    for direction in ("text_to_video", "video_to_text"):
        for metric in ("recall@1", "recall@10", "mrr"):
            value = paired[direction][metric]
            if value["lower"] > 0:
                interpretation = "positive under this protocol"
            elif value["upper"] < 0:
                interpretation = "negative under this protocol"
            else:
                interpretation = "CI crosses zero; no directional claim"
            lines.append(
                f"| {direction} {metric} | {value['delta']:+.4f} | "
                f"[{value['lower']:+.4f}, {value['upper']:+.4f}] | {interpretation} |"
            )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "This is a single controlled offline evaluation on a public MSR-VTT mirror. "
            "It does not establish SOTA, online lift, production latency, audio/ASR/OCR quality, "
            "or independence from CLIP pretraining data. The one-caption 1K-A protocol is not "
            "interchangeable with evaluations using all test captions.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_msrvtt(
    *,
    repo_root: Path,
    manifest_path: Path,
    feature_path: Path,
    checkpoint_dir: Path,
    config_path: Path,
    report_path: Path,
    markdown_path: Path,
    audit_path: Path,
    device: str = "auto",
    expected_videos: int = 1000,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Run the frozen Test exactly once for one audit filename."""

    root = Path(repo_root).resolve()
    paths = {
        "test_manifest": Path(manifest_path).resolve(),
        "test_features": Path(feature_path).resolve(),
        "checkpoint_weights": (Path(checkpoint_dir) / "video_encoder.safetensors").resolve(),
        "checkpoint_config": (Path(checkpoint_dir) / CHECKPOINT_CONFIG_NAME).resolve(),
        "training_summary": (Path(checkpoint_dir) / SUMMARY_NAME).resolve(),
        "config": Path(config_path).resolve(),
    }
    report_path = Path(report_path).resolve()
    markdown_path = Path(markdown_path).resolve()
    audit_path = Path(audit_path).resolve()
    if report_path.exists() or markdown_path.exists() or audit_path.exists():
        raise FileExistsError("formal MSR-VTT report/audit already exists")
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    commit, status = _git_state(root)
    if require_clean and (commit is None or status):
        raise RuntimeError(f"formal finalization requires a clean tracked worktree: {status}")
    input_commit = commit or "unversioned-test-fixture"
    checkpoint_payload = json.loads(paths["checkpoint_config"].read_text(encoding="utf-8"))
    training_summary = json.loads(paths["training_summary"].read_text(encoding="utf-8"))
    if checkpoint_payload.get("test_labels_accessed") is not False:
        raise ValueError("checkpoint provenance does not prove Dev-only selection")
    if training_summary.get("selection", {}).get("test_labels_accessed") is not False:
        raise ValueError("training summary does not prove Dev-only selection")
    training_commit = checkpoint_payload.get("training_input_commit")
    if training_summary.get("input_commit") != training_commit:
        raise ValueError("checkpoint/training-summary input commits disagree")
    if require_clean:
        if not training_commit:
            raise ValueError("formal checkpoint is missing its training input commit")
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", str(training_commit), input_commit],
            cwd=root,
            check=False,
        ).returncode
        if ancestor != 0:
            raise ValueError("training input commit is not an ancestor of Final input commit")
    started = {
        "schema_version": "mosaic.msrvtt_final_audit.v1",
        "status": "started",
        "started_at": _now(),
        "input_commit": input_commit,
        "policy": (
            "O_EXCL reservation before this finalizer loads the frozen derived Test "
            "features or computes retrieval metrics; feature extraction occurred after "
            "Dev freeze and did not compute selection metrics"
        ),
    }
    _reserve(audit_path, started)
    try:
        config = json.loads(paths["config"].read_text(encoding="utf-8"))
        input_files = {
            key: {"path": _relative(root, path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for key, path in paths.items()
        }
        suite = evaluate_video_suite(
            paths["test_manifest"],
            paths["test_features"],
            config,
            checkpoint_dir=Path(checkpoint_dir),
            device=device,
            split="test",
        )
        if suite["videos"] != int(expected_videos) or suite["captions"] != int(expected_videos):
            raise ValueError("formal 1K-A requires exactly one caption for every expected video")
        report = {
            "schema_version": "mosaic.msrvtt_frozen_final.v1",
            "experiment_id": config.get("experiment_id"),
            "created_at": _now(),
            "protocol": {
                "dataset": "MSR-VTT-1K-A",
                "mirror": config.get("dataset", {}).get("mirror"),
                "mirror_commit": config.get("dataset", {}).get("mirror_commit"),
                "query_policy": "one official JSFusion query caption per video",
                "selection": "all model/epoch decisions completed on deterministic Dev only",
                "bootstrap_cluster": "video_id",
            },
            "provenance": {
                "input_commit": input_commit,
                "training_input_commit": training_commit,
                "files": input_files,
            },
            "training_selection": training_summary.get("selection"),
            "evaluation": suite,
            "claim_boundary": {
                **config.get("claim_boundary", {}),
                "public_data_offline_only": True,
                "clip_pretraining_overlap_unknown": True,
                "one_caption_protocol_not_all_caption_protocol": True,
                "dataset_license_declared_by_mirror": False,
            },
        }
        report["evaluation_sha256"] = canonical_sha256(report["evaluation"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(report_path, report)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        completed = {
            **started,
            "status": "completed",
            "completed_at": _now(),
            "inputs": input_files,
            "evaluation_sha256": report["evaluation_sha256"],
            "report": {
                "json_path": _relative(root, report_path),
                "json_sha256": sha256_file(report_path),
                "markdown_path": _relative(root, markdown_path),
                "markdown_sha256": sha256_file(markdown_path),
            },
        }
        _atomic_json(audit_path, completed)
        return report
    except BaseException as error:
        failed = {
            **started,
            "status": "failed",
            "failed_at": _now(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _atomic_json(audit_path, failed)
        raise


def verify_final_audit(repo_root: Path, audit_path: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    if audit.get("status") != "completed":
        raise ValueError("formal audit is not completed")
    checks: dict[str, bool] = {}
    for key, detail in audit["inputs"].items():
        path = root / detail["path"]
        checks[f"input:{key}"] = path.is_file() and path.stat().st_size == detail["bytes"] and sha256_file(path) == detail["sha256"]
    for kind in ("json", "markdown"):
        detail = audit["report"]
        path = root / detail[f"{kind}_path"]
        checks[f"report:{kind}"] = path.is_file() and sha256_file(path) == detail[f"{kind}_sha256"]
    report_path = root / audit["report"]["json_path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks["evaluation_digest"] = canonical_sha256(report["evaluation"]) == audit["evaluation_sha256"] == report["evaluation_sha256"]
    if not all(checks.values()):
        raise ValueError(f"formal audit verification failed: {checks}")
    return {"status": "verified", "checks": checks}


__all__ = [
    "canonical_sha256",
    "finalize_msrvtt",
    "render_markdown",
    "sha256_file",
    "verify_final_audit",
]
