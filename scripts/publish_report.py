from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mosaic.data import load_manifest
from mosaic.experiment import evaluate_model, strip_internal
from mosaic.metrics import metric_vectors_from_ranks, paired_bootstrap_delta_ci
from mosaic.reranking import evaluate_reranker


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_snapshot(root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    clean = not subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    return commit, clean


def paired_section(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    mode: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    candidate_ranks = candidate["_ranks"][mode]
    baseline_ranks = baseline["_ranks"][mode]
    clusters = candidate["_clusters"][mode]
    if not np.array_equal(clusters, baseline["_clusters"][mode]):
        raise ValueError(f"paired cluster alignment failed for {mode}")
    left = metric_vectors_from_ranks(candidate_ranks, (1, 10))
    right = metric_vectors_from_ranks(baseline_ranks, (1, 10))
    return {
        key: paired_bootstrap_delta_ci(
            left[key], right[key], clusters, replicates=replicates, seed=seed + index
        )
        for index, key in enumerate(("recall@1", "recall@10", "mrr"))
    }


def markdown(report: dict[str, Any]) -> str:
    zero = report["metrics"]["zero_shot_clip"]
    trained = report["metrics"]["mosaic_adapter"]
    reranker = report["metrics"]["interaction_reranker"]
    paired = report["statistics"]["adapter_vs_zero_shot_image_only"]
    return f"""# MOSAIC-Retrieval External Final

## Protocol

- Dataset: {report['dataset']['name']}
- Images/captions: {report['dataset']['images']} / {report['dataset']['captions']}
- Split: `{report['dataset']['split']}`; these images were not used for this project's adapter/reranker training or Dev selection.
- Pretrained CLIP may have seen public COCO during internet-scale pretraining; this limitation is explicit.

## Main full-catalog results by retrieval view

The Text→Image columns below use the **image-only item view**. They measure the residual adapter over frozen CLIP image embeddings; they are not a fusion/gating gain. Image→Text uses an image query against the caption catalog.

| Model | Image-only Text→Image R@1 | R@10 | MRR | Image-query Image→Text R@1 | R@10 |
|---|---:|---:|---:|---:|---:|
| Zero-shot CLIP | {zero['text_to_image']['image_only']['recall_at']['1']:.4f} | {zero['text_to_image']['image_only']['recall_at']['10']:.4f} | {zero['text_to_image']['image_only']['mrr']:.4f} | {zero['image_to_text']['recall_at']['1']:.4f} | {zero['image_to_text']['recall_at']['10']:.4f} |
| MOSAIC residual adapter | {trained['text_to_image']['image_only']['recall_at']['1']:.4f} | {trained['text_to_image']['image_only']['recall_at']['10']:.4f} | {trained['text_to_image']['image_only']['mrr']:.4f} | {trained['image_to_text']['recall_at']['1']:.4f} | {trained['image_to_text']['recall_at']['10']:.4f} |
| Adapter + interaction reranker | {reranker['reranked']['recall_at']['1']:.4f} | {reranker['reranked']['recall_at']['10']:.4f} | {reranker['reranked']['mrr']:.4f} | — | — |

Paired image-cluster bootstrap, adapter minus zero-shot on the image-only Text→Image view:

- Text→Image R@1: {paired['recall@1']['delta']:+.4f}, 95% CI [{paired['recall@1']['lower']:+.4f}, {paired['recall@1']['upper']:+.4f}]
- Text→Image R@10: {paired['recall@10']['delta']:+.4f}, 95% CI [{paired['recall@10']['lower']:+.4f}, {paired['recall@10']['upper']:+.4f}]
- Text→Image MRR: {paired['mrr']['delta']:+.4f}, 95% CI [{paired['mrr']['lower']:+.4f}, {paired['mrr']['upper']:+.4f}]

## Modality diagnostic

The leave-one-caption-out cold-start protocol never inserts a query caption verbatim into its own item metadata. Full, image-only and text-only results are all retained in the JSON fact source. Modality dropout did not improve the standard dual-encoder Dev score in the ablation; no unsupported robustness gain is claimed.

## Claim boundary

This is offline public-data evidence. It is not evidence of SOTA, video/ASR/OCR completion, production traffic, revenue lift, or an online A/B test.

Input commit: `{report['provenance']['input_commit']}`
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and freeze MOSAIC final evaluation")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--reranker-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/coco5k_v1.json"))
    parser.add_argument("--split", default="external_final")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    commit, clean = git_snapshot(root)
    if not args.diagnostic and not clean:
        raise RuntimeError("formal final requires a clean Git input snapshot")
    manifest = load_manifest(args.manifest)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    audit = args.output.with_suffix(".audit.json")
    if not args.diagnostic:
        if audit.exists() or args.output.exists():
            raise RuntimeError("formal final audit/report already exists; refusing a second read")
        audit.parent.mkdir(parents=True, exist_ok=True)
        initial_audit = {
            "schema_version": "mosaic.final_audit.v1",
            "status": "started_before_metric_read",
            "input_commit": commit,
            "manifest_sha256": manifest["manifest_sha256"],
            "feature_sha256": sha256_file(args.features),
            "config_sha256": sha256_file(args.config),
        }
        fd = os.open(audit, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(initial_audit, handle, indent=2)
            handle.write("\n")

    zero = evaluate_model(
        args.manifest,
        args.features,
        config,
        checkpoint_dir=None,
        device=args.device,
        split=args.split,
        bootstrap_replicates=args.replicates,
    )
    trained = evaluate_model(
        args.manifest,
        args.features,
        config,
        checkpoint_dir=args.adapter_dir,
        device=args.device,
        split=args.split,
        bootstrap_replicates=args.replicates,
    )
    reranker = evaluate_reranker(
        args.manifest,
        args.features,
        args.adapter_dir,
        args.reranker_dir,
        config,
        split=args.split,
        device=args.device,
        bootstrap_replicates=args.replicates,
    )
    counts = manifest["split"]["counts"]
    image_count = int(counts[args.split])
    caption_count = int(sum(len(row["captions"]) for row in manifest["images"] if row["split"] == args.split))
    seed = int(config["evaluation"]["bootstrap_seed"]) + 1000
    ablation_path = root / "reports" / "mosaic_dev_ablation_v1.json"
    report = {
        "schema_version": "mosaic.external_final.v1" if not args.diagnostic else "mosaic.diagnostic.v1",
        "dataset": {
            "name": manifest["dataset"],
            "images": image_count,
            "captions": caption_count,
            "split": args.split,
            "manifest_sha256": manifest["manifest_sha256"],
            "selection": manifest["split"],
            "pretrained_clip_public_data_exposure_unknown": True,
        },
        "model": {
            "backbone": config["backbone"],
            "adapter": config["model"],
            "training_summary": json.loads((args.adapter_dir / "training_summary.json").read_text(encoding="utf-8")),
            "reranker_summary": json.loads((args.reranker_dir / "reranker_summary.json").read_text(encoding="utf-8")),
        },
        "metrics": {
            "zero_shot_clip": strip_internal(zero),
            "mosaic_adapter": strip_internal(trained),
            "interaction_reranker": reranker,
        },
        "statistics": {
            "method": "paired image-cluster percentile bootstrap",
            "replicates": args.replicates,
            "adapter_vs_zero_shot_image_only": paired_section(
                trained, zero, "image_only", replicates=args.replicates, seed=seed
            ),
            "adapter_vs_zero_shot_image_to_text": paired_section(
                trained, zero, "image_to_text", replicates=args.replicates, seed=seed + 10
            ),
            "trained_full_vs_trained_image_only": None,
        },
        "dev_ablation": json.loads(ablation_path.read_text(encoding="utf-8")) if ablation_path.is_file() else None,
        "claim_boundary": {
            "offline_public_data": True,
            "sota_claimed": False,
            "production_traffic": False,
            "online_ab_test": False,
            "video_asr_ocr_evaluated": False,
            "project4_catalog_directly_evaluated": False,
        },
        "provenance": {
            "input_commit": commit,
            "snapshot_files_clean": clean,
            "config_sha256": sha256_file(args.config),
            "features_sha256": sha256_file(args.features),
            "adapter_sha256": sha256_file(args.adapter_dir / "adapter.safetensors"),
            "reranker_sha256": sha256_file(args.reranker_dir / "reranker.safetensors"),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "safe_npz_allow_pickle": False,
        },
    }
    # Replace the accidental self-comparison with the intended full-vs-image
    # paired vectors while keeping one shared cluster resampling sequence.
    full_left = {"_ranks": {"x": trained["_ranks"]["full"]}, "_clusters": {"x": trained["_clusters"]["full"]}}
    image_right = {"_ranks": {"x": trained["_ranks"]["image_only"]}, "_clusters": {"x": trained["_clusters"]["image_only"]}}
    report["statistics"]["trained_full_vs_trained_image_only"] = paired_section(
        full_left, image_right, "x", replicates=args.replicates, seed=seed + 20
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    if not args.diagnostic:
        completed_audit = json.loads(audit.read_text(encoding="utf-8"))
        completed_audit.update(
            {
                "status": "complete",
                "report_sha256": sha256_file(args.output),
                "report_markdown_sha256": sha256_file(args.output.with_suffix(".md")),
            }
        )
        audit.write_text(json.dumps(completed_audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "schema_version": report["schema_version"], "input_commit": commit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
