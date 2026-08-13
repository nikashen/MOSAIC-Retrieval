from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from mosaic.metrics import metric_vectors_from_ranks, paired_bootstrap_delta_ci
from mosaic.video.data import load_video_manifest
from mosaic.video.experiment import evaluate_video_model, train_video_encoder


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": result["model_name"],
        "checkpoint_epoch": int(result.get("checkpoint_epoch", 0)),
        "text_to_video": result["text_to_video"],
        "video_to_text": result["video_to_text"],
        "rank_sha256": {
            direction: hashlib.sha256(
                np.asarray(result["_ranks"][direction], dtype="<i8").tobytes(order="C")
            ).hexdigest()
            for direction in ("text_to_video", "video_to_text")
        },
        "paired_vs_frozen_mean": result.get(
            "paired_video_cluster_bootstrap_vs_frozen_mean"
        ),
    }


def _metric(result: dict[str, Any], direction: str, metric: str) -> float:
    if metric.startswith("recall@"):
        return float(result[direction]["recall_at"][metric.split("@", 1)[1]])
    return float(result[direction][metric])


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"seeds": len(runs), "metrics": {}}
    for direction in ("text_to_video", "video_to_text"):
        output["metrics"][direction] = {}
        for metric in ("recall@1", "recall@5", "recall@10", "mrr"):
            values = [_metric(run["evaluation"], direction, metric) for run in runs]
            output["metrics"][direction][metric] = {
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
    output["best_epochs"] = [int(run["training"]["best_epoch"]) for run in runs]
    output["trainable_parameters"] = sorted(
        {int(run["training"]["trainable_parameters"]) for run in runs}
    )
    return output


def _paired(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    bootstrap_seed: int,
    replicates: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    offset = 0
    for direction in ("text_to_video", "video_to_text"):
        if not np.array_equal(left["_clusters"][direction], right["_clusters"][direction]):
            raise ValueError("paired ablation clusters differ")
        left_vectors = metric_vectors_from_ranks(left["_ranks"][direction], (1, 10))
        right_vectors = metric_vectors_from_ranks(right["_ranks"][direction], (1, 10))
        output[direction] = {}
        for metric in ("recall@1", "recall@10", "mrr"):
            output[direction][metric] = paired_bootstrap_delta_ci(
                left_vectors[metric],
                right_vectors[metric],
                left["_clusters"][direction],
                replicates=replicates,
                seed=bootstrap_seed + offset,
            )
            offset += 1
    return output


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MSR-VTT Dev-only 多种子归因消融",
        "",
        "本报告在既有 Frozen Final 之后生成，但只读取 Train/Dev；Test 未打开、未重跑，",
        "也不改变 best checkpoint 或正式 Final。",
        "",
        "## 三种子均值 ± sample std",
        "",
        "| Variant | Params | T2V R@1 | T2V R@10 | V2T R@1 | V2T R@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, aggregate in report["aggregates"].items():
        metrics = aggregate["metrics"]
        def value(direction: str, metric: str) -> str:
            row = metrics[direction][metric]
            return f"{row['mean']:.4f} ± {row['sample_std']:.4f}"
        params = "/".join(str(item) for item in aggregate["trainable_parameters"])
        lines.append(
            f"| {name} | {params} | {value('text_to_video', 'recall@1')} | "
            f"{value('text_to_video', 'recall@10')} | "
            f"{value('video_to_text', 'recall@1')} | "
            f"{value('video_to_text', 'recall@10')} |"
        )
    lines.extend(
        [
            "",
            "## Seed-matched paired comparisons",
            "",
            "每个 delta 均为左侧 variant − 右侧 variant；CI 是同一 Dev 用户/视频 cluster",
            "内的 paired bootstrap，不是跨 seed 置信区间。",
            "",
        ]
    )
    for comparison, rows in report["paired_comparisons"].items():
        lines.append(f"### {comparison}")
        lines.append("")
        lines.append("| Seed | T2V R@1 delta [CI] | T2V R@10 delta [CI] |")
        lines.append("|---:|---:|---:|")
        for row in rows:
            r1 = row["paired"]["text_to_video"]["recall@1"]
            r10 = row["paired"]["text_to_video"]["recall@10"]
            lines.append(
                f"| {row['seed']} | {r1['delta']:+.4f} "
                f"[{r1['lower']:+.4f},{r1['upper']:+.4f}] | "
                f"{r10['delta']:+.4f} [{r10['lower']:+.4f},{r10['upper']:+.4f}] |"
            )
        lines.append("")
    lines.extend(
        [
            "结论必须以 seedwise 方向和 CI 为准；如果去掉组件没有稳定退化，应报告为",
            "‘该组件未得到独立支持’，不能继续把整模型增益归因给它。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MSR-VTT Train/Dev attribution only")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/msrvtt_train_dev_v1.json"))
    parser.add_argument("--features", type=Path, default=Path("artifacts/mosaic_msrvtt_1ka_v1/train_dev_clip_features.npz"))
    parser.add_argument("--base-config", type=Path, default=Path("configs/msrvtt_1ka_v1.json"))
    parser.add_argument("--ablation-config", type=Path, default=Path("configs/msrvtt_dev_ablation_v1.json"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/mosaic_msrvtt_dev_attribution_v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/mosaic_msrvtt_dev_attribution_v1.json"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/mosaic_msrvtt_dev_attribution_v1.md"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.report.exists() or args.markdown.exists():
        raise FileExistsError("Dev attribution report already exists")
    manifest = load_video_manifest(args.manifest)
    splits = {str(row["split"]) for row in manifest["videos"]}
    if splits != {"train", "dev"}:
        raise ValueError("attribution runner requires a Train/Dev-only manifest")
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    plan = json.loads(args.ablation_config.read_text(encoding="utf-8"))
    if plan.get("scope") != "train_dev_only_never_open_test":
        raise ValueError("ablation plan is not Dev-only")
    root = Path(__file__).resolve().parents[1]
    manifest_file_sha = _sha256_file(args.manifest)
    feature_file_sha = _sha256_file(args.features)
    base_config_sha = _sha256_file(args.base_config)
    ablation_config_sha = _sha256_file(args.ablation_config)
    runs: dict[str, list[dict[str, Any]]] = {}
    internal: dict[tuple[str, int], dict[str, Any]] = {}
    for variant in plan["variants"]:
        name = str(variant["name"])
        runs[name] = []
        for seed in (int(value) for value in plan["seeds"]):
            config = copy.deepcopy(base)
            config["model"].update(
                {
                    "aggregator": variant["aggregator"],
                    "hidden_dim": int(variant["hidden_dim"]),
                    "hard_negative_weight": float(variant["hard_negative_weight"]),
                    "teacher_preservation_weight": float(variant["teacher_preservation_weight"]),
                }
            )
            config["training"]["seed"] = seed
            run_dir = args.output_root / name / str(seed)
            run_config = {
                "schema_version": "mosaic.video_dev_attribution_run.v1",
                "variant": variant,
                "seed": seed,
                "base_config_sha256": base_config_sha,
                "manifest_sha256": manifest_file_sha,
                "feature_sha256": feature_file_sha,
            }
            run_config["run_sha256"] = _canonical_sha(run_config)
            config_path = run_dir / "run_config.json"
            if config_path.is_file():
                existing = json.loads(config_path.read_text(encoding="utf-8"))
                if existing != run_config:
                    raise ValueError(f"incompatible resumable run: {name}/{seed}")
                summary_path = run_dir / "training_summary.json"
                if summary_path.is_file() and (run_dir / "video_encoder.safetensors").is_file():
                    training = json.loads(summary_path.read_text(encoding="utf-8"))
                else:
                    training = train_video_encoder(
                        args.manifest,
                        args.features,
                        run_dir,
                        config,
                        device=args.device,
                        seed=seed,
                    )
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
                training = train_video_encoder(
                    args.manifest,
                    args.features,
                    run_dir,
                    config,
                    device=args.device,
                    seed=seed,
                )
            evaluation = evaluate_video_model(
                args.manifest,
                args.features,
                config,
                checkpoint_dir=run_dir,
                device=args.device,
                split="dev",
            )
            internal[(name, seed)] = evaluation
            runs[name].append(
                {
                    "seed": seed,
                    "purpose": variant["purpose"],
                    "run_sha256": run_config["run_sha256"],
                    "training": {
                        "best_epoch": training["best_epoch"],
                        "best_score": training["selection"]["best_score"],
                        "baseline_score": training["selection"]["baseline_score"],
                        "trainable_parameters": training["trainable_parameters"],
                        "elapsed_seconds": training["elapsed_seconds"],
                    },
                    "evaluation": _compact(evaluation),
                }
            )
    comparisons: dict[str, list[dict[str, Any]]] = {}
    bootstrap_seed = int(base["evaluation"]["bootstrap_seed"]) + 10_000
    replicates = int(base["evaluation"]["bootstrap_replicates"])
    for left, right in plan["paired_comparisons"]:
        label = f"{left}_minus_{right}"
        comparisons[label] = []
        for index, seed in enumerate(int(value) for value in plan["seeds"]):
            comparisons[label].append(
                {
                    "seed": seed,
                    "paired": _paired(
                        internal[(left, seed)],
                        internal[(right, seed)],
                        bootstrap_seed=bootstrap_seed + index * 100,
                        replicates=replicates,
                    ),
                }
            )
    report = {
        "schema_version": "mosaic.video_dev_attribution.v1",
        "scope": "post_final_train_dev_only_diagnostic",
        "test_accessed": False,
        "selection_or_final_changed": False,
        "input_commit": _git_commit(root),
        "provenance": {
            "manifest_sha256": manifest_file_sha,
            "feature_sha256": feature_file_sha,
            "base_config_sha256": base_config_sha,
            "ablation_config_sha256": ablation_config_sha,
        },
        "runs": runs,
        "aggregates": {name: _aggregate(values) for name, values in runs.items()},
        "paired_comparisons": comparisons,
        "claim_boundary": plan["claim_boundary"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report.resolve()),
                "variants": len(runs),
                "runs": sum(len(values) for values in runs.values()),
                "test_accessed": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
