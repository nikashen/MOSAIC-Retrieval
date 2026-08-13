from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mosaic.video.experiment import evaluate_video_model, strip_internal


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _markdown(suite: dict[str, object]) -> str:
    models = (
        ("Frozen CLIP mean", suite["frozen_mean"]),
        ("Frozen CLIP max", suite["frozen_max"]),
        ("MOSAIC temporal", suite["trained_temporal"]),
    )
    lines = [
        "# MOSAIC MSR-VTT Dev 结果",
        "",
        "本报告仅使用冻结的 1,000-video Dev；`test_accessed=false`，不得当作 Test 结论。",
    ]
    for direction, title in (("text_to_video", "Text-to-video"), ("video_to_text", "Video-to-text")):
        lines.extend(["", f"## {title}", "", "| Model | R@1 | R@5 | R@10 | MRR |", "|---|---:|---:|---:|---:|"])
        for label, result in models:
            metric = result[direction]
            recall = metric["recall_at"]
            lines.append(
                f"| {label} | {recall['1']:.4f} | {recall['5']:.4f} | "
                f"{recall['10']:.4f} | {metric['mrr']:.4f} |"
            )
    paired = suite["trained_temporal"]["paired_video_cluster_bootstrap_vs_frozen_mean"]
    lines.extend(["", "## Trained − frozen mean paired bootstrap", "", "| Direction / metric | Delta | 95% CI |", "|---|---:|---:|"])
    for direction in ("text_to_video", "video_to_text"):
        for metric in ("recall@1", "recall@10", "mrr"):
            value = paired[direction][metric]
            lines.append(
                f"| {direction} {metric} | {value['delta']:+.4f} | "
                f"[{value['lower']:+.4f}, {value['upper']:+.4f}] |"
            )
    lines.extend(
        [
            "",
            "T2V 的本次 Dev 差值 CI 为正；V2T 的列示差值 CI 跨 0。它们只用于冻结模型，",
            "不能提前写成 1K-A Test 结果、SOTA 或线上收益。",
            "",
        ]
    )
    return "\n".join(lines)


def _compact_result(result: dict[str, object]) -> dict[str, object]:
    ranks = result.pop("ranks")
    result["rank_sha256"] = {
        direction: hashlib.sha256(
            bytes().join(int(value).to_bytes(8, "little", signed=True) for value in values)
        ).hexdigest()
        for direction, values in ranks.items()
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MSR-VTT Train/Dev without touching Test")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/msrvtt_1ka_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/mosaic_msrvtt_dev_v1.json"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/mosaic_msrvtt_dev_v1.md"))
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training_summary = json.loads(
        (args.checkpoint_dir / "training_summary.json").read_text(encoding="utf-8")
    )
    checkpoint_config = json.loads(
        (args.checkpoint_dir / "video_encoder_config.json").read_text(encoding="utf-8")
    )
    suite = {
        "schema_version": "mosaic.video_dev_suite.v1",
        "split": args.split,
        "test_accessed": False,
        "protocol": "full split catalog; all captions remain clustered by video id",
        "provenance": {
            "manifest_sha256": _sha256(args.manifest),
            "feature_npz_sha256": _sha256(args.features),
            "config_sha256": _sha256(args.config),
            "checkpoint_weights_sha256": _sha256(args.checkpoint_dir / "video_encoder.safetensors"),
            "checkpoint_config_sha256": _sha256(args.checkpoint_dir / "video_encoder_config.json"),
            "training_summary_sha256": _sha256(args.checkpoint_dir / "training_summary.json"),
        },
        "training": training_summary,
        "selected_checkpoint": checkpoint_config,
        "frozen_mean": _compact_result(strip_internal(
            evaluate_video_model(
                args.manifest,
                args.features,
                config,
                aggregator="mean",
                split=args.split,
            )
        )),
        "frozen_max": _compact_result(strip_internal(
            evaluate_video_model(
                args.manifest,
                args.features,
                config,
                aggregator="max",
                split=args.split,
            )
        )),
        "trained_temporal": _compact_result(strip_internal(
            evaluate_video_model(
                args.manifest,
                args.features,
                config,
                checkpoint_dir=args.checkpoint_dir,
                device=args.device,
                split=args.split,
            )
        )),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(suite), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "markdown": str(args.markdown.resolve()),
                "split": args.split,
                "frozen_mean": suite["frozen_mean"]["text_to_video"],
                "trained_temporal": suite["trained_temporal"]["text_to_video"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
