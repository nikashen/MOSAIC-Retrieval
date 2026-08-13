from __future__ import annotations

import copy
import json
from pathlib import Path

from mosaic.experiment import train_adapter


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / "configs" / "coco5k_v1.json").read_text(encoding="utf-8"))
    variants = {
        "full": {},
        "no_hard_negative": {"hard_negative_weight": 0.0},
        "no_modality_dropout": {"modality_dropout": 0.0},
        "no_teacher_preservation": {"teacher_preservation_weight": 0.0},
    }
    results = {}
    for name, overrides in variants.items():
        if name == "full":
            results[name] = json.loads(
                (root / "artifacts" / "mosaic_coco5k_v1" / "training_summary.json").read_text(encoding="utf-8")
            )
            continue
        config = copy.deepcopy(base)
        config["model"].update(overrides)
        results[name] = train_adapter(
            root / "data" / "processed" / "coco_manifest.json",
            root / "artifacts" / "mosaic_coco5k_v1" / "clip_features.npz",
            root / "artifacts" / "ablations" / name,
            config,
            device="cuda",
        )
    report = {
        "schema_version": "mosaic.dev_ablation.v1",
        "scope": "internal_dev_only_not_external_final",
        "selection_metric": "joint bidirectional R@1/R@10 mean with directional R@10 gates",
        "variants": {
            name: {
                "best_epoch": result["best_epoch"],
                "baseline_score": result["selection"]["baseline_score"],
                "best_score": result["selection"]["best_score"],
                "delta": result["selection"]["best_score"] - result["selection"]["baseline_score"],
            }
            for name, result in results.items()
        },
    }
    output = root / "reports" / "mosaic_dev_ablation_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

