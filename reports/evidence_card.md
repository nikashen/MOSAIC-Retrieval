# MOSAIC-Retrieval evidence card

This public snapshot separates three evidence levels:

1. **Frozen external evidence.** COCO uses 1,000 train2017 images selected after the adapter and Dev choices were frozen. MSR-VTT uses the JSFusion 1K-A one-caption protocol after an 8k Train / 1k deterministic Dev selection stage.
2. **Post-Final Dev attribution.** Three Train/Dev-only seeds compare temporal aggregation with parameter-matched mean pooling. They do not reopen Test or replace the frozen checkpoint.
3. **Synthetic public contracts.** Unit tests and Pages demonstrate implementation behaviour without downloading or reconstructing datasets, weights, features or rankings.

## Supported claims

- COCO **image-only, full-catalog text-to-image** adapter improvements over frozen CLIP: R@1 `+2.89 pp`, R@10 `+1.36 pp`; paired image-cluster bootstrap intervals are positive. This is not a fusion-gate improvement: trained full fusion did not beat zero-shot full fusion on the Final.
- MSR-VTT temporal aggregation over frozen mean pooling: T2V R@1/R@10 `+3.1/+7.9 pp`; V2T R@1/R@10 `+4.1/+9.8 pp`; paired video-cluster intervals are positive under the declared one-caption protocol.
- Temporal attention has modest post-Final Dev-only T2V R@1 diagnostic evidence beyond a parameter-matched mean projection. These seedwise intervals reuse the same Dev after checkpoint selection and resample clusters within each seed; they do not include selection, seed or multiple-comparison uncertainty. The 15 run configs and shared inputs match embedded provenance, but no separate audit or per-run checkpoint hashes were recorded, so this evidence is weaker than the Frozen Final. The hard-negative term did not receive independent support.

## Unsupported claims

Do not claim SOTA, production throughput/SLA, online recommendation or revenue lift, all-caption MSR-VTT comparability, audio/ASR/OCR capability, or a proven independent hard-negative contribution.

The published site includes only aggregate metrics and code-drawn synthetic cards. No COCO/MSR-VTT image, frame, caption, query-level ranking, model weight, feature array or index is redistributed. Before sanitization, the local private asset chain passed the complete COCO verifier and all 9 MSR-VTT input/report/digest checks. A fresh public clone verifies the preserved report/audit consistency and implementation contracts, but cannot independently recompute frozen metrics from unpublished raw ranks.
