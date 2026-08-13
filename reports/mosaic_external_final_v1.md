# MOSAIC-Retrieval External Final

## Protocol

- Dataset: COCO-2017-train-external-final-1k
- Images/captions: 1000 / 5003
- Split: `external_final`; these images were not used for this project's adapter/reranker training or Dev selection.
- Pretrained CLIP may have seen public COCO during internet-scale pretraining; this limitation is explicit.

## Main full-catalog results by retrieval view

The Text→Image columns below use the **image-only item view**. They measure the residual adapter over frozen CLIP image embeddings; they are not a fusion/gating gain. Image→Text uses an image query against the caption catalog.

| Model | Image-only Text→Image R@1 | R@10 | MRR | Image-query Image→Text R@1 | R@10 |
|---|---:|---:|---:|---:|---:|
| Zero-shot CLIP | 0.5019 | 0.8931 | 0.6351 | 0.6980 | 0.9620 |
| MOSAIC residual adapter | 0.5309 | 0.9067 | 0.6580 | 0.7020 | 0.9700 |
| Adapter + interaction reranker | 0.5409 | 0.9196 | 0.6682 | — | — |

Paired image-cluster bootstrap, adapter minus zero-shot on the image-only Text→Image view:

- Text→Image R@1: +0.0289, 95% CI [+0.0200, +0.0392]
- Text→Image R@10: +0.0136, 95% CI [+0.0078, +0.0196]
- Text→Image MRR: +0.0229, 95% CI [+0.0165, +0.0294]

## Modality diagnostic

The leave-one-caption-out cold-start protocol never inserts a query caption verbatim into its own item metadata. Full, image-only and text-only results are all retained in the JSON fact source. Modality dropout did not improve the standard dual-encoder Dev score in the ablation; no unsupported robustness gain is claimed.

## Claim boundary

This is offline public-data evidence. It is not evidence of SOTA, video/ASR/OCR completion, production traffic, revenue lift, or an online A/B test.

Input commit: `a19d646e724f0a97657ceb333453d163f44a7493`

Public wording correction: the original frozen Markdown used a generic Text→Image heading. This revision makes the already-recorded `image_only` JSON scope explicit; no metric or statistic was changed or reread. The historical audit record was not edited; [the public correction record](mosaic_external_final_v1.public_wording_correction.json) binds the original and corrected Markdown hashes.
