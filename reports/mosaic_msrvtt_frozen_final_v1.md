# MOSAIC MSR-VTT-1K-A Frozen Final

- Input commit: `4cf63f01a0bb81e6043535c470203177cafc5ea6`
- Videos / query captions: `1000 / 1000`
- Protocol: JSFusion 1K-A, one official query caption per video.

## Text-to-video

| Model | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| Frozen CLIP mean | 0.3040 | 0.5240 | 0.6310 | 0.4094 |
| Frozen CLIP max | 0.1970 | 0.3990 | 0.5060 | 0.2993 |
| MOSAIC temporal | 0.3350 | 0.6080 | 0.7100 | 0.4579 |

## Video-to-text

| Model | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| Frozen CLIP mean | 0.2700 | 0.5100 | 0.6100 | 0.3834 |
| Frozen CLIP max | 0.1740 | 0.3810 | 0.4790 | 0.2762 |
| MOSAIC temporal | 0.3110 | 0.5960 | 0.7080 | 0.4453 |

## Paired video-cluster bootstrap: trained minus frozen mean

| Direction / metric | Delta | 95% CI | Interpretation |
|---|---:|---:|---|
| text_to_video recall@1 | +0.0310 | [+0.0070, +0.0550] | positive under this protocol |
| text_to_video recall@10 | +0.0790 | [+0.0560, +0.1040] | positive under this protocol |
| text_to_video mrr | +0.0485 | [+0.0311, +0.0662] | positive under this protocol |
| video_to_text recall@1 | +0.0410 | [+0.0140, +0.0680] | positive under this protocol |
| video_to_text recall@10 | +0.0980 | [+0.0730, +0.1240] | positive under this protocol |
| video_to_text mrr | +0.0619 | [+0.0413, +0.0834] | positive under this protocol |

## Boundaries

This is a single controlled offline evaluation on a public MSR-VTT mirror. It does not establish SOTA, online lift, production latency, audio/ASR/OCR quality, or independence from CLIP pretraining data. The one-caption 1K-A protocol is not interchangeable with evaluations using all test captions.
