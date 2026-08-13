# MOSAIC MSR-VTT Dev 结果

本报告仅使用冻结的 1,000-video Dev；`test_accessed=false`，不得当作 Test 结论。

## Text-to-video

| Model | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| Frozen CLIP mean | 0.3289 | 0.5608 | 0.6532 | 0.4380 |
| Frozen CLIP max | 0.2124 | 0.4117 | 0.5101 | 0.3114 |
| MOSAIC temporal | 0.3636 | 0.6223 | 0.7237 | 0.4842 |

## Video-to-text

| Model | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| Frozen CLIP mean | 0.5640 | 0.8030 | 0.8800 | 0.6688 |
| Frozen CLIP max | 0.3870 | 0.6540 | 0.7470 | 0.5063 |
| MOSAIC temporal | 0.5930 | 0.8240 | 0.9000 | 0.6930 |

## Trained − frozen mean paired bootstrap

| Direction / metric | Delta | 95% CI |
|---|---:|---:|
| text_to_video recall@1 | +0.0347 | [+0.0236, +0.0462] |
| text_to_video recall@10 | +0.0704 | [+0.0581, +0.0822] |
| text_to_video mrr | +0.0462 | [+0.0360, +0.0558] |
| video_to_text recall@1 | +0.0290 | [-0.0050, +0.0630] |
| video_to_text recall@10 | +0.0200 | [-0.0010, +0.0410] |
| video_to_text mrr | +0.0243 | [-0.0001, +0.0474] |

T2V 的本次 Dev 差值 CI 为正；V2T 的列示差值 CI 跨 0。它们只用于冻结模型，
不能提前写成 1K-A Test 结果、SOTA 或线上收益。
