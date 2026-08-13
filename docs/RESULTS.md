# External Final 结果摘要

正式机器可读事实源：[mosaic_external_final_v1.json](../reports/mosaic_external_final_v1.json)。

数据：COCO train2017 外部 1,000 图像、5,003 captions；adapter 未见这些图像。

| Text-to-image model | R@1 | R@10 | MRR |
|---|---:|---:|---:|
| Zero-shot CLIP | 0.5019 | 0.8931 | 0.6351 |
| MOSAIC residual adapter | 0.5309 | 0.9067 | 0.6580 |
| + interaction reranker | 0.5409 | 0.9196 | 0.6682 |

| 差值 | 点估计 | 95% paired image-cluster CI | 结论 |
|---|---:|---:|---|
| Adapter − zero-shot, T2I R@1 | +0.0289 | [+0.0200, +0.0392] | 支持正向增益 |
| Adapter − zero-shot, T2I R@10 | +0.0136 | [+0.0078, +0.0196] | 支持正向增益 |
| Adapter − zero-shot, T2I MRR | +0.0229 | [+0.0165, +0.0294] | 支持正向增益 |
| Reranker − adapter, T2I R@10 | +0.0130 | [+0.0038, +0.0232] | 支持正向增益 |
| Reranker − adapter, T2I R@1 | +0.0098 | [−0.0068, +0.0251] | 不宣称显著 |

Image-to-text 的点估计也略升（R@10 0.962 → 0.970），但其 paired CI 跨 0，
因此不把它包装为显著结论。

冷启动 full fusion 相对 image-only 的留一 caption 测试有明显正向差值；该结果仅
表明在 COCO caption metadata 条件下多模态内容向量更强，不能外推为真实短视频
线上冷启动收益。

本机 FAISS 5,000 × 512 exact-index、Top-10、单进程 1,000 次 in-process vector
microbenchmark：p50 0.360ms、p95 0.480ms、顺序吞吐约 2,629 QPS。它不包含文本
CLIP 编码、HTTP、并发或跨机网络，因此不是生产 SLA。

## MSR-VTT-1K-A Frozen Final

视频扩展的正式事实源是
[mosaic_msrvtt_frozen_final_v1.json](../reports/mosaic_msrvtt_frozen_final_v1.json)，
audit 是
[mosaic_msrvtt_frozen_final_v1.audit.json](../reports/mosaic_msrvtt_frozen_final_v1.audit.json)。
训练只使用 8,000 Train / 1,000 Dev；Final 为 1,000 videos / 1,000 official
JSFusion queries。

| Model | T2V R@1 | T2V R@10 | V2T R@1 | V2T R@10 |
|---|---:|---:|---:|---:|
| Frozen CLIP mean | 0.304 | 0.631 | 0.270 | 0.610 |
| Frozen CLIP max | 0.197 | 0.506 | 0.174 | 0.479 |
| MOSAIC temporal | 0.335 | 0.710 | 0.311 | 0.708 |

Temporal 相对 mean 的 T2V R@1/R@10 paired 95% CI 为
`[+0.007,+0.055] / [+0.056,+0.104]`；V2T 对应 CI 为
`[+0.014,+0.068] / [+0.073,+0.124]`。这支持当前协议下的正向差异，但不能写成
SOTA、线上收益或 all-caption MSR-VTT 指标。模型不使用音频、ASR 或 OCR；公开
MSR-VTT 是否出现在 CLIP 预训练中未知。

### Post-Final Dev-only attribution

归因实验使用 3 seeds、只读取原 Train/Dev feature，不打开 Test，也不改变正式模型：

| Variant | Parameters | T2V R@1 mean±sd | T2V R@10 mean±sd |
|---|---:|---:|---:|
| Projection mean | 527,875 | 0.3461±0.0114 | 0.7010±0.0237 |
| Capacity-matched mean | 667,275 | 0.3547±0.0002 | 0.7179±0.0010 |
| Temporal full | 666,628 | 0.3633±0.0004 | 0.7233±0.0013 |
| Temporal without hard-negative | 666,628 | 0.3631±0.0004 | 0.7232±0.0013 |
| Temporal without teacher | 666,628 | 0.3619±0.0010 | 0.7216±0.0009 |

Temporal 相对容量匹配 mean 的 T2V R@1 三个 seed paired CI 均为正；hard-negative
没有稳定独立贡献，teacher preservation 只有小幅正向证据。因此简历可归因于
temporal aggregation，但不能继续声称 hard-negative 已被消融证明有效。
