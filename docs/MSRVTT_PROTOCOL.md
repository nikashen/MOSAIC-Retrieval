# MOSAIC MSR-VTT-1K-A 视频检索协议

## 数据与许可边界

本扩展使用 `friedrichor/MSR-VTT` 镜像提交
`c1af215a96934854f42683c19c51391aaee6f962`。下载器固定三个文件的字节数和
SHA-256，并逐个核对 ZIP 与 10,000 个落盘 MP4 的 size/CRC32。该镜像没有声明
可再分发的数据许可证，因此视频仅用于本地学术/非商业评测，不进入 Git、wheel、
Docker 或发布资产。

| Split | Videos | Captions used | Purpose |
|---|---:|---:|---|
| Train | 8,000 | 原始每视频 20 条；每 epoch 确定性采 1 条 | 时序 encoder 训练 |
| Dev | 1,000 | 每视频 20 条 | epoch/门禁选择与开发诊断 |
| Frozen Test | 1,000 | JSFusion 1K-A 每视频 1 条官方 query | 一次性最终评测 |

Train/Dev 从官方 9K Train 中按
`sha256("mosaic-msrvtt-dev-v1:video_id")` 最小的 1,000 个视频冻结 Dev。三个
split 的 video id 完全不交叉。Test 的 manifest 可以在数据准备阶段校验结构，
但 Test 特征、相似度、rank 和指标均不得用于模型、epoch 或超参数选择。

## 特征与模型

- 冻结 `openai/clip-vit-base-patch32` revision
  `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`；本机权重 SHA-256 为
  `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`。
- 每个视频按声明时长的 12 个等宽区间中点取帧；FFmpeg 流式解码，不缓存原始帧。
- 基线为 L2-normalized frozen CLIP frame mean；另报告 frozen elementwise max
  诊断。
- Trainable 模型为 residual video/text projection + 带位置嵌入的 temporal
  attention。attention scorer 与 residual 输出在 epoch 0 精确退化为 mean-pool
  CLIP，因此无合格 Dev checkpoint 时保留真实基线，而不是随机模型。
- 每个 Train 视频每 epoch 用
  `sha256(seed:epoch:video_id)` 选择一条 caption，避免 20-caption 视频在 batch 中
  形成假负样本；损失为对称 InfoNCE、batch hard-negative margin 与 teacher
  preservation。

## Dev 选择门禁

唯一选择分数是

```text
mean(T2V R@1, T2V R@10, V2T R@1, V2T R@10)
```

候选 checkpoint 还必须满足 T2V 和 V2T 的 R@10 均不比 frozen mean Dev 基线低
超过 0.002。只有严格高于 baseline score 的候选会替换 epoch-0 checkpoint。
Frozen max 是诊断对照，不参与选择。Test 不参与消融、epoch、聚合器或阈值决策。

## Frozen Test 与统计

正式 Final 必须在代码、配置和 Dev-selected checkpoint 提交且 tracked worktree
干净后运行。Test 特征只允许在此冻结之后生成；正式指标由带 `O_EXCL` audit 的
finalizer 每个 audit 文件名运行一次。

报告同时给出 frozen mean、frozen max、MOSAIC temporal 的双向全库
Recall@1/5/10/50、MRR、mean/median rank，并以 video id 为 cluster 做 1,000 次
paired bootstrap，比较 trained minus frozen mean。CI 跨 0 时不声称方向性差异。

该 one-caption 1K-A 口径不能与使用全部 Test captions 的口径互换。结果也不能
外推为 SOTA、线上推荐收益、生产 SLA、ASR/OCR/音频能力；公开 MSR-VTT 是否被
CLIP 预训练见过未知。

## 已完成证据

以下 commit 是未公开原实验历史的冻结锚点；公共单根仓以随仓报告 SHA-256、evaluation digest 和 audit 绑定为可核验边界。这里的“一次性”仅指按声明协议记录的一次 formal finalizer。

- Train/Dev core commit：`3feffb2f1ac6c3d6cf77bfbd567fe2cc8992b83e`
- Frozen Final input commit：`4cf63f01a0bb81e6043535c470203177cafc5ea6`
- Test feature SHA-256：`701ea32f5718ca818d3e90f761629a21ed5386ea0d64e6d4a694ae3cc2a146eb`
- Evaluation digest：`d2db1ff29fe78782519cbd2745d40c4f05ff855fc3e27bc94dff275fd32692e6`
- Post-run verifier：9/9 input/report/digest checks PASS

结果解释以不可改写的 Final JSON/audit 为准；本节仅提供导航。

## Post-Final Dev attribution（0.2.1）

为回答“增益来自 temporal attention 还是更多参数/投影头”，另做 3 seeds、
Train/Dev-only 的诊断；它不读取 Test、不选择新 checkpoint、不修改 Final。对照包括
projection mean、667,275 参数的 capacity-matched mean、666,628 参数 temporal、
去 hard-negative、去 teacher。正式报告位于
`reports/mosaic_msrvtt_dev_attribution_v1.*`。

结果支持 temporal 对 T2V R@1 的小幅稳定贡献；hard-negative 未显示独立贡献；
teacher 对 R@10 有小幅一致帮助。所有表述必须保留 post-Final Dev diagnostic 边界。
