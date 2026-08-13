# MOSAIC 实验协议

## 目标与数据边界

主任务是图像—文本双向全库检索；冷启动诊断任务是文本 query 检索融合了图像和
非 query caption metadata 的内容向量。不是视频、ASR、OCR、线上推荐或 A/B 实验。

| 阶段 | 数据 | 用途 | 是否可用于调参 |
|---|---|---|---|
| Train | COCO val2017，3,450 images | adapter / interaction reranker 拟合 | 是 |
| Dev | COCO val2017，739 images | epoch、alpha、候选池和消融选择 | 是 |
| Internal diagnostic Test | COCO val2017，811 images | 修复前的开发诊断 | 否，不作为最终结论 |
| External Final | COCO train2017，1,000 images / 5,003 captions | 冻结后一次性最终评测 | 否 |

前三个切分按 `sha256("mosaic-coco5k-v1:image_id") % 1000`，每张图的全部
captions 永远在同一 split。External Final 从 118,287 个 train2017 图像中按
`sha256("mosaic-external-final-v1-frozen:image_id")` 排序取前 1,000 个；它们
没有用于 adapter 训练、Dev 选择或 Internal diagnostic。

公开 COCO 可能已包含在 CLIP 的互联网预训练语料中，因此结论是“在冻结 CLIP
上的小样本适配增益”，不是对从零训练视觉语言模型的泛化证明。

## 模型与选择规则

1. `openai/clip-vit-base-patch32` 全部冻结。
2. 只训练 1,316,101 个 residual projection/gate 参数；投影最后层零初始化，
   epoch 0 恰为 CLIP identity baseline。
3. 损失为对称 InfoNCE + 正确方向的 batch hard-negative margin + teacher
   preservation + modality dropout fusion loss。
4. checkpoint 仅按 Dev 的
   `mean(T2I R@1, T2I R@10, I2T R@1, I2T R@10)` 选择；任一方向 R@10 比
   zero-shot Dev 低超过 0.002 时拒绝。
5. interaction reranker 仅在 Train mined hard negatives 上训练，Dev 从
   `{0, .02, .05, .1, .2, .5}` 选择组合系数，候选池固定为 top-50。

## 指标和不确定性

- Text-to-image 与 image-to-text：full-catalog Recall@1/5/10/50、MRR、mean/
  median rank；同分按稳定原始索引排序。
- 冷启动 fusion：每一个 query caption 都从目标 item metadata 中排除；因此没有
  把 query 字符串逐字放入答案向量。
- 所有差值按 image id cluster 的 1,000 次 paired percentile bootstrap 报 95% CI。
- CI 跨 0 表示此数据/协议下不能声称差异显著，哪怕点估计为正。

## External Final 审计

历史 commit 属于未公开实验仓；公共快照通过报告哈希和 audit 绑定验证制品。互斥审计名称记录该协议下的一次 finalizer，不证明仓库外不存在手工查看或另名运行。

`scripts/publish_report.py` 会在读取指标之前以 `O_EXCL` 原子创建 audit 文件；
已有 audit/report 时拒绝第二次正式运行。最终 audit 位于：

```text
reports/mosaic_external_final_v1.audit.json
```

输入 commit 为 `a19d646`。报告之后仅做过一次 Dev ablation 元数据更正；audit
保存了更正前后 hash，并证明 `metrics + statistics` 的 SHA-256 没有改变，未重读
External Final。
