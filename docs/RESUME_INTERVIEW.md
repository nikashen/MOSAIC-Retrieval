# 简历与面试表述

## 可以使用的简历版本

> **MOSAIC-Retrieval｜多模态检索与冷启动内容编码**：在 COCO 上构建图像级严格
> Train/Dev 与未参与本项目 adapter/reranker 训练和选模的 1k External Final；冻结 CLIP，训练 131 万参数 residual
> adapter、模态门控与 hard-negative 对比目标；interaction reranker 在 Train mined
> negatives 上训练，并在 Dev 选择 epoch 与 alpha（reranker 另含约 52.9 万参数；
> 两部分合计 1,844,998 个可训练参数）。
> External Final 的 **image-only full-catalog text-to-image** R@1/R@10 相对
> zero-shot 分别提升 2.89/1.36 个
> 百分点（1,000 次 image-cluster paired bootstrap 95% CI 均为正）；导出 FAISS/FastAPI
> 内容检索服务和安全的 TRACE-Rec 冷启动向量契约。

视频方向可追加一句：

> 扩展 MSR-VTT-1K-A 视频检索，完成 10,000 MP4 CRC 门禁、12 帧流式 CLIP 与
> Dev-only temporal-attention 选模；一次性 one-caption Final 上 T2V R@1/R@10
> `0.304/0.631 → 0.335/0.710`，V2T `0.270/0.610 → 0.311/0.708`，四项
> paired video-cluster bootstrap 95% CI 均为正。

若被追问算法归因，可补充：三种子 post-Final Dev-only 实验中，temporal T2V R@1/R@10
`0.3633/0.7233`，参数量匹配 mean 为 `0.3547/0.7179`；R@1 三个 seed 的 paired
CI 均为正。每个 CI 是 seed 内按 video cluster bootstrap，不是跨 seed CI；
hard-negative 消融几乎不变，因此不把它写成已被证明有效的组件。

## 高频追问

**为什么冻结 CLIP？**

RTX 3050 Ti 4GB 不适合把全参数微调伪装成真实实验。冻结主干、训练小 adapter 是
可复现的算力约束设计；External Final 在该冻结公开数据协议下观察到正向差异，
paired CI 不跨 0。公开 COCO 是否进入 CLIP 预训练未知，因此不把它写成泛化证明。

**怎样避免 caption 泄漏？**

切分单位是 image，不是 caption；融合评测对每条 query caption 采用 leave-one-caption-out
metadata，因此 query 文本不直接进入其 target item vector。

**为什么要说 reranker R@1 不显著？**

点估计增益不等于稳健增益。它的 R@10 CI 为正，但 R@1/MRR CI 跨 0；面试中应明确
区分两个结论。

**项目四真的接上了吗？**

目前完成的是 schema 和安全 exporter，而非伪造跨数据集业务效果。真实接入还需要
同一内容 catalog 的 immutable ID mapping、把当前视频 encoder 应用于该 catalog，
以及项目四 as-of 再训练；MSR-VTT 实验本身不等于 KuaiRec 线上接入。

**为什么不能和论文里的任意 MSR-VTT 数字直接横比？**

这里固定的是 JSFusion 1K-A、每视频一条官方 query 的双向一对一口径；有些工作使用
每视频 20 条 captions、不同 split 或额外预训练。必须先对齐协议，且公开 MSR-VTT
可能被 CLIP 预训练见过，所以只陈述本冻结口径下相对 mean-pool 的差值与 CI。
