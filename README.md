# MOSAIC-Retrieval

[在线证据工作台](https://nikashen.github.io/MOSAIC-Retrieval/) | [GitHub 仓库](https://github.com/nikashen/MOSAIC-Retrieval) | [结果与边界](docs/RESULTS.md)

## 多模态图像/视频—文本检索与冷启动内容编码器

MOSAIC（**M**odality-aware **O**ptimization with **S**afe **A**daptive
**I**nteraction and **C**ross-modal retrieval）是项目五的真实多模态算法项目。
第一版使用 COCO 2017 validation 的 5,000 张图像和 25,014 条人工 captions，
构建一个可复现、可审计的图像—文本检索基准，并把融合后的内容向量作为项目四
TRACE-Rec 的冷启动 item feature。

`0.2.0` 在同一审计框架下增加真实 MSR-VTT-1K-A 视频—文本检索：10,000 个
MP4 全量 size/CRC 门禁、12 帧流式 CLIP、mean/max/temporal-attention 对照、
Dev-only checkpoint 选择与 video-cluster paired bootstrap。

`0.2.1` 追加不读取 Test 的三种子归因消融，区分 projection、参数量匹配 mean、
temporal attention、hard-negative 与 teacher preservation，不改写 0.2.0 Final。

`0.2.2` 修复 FastAPI app 的 per-instance 日志 handler 生命周期，避免 Windows 测试
临时日志文件被悬挂句柄锁定；该补丁只影响 serving 资源清理，不读取或改写任何 Final。

## 当前正式结果

下列短提交号属于未公开的原实验历史，用于说明冻结顺序，并不能在本公共单根仓内解引用；公开核验以随仓报告 SHA-256、audit 绑定和 evaluation digest 为准。“一次性 Final”表示按冻结协议与审计记录运行一次，不构成对仓库外行为的密码学证明。

冻结输入提交 `a19d646` 后，在完全未参与 adapter 训练/Dev 选择的 COCO
train2017 External Final（1,000 images / 5,003 captions）只读取一次：

| Text-to-image | R@1 | R@10 | MRR |
|---|---:|---:|---:|
| Zero-shot CLIP | 0.5019 | 0.8931 | 0.6351 |
| MOSAIC adapter | 0.5309 | 0.9067 | 0.6580 |
| + interaction reranker | 0.5409 | 0.9196 | 0.6682 |

Adapter 相对 zero-shot 的 R@1/R@10 paired image-cluster bootstrap 95% CI 分别为
`[+0.0200,+0.0392]` 与 `[+0.0078,+0.0196]`。Reranker 的 R@10 CI 为正，
但 R@1/MRR CI 跨 0，不宣称显著。完整协议和结果见
[实验协议](docs/EXPERIMENT_PROTOCOL.md) 与 [结果摘要](docs/RESULTS.md)。

### MSR-VTT-1K-A Frozen Final

核心训练提交 `3feffb2` 完成 8,000 Train / 1,000 Dev 选模，证据提交
`4cf63f0` 后才生成 Test 特征并运行一次性 Final。正式 one-caption 1K-A 结果：

| Final model | T2V R@1 | T2V R@10 | V2T R@1 | V2T R@10 |
|---|---:|---:|---:|---:|
| Frozen CLIP mean | 0.3040 | 0.6310 | 0.2700 | 0.6100 |
| Frozen CLIP max | 0.1970 | 0.5060 | 0.1740 | 0.4790 |
| MOSAIC temporal | 0.3350 | 0.7100 | 0.3110 | 0.7080 |

Temporal − mean 的 T2V R@1/R@10 为 `+0.031/+0.079`，95% CI 分别为
`[+0.007,+0.055] / [+0.056,+0.104]`；V2T R@1/R@10 为
`+0.041/+0.098`，CI 为 `[+0.014,+0.068] / [+0.073,+0.124]`。这些结论只
适用于当前固定数据、模型和 one-caption 协议。详见
[正式报告](reports/mosaic_msrvtt_frozen_final_v1.md)、
[机器事实源](reports/mosaic_msrvtt_frozen_final_v1.json)、
[audit](reports/mosaic_msrvtt_frozen_final_v1.audit.json) 与
[Dev 选择证据](reports/mosaic_msrvtt_dev_v1.md)。

### MSR-VTT Dev-only 归因结论

三 seed 下，temporal 的 T2V R@1/R@10 为 `0.3633±0.0004 / 0.7233±0.0013`；
参数量匹配 mean 为 `0.3547±0.0002 / 0.7179±0.0010`。Temporal 的 R@1 在三个
seed 的 paired CI 均为正；R@10 在两个 seed 为正、一个跨 0，支持较小但稳定的
时序路由贡献。

去掉 hard-negative 后结果几乎不变，说明它没有获得独立经验支持；去掉 teacher
使 T2V R@10 在三个 seed 均小幅下降。此结论仅来自 post-Final Dev diagnostic，
不改变正式 checkpoint/Final。详见
[归因消融](reports/mosaic_msrvtt_dev_attribution_v1.md)。

它不是“调用 CLIP 得到一个向量”的页面：

1. 主干 CLIP 冻结，训练可审计的双塔投影、缺失模态门控和 modality-dropout；
2. 对称 InfoNCE 使用 batch 内负样本，并对相似但错误的 hard negatives 加 margin；
3. 训练后使用独立 dev 选择的 pair-interaction reranker，不在 Test 上调参；
4. image→text 和 text→image 均在整库上检索，按 image cluster 做 1,000 次 paired
   bootstrap 置信区间；
5. 报告同时给出 full、image-only、text-only、zero-shot CLIP 和 trainable
   fusion 的结果，避免把“缺模态时的退化”隐藏掉；
6. 生成安全的 `npz`/JSON artifact、FAISS 索引和 FastAPI 演示服务，服务输出
   模型、数据、checkpoint 和 evidence hash。

视频扩展另外实现 12 帧中点抽样、可恢复/并行 FFmpeg 解码、epoch-0 精确
mean-pool fallback、确定性每视频 caption 采样、双向全库检索与 video-cluster
bootstrap；Test finalizer 在读取派生 Test 特征计算指标前原子占用 audit 文件名。

### 当前版本的边界

COCO 与 MSR-VTT 是两条独立证据链，图像结果不会改写成视频结果。视频扩展只用
视觉帧与 caption，不含音频、ASR 或 OCR；官方 one-caption 1K-A 也不能与
all-caption 评测互换。两条链路都只是公开数据离线实验，不证明线上推荐、生产
SLA 或 SOTA。`integrations/project4_content_vector.py` 仍只是冷启动向量契约，
没有把 MSR-VTT/COCO id 冒充为 KuaiRec `video_id`。

## 算法链路

```text
image --------------------> frozen CLIP image tower --+
                                                       |
caption/query ------------> frozen CLIP text tower ----+--> trainable projections
                                                       |       + hard-negative loss
missing-modality mask ----- modality reliability gate -+       + interaction rerank
                                                               |
                                                               v
                                                       FAISS exact/ANN retrieval
                                                               |
                                                               v
                                           project-4 cold-start content_vector
```

门控输出满足：

```text
z_item = normalize(g_img * p_img(image) + g_txt * p_txt(caption))
g_img + g_txt = 1,  g_m absent = 0
```

训练时随机屏蔽一种模态，并把 mask 显式输入 gate；因此 `image-only`、
`text-only` 和 `full` 是预先声明的评测协议，而不是演示时临时改输入。

## 运行环境

实测环境为 Windows + Python 3.10 + PyTorch 2.6.0 CUDA 11.8 + RTX 3050 Ti
4GB。默认冻结 CLIP、224px、4GB 显存安全配置，不需要全参数微调。

建议把 HuggingFace 缓存放在 F 盘，并使用镜像（网络受限时）：

```powershell
$env:HF_HOME = "$HOME\.cache\huggingface"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:PYTHONPATH = "src"
```

## 从零开始

```powershell
Set-Location "MOSAIC-Retrieval"
$env:SHORTREC_PYTHON = ".\.venv\Scripts\python.exe"
.\run_project.ps1 data
.\run_project.ps1 features --limit 5000
.\run_project.ps1 train
.\run_project.ps1 reranker
.\run_project.ps1 evaluate
.\run_project.ps1 index
.\run_project.ps1 serve
```

首次下载的文件约 1.1GB（COCO val images + annotations）；CLIP checkpoint
约 600MB。下载脚本支持断点续传和 MD5 校验。

视频 Train/Dev 链路：

```powershell
.\run_project.ps1 video-data
.\run_project.ps1 video-features-train -Device cuda
.\run_project.ps1 video-train -Device cuda
.\run_project.ps1 video-dev -Device cuda
.\run_project.ps1 video-ablation -Device cuda
```

视频下载约 2.2GB ZIP / 2.2GB 解压文件；derived Train/Dev feature NPZ 约
261MB。只有代码、配置和 Dev-selected checkpoint 证据提交且 tracked worktree
干净后，才允许执行 `video-features-test` 和一次性 `video-final`。

低内存 smoke：

```powershell
.\run_project.ps1 smoke
```

它使用仓库内生成的 12 行确定性 toy manifest，并执行完整单元测试；不会把 toy
指标写入正式报告。

## 评测输出

正式输出位于：

```text
reports/mosaic_external_final_v1.json  # COCO 正式机器事实源
reports/mosaic_external_final_v1.md    # COCO 正式阅读版
reports/mosaic_msrvtt_dev_v1.*         # 视频 Dev-only 选择证据
reports/mosaic_msrvtt_frozen_final_v1.* # 视频正式事实源/阅读版/audit
reports/mosaic_msrvtt_dev_attribution_v1.* # 视频三种子Dev-only归因
reports/mosaic_release_0.2.0.*           # wheel/测试/部署结构核验
reports/mosaic_release_0.2.1.*           # 归因版wheel/测试/边界核验
reports/mosaic_release_0.2.2.*           # Windows日志生命周期补丁核验
artifacts/mosaic_coco5k_v1/            # 本地 checkpoint/index（Git 忽略）
artifacts/mosaic_msrvtt_1ka_v1/        # 本地视频 feature/checkpoint（Git 忽略）
```

报告至少包含：

- image→text/text→image Recall@1/5/10/50、MRR、median rank；
- full/image-only/text-only 的 paired bootstrap 95% CI；
- zero-shot、residual projection、gated fusion、fusion+interaction-reranker 对照；
- hard-negative 和 modality-dropout 消融；
- 参数量、显存、吞吐、p50/p95 检索延迟；
- 数据切分、模型 commit、输入提交、SHA-256 和明确的非结论。

## 与项目四的接口

离线生成：

```powershell
.\run_project.ps1 export-project4
```

输出契约是：

```text
video_id (int64) | content_vector (float32, L2-normalized)
modality_mask (uint8) | encoder_version | source_sha256
```

项目四的历史 KuaiRec item 不会被伪装成 COCO 视频；接入脚本只在双方明确
`content_id -> video_id` 映射文件存在时生成映射，否则安全失败。

## 双轨演示工作台

GitHub Pages 是不依赖模型或数据下载的**聚合证据浏览器**，使用代码绘制的合成检索卡片说明交互，不展示或再分发 COCO/MSR-VTT 媒体，也不冒充真实逐 Query 排名。真实 FastAPI/FAISS 检索服务需要用户自行提供有权使用的数据与本地 artifact。

运行：

```powershell
.\run_project.ps1 serve
```

打开 <http://127.0.0.1:8050/>。页面保留两条互不混写的展示轨道：

- `COCO 图文检索` 调用冻结 CLIP + adapter 编码文本，并对 5,000 张本地图像执行真实 FAISS 全库检索。
- `MSR-VTT 视频证据` 读取 one-caption 1K-A Frozen Final 与 audit，展示聚合 T2V/V2T 指标和 paired video-cluster CI；在用户提供本地授权数据及未跟踪 allowlist 时，服务才会播放对应样本。

视频报告只发布聚合指标与排名 digest，没有发布逐 Query 排名行。因此视频页明确不提供文本检索框，也不会从代表视频中拼造“Top-K”。Final JSON、evaluation digest 或 audit report SHA 任一不匹配时，指标接口 fail closed。面试顺序与口径见 [RESUME_INTERVIEW.md](docs/RESUME_INTERVIEW.md)。

## 简历表述（仅在正式报告生成后使用）

> MOSAIC-Retrieval：面向冷启动推荐的模态可靠性门控图文双塔。基于 COCO-5K
> 构建 image/text strict cluster split，冻结 CLIP 主干并训练 projection +
> modality dropout/gating，加入 in-batch hard-negative contrastive loss 与
> Train-negative pair-interaction reranker（Dev 选 epoch/alpha）；在 **image-only
> T2I full-catalog** 检索上报告 adapter 增量及 Recall@K、
> MRR 和 1,000 次 user/image-cluster bootstrap CI，并导出 FAISS/HTTP 内容向量
> 服务接入 TRACE-Rec。

> MSR-VTT-1K-A 视频检索扩展：对 10,000 个视频做全量 CRC 门禁与 12 帧流式
> frozen-CLIP 编码，训练 residual projection + temporal attention；在 Dev-only
> 选模后的一次性 1K-A Final 上，T2V R@1/R@10 从 0.304/0.631 提升至
> 0.335/0.710，V2T 从 0.270/0.610 提升至 0.311/0.708，并报告 1,000 次
> video-cluster paired bootstrap CI。

本地演示与部署方法见 [DEPLOY_OTHER_MACHINE.md](docs/DEPLOY_OTHER_MACHINE.md)，面试问答见
[RESUME_INTERVIEW.md](docs/RESUME_INTERVIEW.md)。

不要把本项目写成“线上 A/B”“SOTA”或“完成视频 ASR/OCR”，除非后续真的产生
相应的独立证据。
