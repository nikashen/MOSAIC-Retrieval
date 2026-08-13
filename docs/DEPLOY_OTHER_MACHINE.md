# 换电脑部署

## 最小条件

- Python 3.10–3.12；NVIDIA GPU 可选，服务 CPU 可运行。
- 约 3GB：COCO val images/annotations + CLIP checkpoint + feature/index artifacts。
- 若复现视频实验，另需约 4.4GB MSR-VTT ZIP+解压视频、约 275MB derived features
  与本地 checkpoint；镜像未声明许可证，只能在确认授权后复制/下载。
- Windows 路径含中文时，服务需从仓库根目录启动；FAISS 读取的是 ASCII 相对路径，
  已规避 Windows C++ FileIO 对 Unicode absolute path 的限制。

## 安装

```powershell
git clone <your-repository-url> MOSAIC-Retrieval
Set-Location MOSAIC-Retrieval
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[serve,dev]"
```

GPU 特征提取建议安装与驱动匹配的 CUDA PyTorch wheel；CPU 服务不需要 CUDA。

## 数据与 artifact

本仓库不会分发 COCO 图、CLIP 权重和训练 artifact。可选两种方式：

1. 合法地复制本机 `data/raw/`、`artifacts/mosaic_coco5k_v1/` 到新机相同项目根；
2. 在新机运行 `.\run_project.ps1 data`、`features`、`train`、`reranker`、`index`。

MSR-VTT 的本地 Train/Dev 重建命令是 `video-data`、`video-features-train`、
`video-train`、`video-dev`。正式 Test JSON/Markdown/audit 已随 Git 提供，可用
`video-verify` 核对本地复制的 Test feature/checkpoint 哈希；不要删除既有 audit 后
冒充首次 Final。若确需独立复验，应使用新的预注册 protocol/audit 名称并保留旧证据。

首次模型下载如官方 HuggingFace 不通：

```powershell
$env:HF_HOME = "$HOME\.cache\huggingface"
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

然后启动：

```powershell
.\run_project.ps1 serve
Invoke-RestMethod http://127.0.0.1:8050/api/health
```

Linux/macOS 使用：

```bash
chmod +x run_project.sh
./run_project.sh smoke
./run_project.sh serve
```

请遵守 COCO、CLIP 及原始 Flickr 图像许可；MSR-VTT 镜像未声明数据许可证，本项目
不再分发其视频或 captions。所有公开数据证据仅用于研究/展示。
