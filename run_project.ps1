param(
    [Parameter(Position = 0)]
    [ValidateSet("data", "features", "train", "reranker", "evaluate", "index", "all", "smoke", "serve", "export-project4", "verify", "deploy-verify", "video-data", "video-features-train", "video-train", "video-dev", "video-ablation", "video-features-test", "video-final", "video-verify")]
    [string]$Action = "smoke",
    [int]$Limit = 0,
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$candidates = @()
if ($env:SHORTREC_PYTHON) { $candidates += $env:SHORTREC_PYTHON }
$candidates += @("$PSScriptRoot\.venv\Scripts\python.exe", "python")
$python = $null
foreach ($candidate in $candidates) {
    if ((Test-Path -LiteralPath $candidate) -or ($candidate -eq "python" -and (Get-Command python -ErrorAction SilentlyContinue))) {
        & $candidate -B -c "import sys, importlib.util, torch, numpy, PIL; assert (3,10) <= sys.version_info[:2] < (3,13); assert importlib.util.find_spec('transformers'); assert importlib.util.find_spec('imageio_ffmpeg')" 2>$null
        if ($LASTEXITCODE -eq 0) { $python = $candidate; break }
    }
}
if (-not $python) { throw "No compatible Python 3.10+ environment found. Set SHORTREC_PYTHON." }
$env:PYTHONPATH = "$PSScriptRoot\src"
$env:PYTHONIOENCODING = "utf-8"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
if (-not $env:HF_HOME) { $env:HF_HOME = "$HOME\.cache\huggingface" }
if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = "https://hf-mirror.com" }
if (-not $env:MOSAIC_LOG_DIR) { $env:MOSAIC_LOG_DIR = "$PSScriptRoot\logs" }
$manifest = "$PSScriptRoot\data\processed\coco_manifest.json"
$features = "$PSScriptRoot\artifacts\mosaic_coco5k_v1\clip_features.npz"
$checkpoint = "$PSScriptRoot\artifacts\mosaic_coco5k_v1"
$videoConfig = "$PSScriptRoot\configs\msrvtt_1ka_v1.json"
$videoTrainManifest = "$PSScriptRoot\data\processed\msrvtt_train_dev_v1.json"
$videoTestManifest = "$PSScriptRoot\data\processed\msrvtt_test_1ka_v1.json"
$videoArtifact = "$PSScriptRoot\artifacts\mosaic_msrvtt_1ka_v1"
$videoTrainFeatures = "$videoArtifact\train_dev_clip_features.npz"
$videoTestFeatures = "$videoArtifact\test_clip_features.npz"
$clipRevision = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$Arguments = @()
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Executable $($Arguments -join ' ')"
    }
}

switch ($Action) {
    "data" {
        Invoke-Checked $python @("-B", "scripts\download_coco.py")
        Invoke-Checked $python @("-B", "scripts\prepare_coco.py")
    }
    "features" {
        if (-not (Test-Path $manifest)) { throw "Run .\run_project.ps1 data first." }
        $args = @("-B", "scripts\extract_features.py", "--manifest", $manifest, "--output", $features, "--device", $Device)
        if ($Limit -gt 0) { $args += @("--max-images", $Limit) }
        Invoke-Checked $python $args
    }
    "train" {
        if (-not (Test-Path $features)) { throw "Run features first." }
        Invoke-Checked $python @("-B", "scripts\train_mosaic.py", "--manifest", $manifest, "--features", $features, "--output-dir", $checkpoint, "--device", $Device)
    }
    "reranker" {
        Invoke-Checked $python @("-B", "scripts\train_reranker.py", "--manifest", $manifest, "--features", $features, "--adapter-dir", $checkpoint, "--output-dir", $checkpoint, "--device", $Device)
    }
    "evaluate" {
        if (-not (Test-Path $features)) { throw "Run features first." }
        Invoke-Checked $python @("-B", "scripts\evaluate_mosaic.py", "--manifest", $manifest, "--features", $features, "--checkpoint-dir", $checkpoint, "--output-dir", "reports", "--device", $Device)
    }
    "index" {
        if (-not (Test-Path $features)) { throw "Run features first." }
        Invoke-Checked $python @("-B", "scripts\build_index.py", "--features", $features, "--checkpoint-dir", $checkpoint, "--output-dir", $checkpoint, "--device", "cpu")
    }
    "all" {
        Invoke-Checked "$PSScriptRoot\run_project.ps1" @("data")
        Invoke-Checked "$PSScriptRoot\run_project.ps1" @("features", "-Device", $Device)
        Invoke-Checked "$PSScriptRoot\run_project.ps1" @("train", "-Device", $Device)
        Invoke-Checked "$PSScriptRoot\run_project.ps1" @("reranker", "-Device", $Device)
        Invoke-Checked "$PSScriptRoot\run_project.ps1" @("evaluate", "-Device", $Device)
        Invoke-Checked "$PSScriptRoot\run_project.ps1" @("index")
    }
    "smoke" {
        Invoke-Checked $python @("-B", "scripts\build_toy.py")
        Invoke-Checked $python @("-B", "scripts\smoke_mosaic.py")
        Invoke-Checked $python @("-B", "-m", "unittest", "discover", "-s", "tests", "-q")
    }
    "serve" {
        Invoke-Checked $python @("-B", "-m", "mosaic.serving", "--root", $PSScriptRoot)
    }
    "export-project4" {
        Invoke-Checked $python @("-B", "scripts\export_project4.py", "--input", "artifacts\mosaic_coco5k_v1\item_vectors.npz", "--output", "artifacts\project4_content_vectors.npz", "--allow-identity-demo")
    }
    "verify" {
        Invoke-Checked $python @("-B", "scripts\verify_mosaic.py")
    }
    "deploy-verify" {
        Invoke-Checked $python @("-B", "scripts\verify_deployment_bundle.py")
    }
    "video-data" {
        Write-Host "MSR-VTT automatic download is disabled in the public snapshot."
        Write-Host "Place an authorized local copy under data/raw/msrvtt or set MOSAIC_VIDEO_ROOT."
        Invoke-Checked $python @("-B", "scripts\prepare_msrvtt.py")
    }
    "video-features-train" {
        if (-not (Test-Path $videoTrainManifest)) { throw "Run video-data first." }
        $args = @("-B", "scripts\extract_video_features.py", "--manifest", $videoTrainManifest, "--output", $videoTrainFeatures, "--revision", $clipRevision, "--device", $Device, "--video-batch-size", "16", "--decode-workers", "8")
        if ($Limit -gt 0) { $args += @("--max-videos", $Limit) }
        Invoke-Checked $python $args
    }
    "video-train" {
        if (-not (Test-Path $videoTrainFeatures)) { throw "Run video-features-train first." }
        Invoke-Checked $python @("-B", "scripts\train_video.py", "--manifest", $videoTrainManifest, "--features", $videoTrainFeatures, "--output-dir", $videoArtifact, "--config", $videoConfig, "--device", $Device)
    }
    "video-dev" {
        Invoke-Checked $python @("-B", "scripts\evaluate_video_dev.py", "--manifest", $videoTrainManifest, "--features", $videoTrainFeatures, "--checkpoint-dir", $videoArtifact, "--config", $videoConfig, "--device", $Device)
    }
    "video-ablation" {
        Invoke-Checked $python @("-B", "scripts\run_video_dev_ablations.py", "--device", $Device)
    }
    "video-features-test" {
        if (-not (Test-Path "$videoArtifact\training_summary.json")) { throw "Freeze Dev selection with video-train first." }
        $tracked = (& git status --porcelain --untracked-files=no)
        if ($LASTEXITCODE -ne 0 -or $tracked) { throw "Commit a clean tracked Dev-selected implementation before extracting Test features." }
        Invoke-Checked $python @("-B", "scripts\extract_video_features.py", "--manifest", $videoTestManifest, "--output", $videoTestFeatures, "--revision", $clipRevision, "--device", $Device, "--video-batch-size", "16", "--decode-workers", "8")
    }
    "video-final" {
        if (-not (Test-Path $videoTestFeatures)) { throw "Run video-features-test after Dev freeze." }
        Invoke-Checked $python @("-B", "scripts\finalize_msrvtt.py", "--device", $Device)
    }
    "video-verify" {
        Invoke-Checked $python @("-B", "scripts\verify_msrvtt_final.py")
    }
}
if ($LASTEXITCODE -ne 0) {
    throw "MOSAIC action '$Action' failed with exit code $LASTEXITCODE"
}
exit 0
