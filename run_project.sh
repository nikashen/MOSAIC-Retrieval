#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${MOSAIC_PYTHON:-python3}"
export PYTHONPATH="$ROOT/src"
export PYTHONIOENCODING=utf-8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
CLIP_REVISION=3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268

ACTION="${1:-smoke}"
case "$ACTION" in
  smoke)
    "$PYTHON_BIN" -B scripts/build_toy.py
    "$PYTHON_BIN" -B scripts/smoke_mosaic.py
    "$PYTHON_BIN" -B -m unittest discover -s tests -q
    ;;
  verify)
    "$PYTHON_BIN" -B scripts/verify_mosaic.py
    ;;
  deploy-verify)
    "$PYTHON_BIN" -B scripts/verify_deployment_bundle.py
    ;;
  index)
    "$PYTHON_BIN" -B scripts/build_index.py \
      --features artifacts/mosaic_coco5k_v1/clip_features.npz \
      --checkpoint-dir artifacts/mosaic_coco5k_v1 \
      --output-dir artifacts/mosaic_coco5k_v1 --device cpu
    ;;
  serve)
    exec "$PYTHON_BIN" -B -m mosaic.serving --root "$ROOT" \
      --host "${MOSAIC_HOST:-127.0.0.1}" --port "${MOSAIC_PORT:-8050}" \
      --device "${MOSAIC_DEVICE:-cpu}"
    ;;
  video-data)
    echo "MSR-VTT automatic download is disabled; provide an authorized local copy." >&2
    "$PYTHON_BIN" -B scripts/prepare_msrvtt.py
    ;;
  video-features-train)
    "$PYTHON_BIN" -B scripts/extract_video_features.py \
      --manifest data/processed/msrvtt_train_dev_v1.json \
      --output artifacts/mosaic_msrvtt_1ka_v1/train_dev_clip_features.npz \
      --revision "$CLIP_REVISION" --device "${MOSAIC_DEVICE:-auto}" \
      --video-batch-size 16 --decode-workers 8
    ;;
  video-train)
    "$PYTHON_BIN" -B scripts/train_video.py \
      --manifest data/processed/msrvtt_train_dev_v1.json \
      --features artifacts/mosaic_msrvtt_1ka_v1/train_dev_clip_features.npz \
      --output-dir artifacts/mosaic_msrvtt_1ka_v1 \
      --config configs/msrvtt_1ka_v1.json --device "${MOSAIC_DEVICE:-auto}"
    ;;
  video-dev)
    "$PYTHON_BIN" -B scripts/evaluate_video_dev.py \
      --manifest data/processed/msrvtt_train_dev_v1.json \
      --features artifacts/mosaic_msrvtt_1ka_v1/train_dev_clip_features.npz \
      --checkpoint-dir artifacts/mosaic_msrvtt_1ka_v1 \
      --config configs/msrvtt_1ka_v1.json --device "${MOSAIC_DEVICE:-auto}"
    ;;
  video-ablation)
    "$PYTHON_BIN" -B scripts/run_video_dev_ablations.py \
      --device "${MOSAIC_DEVICE:-auto}"
    ;;
  video-final)
    "$PYTHON_BIN" -B scripts/finalize_msrvtt.py --device "${MOSAIC_DEVICE:-auto}"
    ;;
  video-verify)
    "$PYTHON_BIN" -B scripts/verify_msrvtt_final.py
    ;;
  *)
    echo "Unknown action: $ACTION (smoke|verify|deploy-verify|index|serve|video-data|video-features-train|video-train|video-dev|video-ablation|video-final|video-verify)" >&2
    exit 2
    ;;
esac
