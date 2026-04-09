#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENVMAP_DIR="${ENVMAP_DIR:-$ROOT_DIR/data/envmaps}"
R_CKPT="${R_CKPT:-$ROOT_DIR/pretrained_models/network-snapshot-000225.pkl}"

G_CKPT="${G_CKPT:-$ROOT_DIR/pretrained_models/ffhqrebalanced512-128.pkl}"
E_CKPT="${E_CKPT:-$ROOT_DIR/pretrained_models/encoder_FFHQ.pt}"
AFA_CKPT="${AFA_CKPT:-$ROOT_DIR/pretrained_models/afa_FFHQ.pt}"
LIGHT_DIRS_PATH="${LIGHT_DIRS_PATH:-}"
ENV_ZSPIRAL_PATH="${ENV_ZSPIRAL_PATH:-}"
CUDA_IDX="${CUDA_VISIBLE_DEVICES:-0}"

SCAN_ID="${SCAN_ID:-ID00600}"
if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/inference_relighting.sh [scan_id]"
  echo "Examples:"
  echo "  bash scripts/inference_relighting.sh"
  echo "  bash scripts/inference_relighting.sh 600"
  echo "  bash scripts/inference_relighting.sh ID00600"
  exit 1
fi

if [[ $# -eq 1 ]]; then
  if [[ "$1" =~ ^ID[0-9]+$ ]]; then
    SCAN_ID="$1"
  elif [[ "$1" =~ ^[0-9]+$ ]]; then
    SCAN_ID="$(printf "ID%05d" "$1")"
  else
    echo "Invalid scan id: $1"
    echo "Use a numeric id like 600 or a full id like ID00600."
    exit 1
  fi
fi

DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/scans/$SCAN_ID}"
OUTDIR="${OUTDIR:-$ROOT_DIR/outputs/inference/$SCAN_ID}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Input data directory not found: $DATA_DIR"
  exit 1
fi

if [[ ! -d "$ENVMAP_DIR" ]]; then
  echo "Environment map directory not found: $ENVMAP_DIR"
  exit 1
fi

if [[ ! -e "$R_CKPT" ]]; then
  echo "Reflectance checkpoint not found: $R_CKPT"
  exit 1
fi

mkdir -p "$OUTDIR"

echo "CUDA_VISIBLE_DEVICES=${CUDA_IDX}"
echo "Working Directory: $ROOT_DIR"
echo "Scan ID: $SCAN_ID"
echo "Input Data: $DATA_DIR"
echo "Reflectance Checkpoint: $R_CKPT"
echo "Environment Maps: $ENVMAP_DIR"

cd "$ROOT_DIR"

CMD=(
  "$PYTHON_BIN" infer_relighting.py
  --data "$DATA_DIR"
  --G_ckpt "$G_CKPT"
  --E_ckpt "$E_CKPT"
  --AFA_ckpt "$AFA_CKPT"
  --R_ckpt "$R_CKPT"
  --envmap_dir "$ENVMAP_DIR"
  --outdir "$OUTDIR"
  --decoder_depth=1
  --o_dim=32
  --arch=resnet
  --eval_mode=True
  --lightstage_res=full
  --use_viewdirs=True
  --input_view=True
  --multi_view=True
  --video=False
  --w_frames=240
  --add_envmap=True
  --use_mask=True
  --gen_mask_mode=2
  --dataset_name=mpi
  --use_sr_encoder=True
  --emap_sample_fn=max
  --render_mode=0
  --olat_idx=80
  --cuda "$CUDA_IDX"
)

if [[ -n "$LIGHT_DIRS_PATH" ]]; then
  CMD+=(--light_dirs_path "$LIGHT_DIRS_PATH")
fi

if [[ -n "$ENV_ZSPIRAL_PATH" ]]; then
  CMD+=(--envmap_zspiral_path "$ENV_ZSPIRAL_PATH")
fi

"${CMD[@]}"
