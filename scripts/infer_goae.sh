#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/scans/ID00600}"
ENVMAP_DIR="${ENVMAP_DIR:-$ROOT_DIR/data/envmaps}"
OUTDIR="${OUTDIR:-$ROOT_DIR/outputs/inference/ID00600}"
R_CKPT="${R_CKPT:-$ROOT_DIR/pretrained_models/network-snapshot-000225.pkl}"

G_CKPT="${G_CKPT:-$ROOT_DIR/pretrained_models/ffhqrebalanced512-128.pkl}"
E_CKPT="${E_CKPT:-$ROOT_DIR/pretrained_models/encoder_FFHQ.pt}"
AFA_CKPT="${AFA_CKPT:-$ROOT_DIR/pretrained_models/afa_FFHQ.pt}"
LIGHT_DIRS_PATH="${LIGHT_DIRS_PATH:-}"
ENV_ZSPIRAL_PATH="${ENV_ZSPIRAL_PATH:-}"
CUDA_IDX="${CUDA_VISIBLE_DEVICES:-0}"

if [[ $# -gt 0 ]]; then
  echo "This release script no longer uses positional arguments."
  echo "Override paths with environment variables instead:"
  echo "  DATA_DIR, ENVMAP_DIR, OUTDIR, R_CKPT, G_CKPT, E_CKPT, AFA_CKPT"
  exit 1
fi

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
