#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

OUTDIR="${OUTDIR:-$ROOT_DIR/training-MPI-LS/04_runs}"
G_CKPT="${G_CKPT:-$ROOT_DIR/pretrained_models/ffhqrebalanced512-128.pkl}"
E_CKPT="${E_CKPT:-$ROOT_DIR/pretrained_models/encoder_FFHQ.pt}"
AFA_CKPT="${AFA_CKPT:-$ROOT_DIR/pretrained_models/afa_FFHQ.pt}"
R_CKPT="${R_CKPT:-}"

PORT="${PORT:-6000}"
PRINT_INTERVAL="${PRINT_INTERVAL:-1000}"
DATASET_NAME="${DATASET_NAME:-mpi}"

BATCH="${BATCH:-9}"
NUM_GPUS="${NUM_GPUS:-3}"
NUM_IDS="${NUM_IDS:-450}"
NUM_VIEWS="${NUM_VIEWS:-1}"
LIGHTSTAGE_RES="${LIGHTSTAGE_RES:-half}"

LR="${LR:-0.00015}"
WT_COL="${WT_COL:-1}"
WT_LPIPS="${WT_LPIPS:-0.3}"
LPIPS_WT_MODE="${LPIPS_WT_MODE:--1}"
USE_HDR_LOSS="${USE_HDR_LOSS:-False}"

START_STAGE2_AFTER="${START_STAGE2_AFTER:-70000}"
UPDATE_SR_AFTER="${UPDATE_SR_AFTER:-70000}"
GEN_NRR="${GEN_NRR:-128}"
ENC_NRR="${ENC_NRR:-128}"
DECODER_DEPTH="${DECODER_DEPTH:-1}"
O_DIM="${O_DIM:-32}"
USE_VIEWDIRS="${USE_VIEWDIRS:-True}"
ARCH="${ARCH:-resnet}"
ACT_FN="${ACT_FN:-sigmoid}"
TONEMAP="${TONEMAP:-True}"
USE_SR_ENCODER="${USE_SR_ENCODER:-True}"
RELOAD="${RELOAD:-True}"
EVAL_MODE="${EVAL_MODE:-False}"
DEBUG_MODE="${DEBUG_MODE:-False}"

if [[ $# -gt 0 ]]; then
  echo "This release script does not use positional arguments."
  echo "Override settings with environment variables instead."
  exit 1
fi

for f in "$G_CKPT" "$E_CKPT" "$AFA_CKPT"; do
  if [[ ! -f "$f" ]]; then
    echo "Required checkpoint not found: $f"
    exit 1
  fi
done

mkdir -p "$OUTDIR"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Working Directory: $ROOT_DIR"
echo "Output Directory: $OUTDIR"
echo "Dataset Name: $DATASET_NAME"
echo "Num GPUs: $NUM_GPUS"
echo "Batch Size: $BATCH"

cd "$ROOT_DIR"

CMD=(
  "$PYTHON_BIN" train_lightstage_encoder_relight.py
  --G_ckpt "$G_CKPT"
  --E_ckpt "$E_CKPT"
  --AFA_ckpt "$AFA_CKPT"
  --outdir "$OUTDIR"
  --debug_mode="$DEBUG_MODE"
  --print_interval="$PRINT_INTERVAL"
  --dataset_name="$DATASET_NAME"
  --eval_mode="$EVAL_MODE"
  --port="$PORT"
  --reload="$RELOAD"
  --start_stage2_after="$START_STAGE2_AFTER"
  --update_sr_after="$UPDATE_SR_AFTER"
  --batch="$BATCH"
  --num_gpus="$NUM_GPUS"
  --num_ids="$NUM_IDS"
  --num_views="$NUM_VIEWS"
  --lightstage_res="$LIGHTSTAGE_RES"
  --lr="$LR"
  --wt_col="$WT_COL"
  --wt_lpips="$WT_LPIPS"
  --lpips_wt_mode="$LPIPS_WT_MODE"
  --use_hdr_loss="$USE_HDR_LOSS"
  --gen_nrr="$GEN_NRR"
  --enc_nrr="$ENC_NRR"
  --decoder_depth="$DECODER_DEPTH"
  --o_dim="$O_DIM"
  --use_viewdirs="$USE_VIEWDIRS"
  --relight=True
  --arch="$ARCH"
  --act_fn="$ACT_FN"
  --tonemap="$TONEMAP"
  --use_sr_encoder="$USE_SR_ENCODER"
)

if [[ -n "$R_CKPT" ]]; then
  CMD+=(--R_ckpt "$R_CKPT")
fi

"${CMD[@]}"
