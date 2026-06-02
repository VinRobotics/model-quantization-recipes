#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
# =============================================================================
# quantize.sh — NVFP4 quantization runner
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH="./model"
QUANT_DTYPE="nvfp4"
OUTPUT_PATH="./model-NVFP4"
NUM_CALIB_SAMPLES=512
MAX_SEQ_LEN=1024
SCRIPT="quantize.py"

# ── Sanity checks ─────────────────────────────────────────────────────────────
command -v nvidia-smi &>/dev/null || { echo "[ERROR] nvidia-smi not found."; exit 1; }

if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Model path does not exist: $MODEL_PATH"
    exit 1
fi

if ! python3 -c "import modelopt" 2>/dev/null; then
    echo "[ERROR] nvidia-modelopt not installed. Run: pip install nvidia-modelopt[all]"
    exit 1
fi

# ── GPU info ──────────────────────────────────────────────────────────────────
echo "=== GPU Info ==="
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
echo ""

if [[ "$GPU_NAME" != *"H100"* && "$GPU_NAME" != *"Blackwell"* ]]; then
    echo "[WARNING] NVFP4 requires SM100+. GPU '$GPU_NAME' may not be supported."
    echo ""
fi

# ── Runtime tuning ────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "=== Starting Quantization ==="
echo "  model_path   : $MODEL_PATH"
echo "  output_path  : $OUTPUT_PATH"
echo "  quant_dtype  : $QUANT_DTYPE"
echo "  calib_samples: $NUM_CALIB_SAMPLES"
echo "  max_seq_len  : $MAX_SEQ_LEN"
echo ""

python3 "$SCRIPT" \
    --model_path        "$MODEL_PATH"        \
    --output_path       "$OUTPUT_PATH"       \
    --num_calib_samples "$NUM_CALIB_SAMPLES" \
    --max_seq_len       "$MAX_SEQ_LEN"       \
    --quant_dtype       "$QUANT_DTYPE"

echo ""
echo "=== Quantization complete ==="
echo "Output: $OUTPUT_PATH"
