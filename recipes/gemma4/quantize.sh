#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-./gemma4-model}"
QUANTIZATION="${QUANTIZATION:-fp8}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
OUTPUT_PATH="${OUTPUT_PATH:-./gemma4-${QUANTIZATION}}"
NUM_CALIB_SAMPLES="${NUM_CALIB_SAMPLES:-512}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
SCRIPT="quantize_gemma4.py"

if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Model path '$MODEL_PATH' does not exist."
    echo "        Set MODEL_PATH to a local Gemma4 checkpoint directory and retry."
    exit 1
fi

if ! python -c "import modelopt" 2>/dev/null; then
    echo "[ERROR] nvidia-modelopt not found. Run: uv sync --extra gemma4"
    exit 1
fi

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "=== Starting Gemma4 Quantization ==="
echo "  model_path   : $MODEL_PATH"
echo "  quantization : $QUANTIZATION"
echo "  model_dtype  : $MODEL_DTYPE"
echo "  output_path  : $OUTPUT_PATH"
echo "  calib_samples: $NUM_CALIB_SAMPLES"
echo "  max_seq_len  : $MAX_SEQ_LEN"
echo ""

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python "$SCRIPT" \
    --model_path "$MODEL_PATH" \
    --output_path "$OUTPUT_PATH" \
    --quantization "$QUANTIZATION" \
    --model_dtype "$MODEL_DTYPE" \
    --num_calib_samples "$NUM_CALIB_SAMPLES" \
    --max_seq_len "$MAX_SEQ_LEN"

echo ""
echo "=== Quantization complete ==="
echo "Output: $OUTPUT_PATH"
