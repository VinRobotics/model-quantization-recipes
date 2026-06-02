#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
# export.sh — Export a quantized Qwen3-ASR-1.7B checkpoint to ONNX
#             (LLM thinker + Audio Encoder) for TensorRT-Edge-LLM.
#
# Usage:
#   ./export.sh <quantized_checkpoint_dir> <onnx_output_dir> [device]

set -euo pipefail

MODEL_PATH=${1:?"[ERROR] Usage: $0 <quantized_checkpoint_dir> <onnx_output_dir> [device]"}
OUTPUT_DIR=${2:?"[ERROR] Usage: $0 <quantized_checkpoint_dir> <onnx_output_dir> [device]"}
DEVICE=${3:-cuda}

echo "======================================================"
echo "[INFO] Model checkpoint : ${MODEL_PATH}"
echo "[INFO] ONNX output dir  : ${OUTPUT_DIR}"
echo "[INFO] Device           : ${DEVICE}"
echo "======================================================"

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "=== Step 1: Exporting LLM (thinker) ==="
tensorrt-edgellm-export-llm \
    --model_dir  "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --device     "${DEVICE}"

python3 - <<PYEOF
import json, os
config_path = os.path.join("${OUTPUT_DIR}", "config.json")
with open(config_path) as f:
    config = json.load(f)
config.update({"eos_token_id": 151645, "bos_token_id": 151643, "pad_token_id": 151643})
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print("[PATCH] Token IDs written to config.json")
PYEOF

echo ""
echo "=== Step 2: Exporting Audio Encoder ==="
tensorrt-edgellm-export-audio \
    --model_dir  "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}/audio_encoder" \
    --dtype      fp16

echo ""
echo "======================================================"
echo "[DONE] ONNX export complete: ${OUTPUT_DIR}"
echo "======================================================"
