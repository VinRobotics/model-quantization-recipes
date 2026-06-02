#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
set -e

MODEL_DIR=${1:?"Usage: ./serve.sh <model_dir> <format> [gpu_id] [gpu_memory_utilization]"}
FORMAT=${2:?"Usage: ./serve.sh <model_dir> <format> [gpu_id] [gpu_memory_utilization]  (format: fp8|nvfp4)"}
GPU_ID=${3:-0}
GPU_MEM=${4:-0.7}

case "${FORMAT}" in
    fp8)   QUANT_FLAG="modelopt" ;;
    nvfp4) QUANT_FLAG="modelopt_fp4" ;;
    *)
        echo "[ERROR] format must be fp8 or nvfp4"
        exit 1
        ;;
esac

echo "[INFO] model           : ${MODEL_DIR}"
echo "[INFO] format          : ${FORMAT}  (quantization=${QUANT_FLAG})"
echo "[INFO] gpu             : ${GPU_ID}"
echo "[INFO] memory util     : ${GPU_MEM}"

CUDA_VISIBLE_DEVICES=${GPU_ID} qwen-asr-serve "${MODEL_DIR}" \
    --quantization "${QUANT_FLAG}" \
    --gpu-memory-utilization "${GPU_MEM}"
