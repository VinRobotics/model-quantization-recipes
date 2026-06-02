#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
# build_engine.sh — Build TensorRT engines from ONNX checkpoint.
# Run on Jetson Orin AGX (not Nano — see README for OOM details).
#
# Usage:
#   ./build_engine.sh <onnx_dir> <engine_output_dir> [trt_edgellm_dir]

set -euo pipefail

ONNX_DIR=${1:?"[ERROR] Usage: $0 <onnx_dir> <engine_output_dir> [trt_edgellm_dir]"}
ENGINE_DIR=${2:?"[ERROR] Usage: $0 <onnx_dir> <engine_output_dir> [trt_edgellm_dir]"}
TRT_EDGELLM_DIR=${3:-"${HOME}/TensorRT-Edge-LLM"}

MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-1}
MAX_INPUT_LEN=${MAX_INPUT_LEN:-1024}
MAX_KV_CACHE=${MAX_KV_CACHE:-1024}
MAX_TIME_STEPS=${MAX_TIME_STEPS:-1500}
export TRT_MAX_WORKSPACE_SIZE=${TRT_MAX_WORKSPACE_SIZE:-536870912}

echo "======================================================"
echo "[INFO] ONNX dir          : ${ONNX_DIR}"
echo "[INFO] Engine output dir : ${ENGINE_DIR}"
echo "[INFO] TRT-EdgeLLM dir   : ${TRT_EDGELLM_DIR}"
echo "[INFO] max_batch_size    : ${MAX_BATCH_SIZE}"
echo "[INFO] max_input_len     : ${MAX_INPUT_LEN}"
echo "[INFO] max_kv_cache      : ${MAX_KV_CACHE}"
echo "[INFO] max_time_steps    : ${MAX_TIME_STEPS}"
echo "======================================================"

[ -d "${ONNX_DIR}" ]         || { echo "[ERROR] ONNX dir not found: ${ONNX_DIR}"; exit 1; }
[ -d "${TRT_EDGELLM_DIR}" ]  || { echo "[ERROR] TRT-EdgeLLM dir not found: ${TRT_EDGELLM_DIR}"; exit 1; }

mkdir -p "${ENGINE_DIR}" "${ENGINE_DIR}/audio_encoder"
cd "${TRT_EDGELLM_DIR}"

echo ""
echo "=== Step 1: Building LLM core engine ==="
./build/examples/llm/llm_build \
    --onnxDir            "${ONNX_DIR}" \
    --engineDir          "${ENGINE_DIR}" \
    --maxBatchSize       "${MAX_BATCH_SIZE}" \
    --maxInputLen        "${MAX_INPUT_LEN}" \
    --maxKVCacheCapacity "${MAX_KV_CACHE}"

echo ""
echo "--- Clearing page cache before audio encoder build ---"
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

echo ""
echo "=== Step 2: Building Audio Encoder engine ==="
./build/examples/multimodal/audio_build \
    --onnxDir      "${ONNX_DIR}/audio_encoder" \
    --engineDir    "${ENGINE_DIR}/audio_encoder" \
    --maxTimeSteps "${MAX_TIME_STEPS}"

echo ""
echo "======================================================"
echo "[DONE] Engine build complete: ${ENGINE_DIR}"
echo "======================================================"
