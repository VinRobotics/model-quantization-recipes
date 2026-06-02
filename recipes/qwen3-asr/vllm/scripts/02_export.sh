#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
# export.sh — Export quantized checkpoint to vLLM-compatible format.
#
# Environment shortcuts:
#   MODEL_PATH     — path to original Qwen3-ASR-1.7B checkpoint
#   QWEN_ASR_ROOT  — path to Qwen3-ASR repo root
#   DEVICE         — CUDA device (default: cuda:0)
#
# Usage (FP8):
#   ./export.sh \
#       --model_path    /path/to/Qwen3-ASR-1.7B \
#       --qwen_asr_root /path/to/Qwen3-ASR \
#       --quant_pt      ./qwen3asr_thinker_fp8.pt \
#       --tmp_dir       ./qwen3asr_fp8_hf \
#       --export_dir    ./qwen3asr_fp8_vllm

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=("$@")
[[ ! " ${ARGS[*]:-} " =~ "--model_path"    ]] && [ -n "${MODEL_PATH:-}"    ] && ARGS+=(--model_path    "${MODEL_PATH}")
[[ ! " ${ARGS[*]:-} " =~ "--qwen_asr_root" ]] && [ -n "${QWEN_ASR_ROOT:-}" ] && ARGS+=(--qwen_asr_root "${QWEN_ASR_ROOT}")
[[ ! " ${ARGS[*]:-} " =~ "--device"        ]] && [ -n "${DEVICE:-}"        ] && ARGS+=(--device        "${DEVICE}")
echo "[INFO] export.sh → python ${SCRIPT_DIR}/export.py ${ARGS[*]}"
python "${SCRIPT_DIR}/../export.py" "${ARGS[@]}"
