#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
# quantize.sh — PTQ for Qwen3-ASR-1.7B LLM component (FP8 or NVFP4)
#
# Environment shortcuts:
#   MODEL_PATH     — path to Qwen3-ASR-1.7B checkpoint
#   QWEN_ASR_ROOT  — path to Qwen3-ASR repo root
#   DEVICE         — CUDA device (default: cuda:0)
#
# Usage:
#   ./quantize.sh \
#       --model_path    /path/to/Qwen3-ASR-1.7B \
#       --qwen_asr_root /path/to/Qwen3-ASR \
#       --format        fp8 \
#       --algorithm     max \
#       --output_pt     ./qwen3asr_thinker_fp8.pt
#
# Formats  : fp8 | nvfp4
# Algorithms: max (default) | mse | awq_full

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=("$@")
[[ ! " ${ARGS[*]:-} " =~ "--model_path"    ]] && [ -n "${MODEL_PATH:-}"    ] && ARGS+=(--model_path    "${MODEL_PATH}")
[[ ! " ${ARGS[*]:-} " =~ "--qwen_asr_root" ]] && [ -n "${QWEN_ASR_ROOT:-}" ] && ARGS+=(--qwen_asr_root "${QWEN_ASR_ROOT}")
[[ ! " ${ARGS[*]:-} " =~ "--device"        ]] && [ -n "${DEVICE:-}"        ] && ARGS+=(--device        "${DEVICE}")
echo "[INFO] quantize.sh → python ${SCRIPT_DIR}/quantize.py ${ARGS[*]}"
python "${SCRIPT_DIR}/../quantize.py" "${ARGS[@]}"
