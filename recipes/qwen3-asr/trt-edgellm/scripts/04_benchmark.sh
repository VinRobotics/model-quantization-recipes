#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
# benchmark.sh — Batch inference benchmark on Jetson Orin Nano.
#                4-phase pipeline: preprocess → build JSON → inference → WER/RTF.
#
# Environment shortcuts:
#   AUDIO_DIR       — directory containing .wav files
#   PROMPTS         — prompts.txt path (<id> <reference> per line)
#   ENGINE_DIR      — TRT engine directory
#   WORK_DIR        — working directory for intermediates & results
#   TRT_EDGELLM_DIR — TensorRT-Edge-LLM repo (default: ~/TensorRT-Edge-LLM)
#
# Usage:
#   ./benchmark.sh \
#       --audio_dir   /path/to/wav/files \
#       --prompts     /path/to/prompts.txt \
#       --engine_dir  ~/Qwen3-ASR-1.7B-int8-Engines \
#       --work_dir    ./results_int8
#
# NOTE: Audio preprocessing is CPU-bound. On Jetson Orin Nano (6-core ARM)
#       allow ~10-15 minutes for 760 files. Already-processed files are skipped.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=("$@")
[[ ! " ${ARGS[*]:-} " =~ "--audio_dir"       ]] && [ -n "${AUDIO_DIR:-}"       ] && ARGS+=(--audio_dir       "${AUDIO_DIR}")
[[ ! " ${ARGS[*]:-} " =~ "--prompts"          ]] && [ -n "${PROMPTS:-}"         ] && ARGS+=(--prompts         "${PROMPTS}")
[[ ! " ${ARGS[*]:-} " =~ "--engine_dir"       ]] && [ -n "${ENGINE_DIR:-}"      ] && ARGS+=(--engine_dir      "${ENGINE_DIR}")
[[ ! " ${ARGS[*]:-} " =~ "--work_dir"         ]] && [ -n "${WORK_DIR:-}"        ] && ARGS+=(--work_dir        "${WORK_DIR}")
[[ ! " ${ARGS[*]:-} " =~ "--trt_edgellm_dir"  ]] && [ -n "${TRT_EDGELLM_DIR:-}" ] && ARGS+=(--trt_edgellm_dir "${TRT_EDGELLM_DIR}")
echo "[INFO] benchmark.sh → python ${SCRIPT_DIR}/benchmark.py ${ARGS[*]}"
python "${SCRIPT_DIR}/../benchmark.py" "${ARGS[@]}"
