#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
#
# Quantize the repackaged System 2 with one scheme x strategy combination.
#
# Input is the *repackaged* checkpoint, not the InternVLA one -- see 01_repackage.sh.
# The scheme/strategy validity gate lives in quantize/quant_schemes.py and rejects the
# impossible combinations early, with the reason; nvfp4 x {s1,s2} additionally needs
# --allow_experimental because it passes text fluency and still breaks the navigation
# bridge (z_latents 0.931 against a 0.99 gate).
#
# Calibration defaults to 'auto' -- cnn_dailymail text for LLM-only strategies -- matching
# quantize.py. Pass --calib vln to use navigation episodes instead; it needs the scenes from
# 00_fetch_calib_scenes.sh and, measured here, changes held-out z_latents by 0.00003, so
# prefer it for honesty rather than for accuracy.
#
# Pass --dry_run to load the model and validate the configuration without quantizing.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${REPKG_CKPT:-$HOME/vln-opt-work/qwen25vl_system2}"
OUTPUT_PATH=""
SCHEME="${SCHEME:-fp8_default}"
STRATEGY="${STRATEGY:-s1}"
DEVICE="${DEVICE:-cuda}"
CALIB="${CALIB:-auto}"
CALIB_DATA_ROOT="${CALIB_DATA_ROOT:-$HOME/vln-opt-work/calib_scenes}"
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)         MODEL_PATH="$2"; shift 2 ;;
        --output_path)        OUTPUT_PATH="$2"; shift 2 ;;
        --scheme)             SCHEME="$2"; shift 2 ;;
        --strategy)           STRATEGY="$2"; shift 2 ;;
        --device)             DEVICE="$2"; shift 2 ;;
        --calib)              CALIB="$2"; shift 2 ;;
        --calib_data_root)    CALIB_DATA_ROOT="$2"; shift 2 ;;
        --num_calib_samples)  EXTRA+=(--num_calib_samples "$2"); shift 2 ;;
        --allow_experimental) EXTRA+=(--allow_experimental); shift ;;
        --dry_run)            EXTRA+=(--dry_run); shift ;;
        -h|--help)            sed -n "2,19p" "$0"; exit 0 ;;
        *) echo "[ERROR] unknown argument: $1" >&2; exit 2 ;;
    esac
done

OUTPUT_PATH="${OUTPUT_PATH:-$HOME/vln-opt-work/qwen25vl_${STRATEGY}_${SCHEME}}"
[[ -d "$MODEL_PATH" ]] || { echo "[ERROR] checkpoint not found: $MODEL_PATH" >&2; exit 1; }

# ModelOpt's Triton kernels are compiled in-tree on Thor; without this the quantize pass
# fails at import time rather than at use.
export TRITON_BACKENDS_IN_TREE="${TRITON_BACKENDS_IN_TREE:-1}"

exec python -u "$HERE/../quantize.py" \
    --model_path "$MODEL_PATH" \
    --output_path "$OUTPUT_PATH" \
    --scheme "$SCHEME" \
    --strategy "$STRATEGY" \
    --device "$DEVICE" \
    --calib "$CALIB" \
    --calib_data_root "$CALIB_DATA_ROOT" \
    "${EXTRA[@]}"
