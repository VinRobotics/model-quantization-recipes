#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
#
# Strip System 1 out of the InternVLA checkpoint, leaving a stock Qwen2.5-VL System 2.
#
# This is the step that makes the rest of the recipe ordinary: after it, quantize, export,
# build and the bridge verification all operate on a plain Qwen2.5-VL checkpoint and never
# import InternNav. It is pure safetensors manipulation -- no model is constructed -- so it
# runs anywhere, including without a GPU.
#
# Costs one ~15 GB intermediate copy. Pass --free_source to delete each source shard as it
# is consumed if disk is tight; the peak then is one shard rather than two checkpoints.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${INTERNVLA_CKPT:-$HOME/InternNav/checkpoints/InternVLA-N1-DualVLN}"
OUTPUT_PATH="${REPKG_CKPT:-$HOME/vln-opt-work/qwen25vl_system2}"
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)      MODEL_PATH="$2"; shift 2 ;;
        --output_path)     OUTPUT_PATH="$2"; shift 2 ;;
        --free_source)     EXTRA+=(--free_source); shift ;;
        --skip_disk_check) EXTRA+=(--skip_disk_check); shift ;;
        -h|--help)         sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "[ERROR] unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -d "$MODEL_PATH" ]] || { echo "[ERROR] checkpoint not found: $MODEL_PATH" >&2; exit 1; }

exec python -u "$HERE/../repackage_system2.py" \
    --model_path "$MODEL_PATH" \
    --output_path "$OUTPUT_PATH" \
    "${EXTRA[@]}"
