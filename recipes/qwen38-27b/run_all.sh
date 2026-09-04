#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
#
# run_all.sh — produce every published checkpoint, one strategy per invocation
# of quantize.py. Sequential on purpose: each run wants the whole card.
#
#   MODEL_PATH=/path/to/Qwen3.8-27B ./run_all.sh
#   MODEL_PATH=... STRATEGIES="fp8 nvfp4" ./run_all.sh
#
# Per-strategy knobs (NUM_CALIB_SAMPLES, MAX_SEQ_LEN, MAX_MEMORY_PER_GPU, ...)
# are read by run_quantize.sh and inherited from this environment.
# =============================================================================
set -uo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the base Qwen3.8-27B checkpoint}"
OUT_ROOT="${OUT_ROOT:-${RECIPE_DIR}/outputs}"
STRATEGIES="${STRATEGIES:-fp8 nvfp4 fp8-dynamic}"

export MODEL_PATH OUT_ROOT
mkdir -p "${OUT_ROOT}" "${RECIPE_DIR}/logs"

for strategy in ${STRATEGIES}; do
    out="${OUT_ROOT}/qwen38-27b-${strategy}"
    log="${RECIPE_DIR}/logs/quantize_${strategy}.log"
    echo "=============================================================="
    echo "[$(date '+%F %T')] START ${strategy} -> ${out}"
    echo "=============================================================="

    STRATEGY="${strategy}" OUTPUT_DIR="${out}" \
        "${RECIPE_DIR}/run_quantize.sh" > "${log}" 2>&1

    rc=$?
    if [ ${rc} -eq 0 ]; then
        echo "[$(date '+%F %T')] OK ${strategy} — $(du -sh "${out}" | cut -f1)"
    else
        echo "[$(date '+%F %T')] FAILED ${strategy} (rc=${rc}) — see ${log}"
        tail -20 "${log}"
    fi
done

echo "=============================================================="
echo "All done. Checkpoints under ${OUT_ROOT}:"
du -sh "${OUT_ROOT}"/* 2>/dev/null
