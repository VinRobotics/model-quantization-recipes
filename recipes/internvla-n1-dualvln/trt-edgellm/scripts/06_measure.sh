#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
#
# Measure prefill/decode latency and bridge fidelity for every built engine, and write the
# numbers where run_matrix.py can find them.
#
# llm_bench reads base_config.json while llm_build writes config.json, so the copy below is
# not optional -- without it the benchmark cannot open an engine it was just handed.

set -euo pipefail

WORK_DIR="${WORK_DIR:-$HOME/vln-opt-work}"
ENGINE_DIR="${ENGINE_DIR:-$WORK_DIR/engines}"
TRT_EDGELLM_DIR="${TRT_EDGELLM_DIR:-$HOME/modelopt/Hung-TRT-Edge-LLM}"
INPUT_LEN="${INPUT_LEN:-1024}"
PAST_KV_LEN="${PAST_KV_LEN:-1024}"
OUT="${OUT:-$WORK_DIR/out/latency.json}"

LLM_BENCH="$TRT_EDGELLM_DIR/build/examples/llm/llm_bench"
[[ -x "$LLM_BENCH" ]] || { echo "[ERROR] not found: $LLM_BENCH" >&2; exit 1; }
export EDGELLM_PLUGIN_PATH="$TRT_EDGELLM_DIR/build/libNvInfer_edgellm_plugin.so"

# Map engine directory -> the checkpoint name run_matrix.py keys on.
declare -A CKPT_OF=(
    [base_fp16]=qwen25vl_system2
    [s1_fp8]=qwen25vl_s1_fp8
    [s1_nvfp4]=qwen25vl_s1_nvfp4
)

mkdir -p "$(dirname "$OUT")"
echo "{" > "$OUT"
first=1

for name in "${!CKPT_OF[@]}"; do
    engine="$ENGINE_DIR/$name/llm"
    [[ -f "$engine/llm.engine" ]] || { echo "  [skip] $name: no engine"; continue; }
    [[ -f "$engine/base_config.json" ]] || cp "$engine/config.json" "$engine/base_config.json"

    echo "  measuring $name ..."
    prefill=$("$LLM_BENCH" --engineDir "$engine" --mode prefill --inputLen "$INPUT_LEN" 2>&1 \
              | grep -oE "E2E Time \(actual performance\): [0-9.]+" | tail -1 | grep -oE "[0-9.]+$" || echo "")
    decode=$("$LLM_BENCH" --engineDir "$engine" --mode decode --pastKVLen "$PAST_KV_LEN" 2>&1 \
             | grep -oE "E2E Time \(actual performance\): [0-9.]+" | tail -1 | grep -oE "[0-9.]+$" || echo "")

    [[ $first -eq 1 ]] || echo "," >> "$OUT"
    first=0
    printf '  "%s": {"prefill_ms": %s, "decode_ms": %s}' \
        "${CKPT_OF[$name]}" "${prefill:-null}" "${decode:-null}" >> "$OUT"
    echo "    prefill ${prefill:-n/a} ms | decode ${decode:-n/a} ms"
done

echo "" >> "$OUT"
echo "}" >> "$OUT"
echo "Wrote $OUT"
