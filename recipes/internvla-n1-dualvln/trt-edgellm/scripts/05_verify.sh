#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
#
# Run the acceptance gates.
#
# Default (System 2): the z_latents bridge -- the last-layer hidden states of the 4 TRAJ
# tokens through the host-side norm and cond_projector, engine against a PyTorch reference.
# This is the number that decides whether a scheme ships. Text fluency is not sufficient
# evidence: NVFP4 stays fluent and fails this at 0.931.
#
# --system1: trajectory parity for the diffusion head and memory block. This one runs in TWO
# stages under TWO interpreters, and that is not incidental. InternNav targets transformers
# 4.x while the TensorRT bindings ship for Python 3.12 where transformers is 5.x, so no
# single environment has both. Stage A writes the PyTorch reference's inputs and outputs to
# a .pt; stage B feeds the engines those same tensors. Set PYTHON_PT and PYTHON_TRT to the
# two interpreters -- if they are left at the default the stages run in whatever is active,
# which works only if one environment happens to satisfy both.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
SYSTEM1=0
ENGINE_DIR=""
ENGINE_SUFFIX="${ENGINE_SUFFIX:-bf16}"
REPKG_CKPT="${REPKG_CKPT:-$HOME/vln-opt-work/qwen25vl_system2}"
CALIB_DATA_ROOT="${CALIB_DATA_ROOT:-$HOME/vln-opt-work/calib_scenes}"
INTERNVLA_CKPT="${INTERNVLA_CKPT:-$HOME/InternNav/checkpoints/InternVLA-N1-DualVLN}"
INTERNNAV_PATH="${INTERNNAV_PATH:-$HOME/InternNav}"
WORK_DIR="${WORK_DIR:-$HOME/vln-opt-work}"
VLN=0
BENCH_ITERS="${BENCH_ITERS:-0}"
# The System 2 gate needs transformers and the TensorRT bindings in one interpreter; the
# System 1 gate cannot have both and splits across PYTHON_PT / PYTHON_TRT below.
PYTHON="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --system1)         SYSTEM1=1; shift ;;
        --vln)             VLN=1; shift ;;
        --engine_dir)      ENGINE_DIR="$2"; shift 2 ;;
        --engine_suffix)   ENGINE_SUFFIX="$2"; shift 2 ;;
        --repkg_ckpt)      REPKG_CKPT="$2"; shift 2 ;;
        --calib_data_root) CALIB_DATA_ROOT="$2"; shift 2 ;;
        --internvla_ckpt)  INTERNVLA_CKPT="$2"; shift 2 ;;
        --internnav_path)  INTERNNAV_PATH="$2"; shift 2 ;;
        --work_dir)        WORK_DIR="$2"; shift 2 ;;
        --bench_iters)     BENCH_ITERS="$2"; shift 2 ;;
        -h|--help)         sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "[ERROR] unknown argument: $1" >&2; exit 2 ;;
    esac
done

export WORK_DIR REPKG_CKPT INTERNNAV_PATH INTERNVLA_CKPT CALIB_DATA_ROOT
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export EDGELLM_PLUGIN_PATH="${EDGELLM_PLUGIN_PATH:-$HOME/modelopt/Hung-TRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so}"

if [[ "$SYSTEM1" -eq 1 ]]; then
    : "${ENGINE_DIR:=$WORK_DIR/onnx}"
    PYTHON_PT="${PYTHON_PT:-python}"
    PYTHON_TRT="${PYTHON_TRT:-python}"
    REF="${REFERENCE_PATH:-$WORK_DIR/out/system1_reference.pt}"

    echo "== stage A: PyTorch reference (needs InternNav; $PYTHON_PT) =="
    PYTHONPATH="$INTERNNAV_PATH:$PYTHONPATH" "$PYTHON_PT" -u "$ROOT/verify/dump_system1_reference.py" \
        --internvla_ckpt "$INTERNVLA_CKPT" --output_path "$REF"

    echo "== stage B: engines (needs TensorRT; $PYTHON_TRT) =="
    ARGS=(--reference_path "$REF" --engine_dir "$ENGINE_DIR" --engine_suffix "$ENGINE_SUFFIX")
    [[ "$BENCH_ITERS" -gt 0 ]] && ARGS+=(--bench_iters "$BENCH_ITERS")
    exec "$PYTHON_TRT" -u "$ROOT/verify/compare_system1_engines.py" "${ARGS[@]}"
fi

[[ -n "$ENGINE_DIR" ]] || { echo "[ERROR] --engine_dir is required" >&2; exit 2; }
ENGINE_PATH="${ENGINE_PATH:-$ENGINE_DIR/llm/llm.engine}"
[[ -f "$ENGINE_PATH" ]] || { echo "[ERROR] engine not found: $ENGINE_PATH" >&2; exit 1; }
export ENGINE_PATH

if [[ "$VLN" -eq 1 ]]; then
    exec "$PYTHON" -u "$ROOT/verify/verify_latents_vln.py"
fi
exec "$PYTHON" -u "$ROOT/verify/verify_latents.py"
