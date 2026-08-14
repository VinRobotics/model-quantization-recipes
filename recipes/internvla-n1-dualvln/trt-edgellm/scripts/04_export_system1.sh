#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
#
# Export System 1 -- the NextDiT diffusion head and the memory block -- to BF16 engines.
#
# Upstream InternNav ships no ONNX or TensorRT path at all, so this is entirely the
# recipe's. Both exporters need InternNav importable, unlike everything under System 2.
#
# System 1 stays BF16 on purpose. FP8 was measured (trt-edgellm/quantize_system1.py) and
# costs 6x the waypoint deviation to save 0.7% of deployed weights and 1.7% of a planning
# step -- see the README.
#
# The two exporters write into $WORK_DIR/onnx, which is also where verify_system1.py looks;
# --engine_dir is linked to those files rather than holding copies, so both paths stay valid
# without a second 176 MB on disk.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
INTERNVLA_CKPT="${INTERNVLA_CKPT:-$HOME/InternNav/checkpoints/InternVLA-N1-DualVLN}"
INTERNNAV_PATH="${INTERNNAV_PATH:-$HOME/InternNav}"
WORK_DIR="${WORK_DIR:-$HOME/vln-opt-work}"
ENGINE_DIR=""
COMPONENTS="traj_dit,memory"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --internvla_ckpt) INTERNVLA_CKPT="$2"; shift 2 ;;
        --internnav_path) INTERNNAV_PATH="$2"; shift 2 ;;
        --work_dir)       WORK_DIR="$2"; shift 2 ;;
        --engine_dir)     ENGINE_DIR="$2"; shift 2 ;;
        --components)     COMPONENTS="$2"; shift 2 ;;
        -h|--help)        sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "[ERROR] unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -d "$INTERNNAV_PATH" ]] || { echo "[ERROR] InternNav not found: $INTERNNAV_PATH" >&2; exit 1; }
[[ -d "$INTERNVLA_CKPT" ]] || { echo "[ERROR] checkpoint not found: $INTERNVLA_CKPT" >&2; exit 1; }

export INTERNNAV_PATH INTERNVLA_CKPT WORK_DIR
# The exporters import both InternNav and the recipe's own modules.
export PYTHONPATH="$INTERNNAV_PATH:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ",$COMPONENTS," == *",traj_dit,"* ]]; then
    echo "== traj_dit =="
    python -u "$ROOT/export_traj_dit.py"
fi
if [[ ",$COMPONENTS," == *",memory,"* ]]; then
    echo "== memory block =="
    python -u "$ROOT/export_memory_block.py"
fi

if [[ -n "$ENGINE_DIR" ]]; then
    mkdir -p "$ENGINE_DIR"
    for f in "$WORK_DIR"/onnx/system1_*.engine; do
        [[ -e "$f" ]] || continue
        ln -sfn "$f" "$ENGINE_DIR/$(basename "$f")"
    done
    echo "Engines linked into: $ENGINE_DIR"
fi
ls -lh "$WORK_DIR"/onnx/system1_*.engine 2>/dev/null || true
