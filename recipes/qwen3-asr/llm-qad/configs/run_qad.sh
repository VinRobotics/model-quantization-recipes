#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
# run_qad.sh — Run the full 3-stage QAD pipeline: prepare → sanity → train
#
# Exits immediately if any stage fails.
#
# Usage:
#   bash run_qad.sh [--config CONFIG] [--nproc N] [--script SCRIPT] [--verbose]
#
# Examples:
#   bash run_qad.sh --config configs/qad_example.yaml --nproc 8   # H100 8-GPU
#   bash run_qad.sh --config configs/qad_example.yaml --nproc 1   # single GPU
#   bash run_qad.sh --config configs/qad_example.yaml             # auto-detect GPUs

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
CONFIG="configs/qad_example.yaml"
NPROC=""                                         # auto-detect if empty
MASTER_PORT=$(shuf -i 20000-65000 -n 1)         # random port to avoid conflicts
VERBOSE=""
SCRIPT="qad_train.py"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)       CONFIG="$2";       shift 2 ;;
        --nproc)        NPROC="$2";        shift 2 ;;
        --script)       SCRIPT="$2";       shift 2 ;;
        --master_port)  MASTER_PORT="$2";  shift 2 ;;
        --verbose)      VERBOSE="--verbose"; shift ;;
        -h|--help)
            echo "Usage: bash run_qad.sh [--config CONFIG] [--nproc N] [--script SCRIPT] [--verbose]"
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            echo "Usage: bash run_qad.sh [--config CONFIG] [--nproc N] [--verbose]"
            exit 1
            ;;
    esac
done

# ── Auto-detect GPU count ─────────────────────────────────────────────────────
if [[ -z "$NPROC" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        NPROC=$(nvidia-smi --list-gpus | wc -l)
    else
        NPROC=1
    fi
fi

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/qad_run_${TIMESTAMP}.log"

log_info() {
    local msg="[$(date '+%H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

log_error() {
    local msg="[$(date '+%H:%M:%S')] [ERROR] $1"
    echo "$msg" >&2
    echo "$msg" >> "$LOG_FILE"
}

# Tee all output to log file
exec > >(tee -a "$LOG_FILE") 2>&1

log_info "============================================================"
log_info "  QAD Pipeline"
log_info "  Config : $CONFIG"
log_info "  Script : $SCRIPT"
log_info "  GPUs   : $NPROC"
log_info "  Port   : $MASTER_PORT"
log_info "  Log    : $LOG_FILE"
log_info "============================================================"

# ── Preflight checks ─────────────────────────────────────────────────────────
if [[ ! -f "$CONFIG" ]]; then
    log_error "Config file not found: $CONFIG"
    exit 1
fi

if [[ ! -f "$SCRIPT" ]]; then
    log_error "Training script not found: $SCRIPT"
    exit 1
fi

if ! command -v python &>/dev/null; then
    log_error "python not found in PATH"
    exit 1
fi

if [[ "$NPROC" -gt 1 ]] && ! command -v torchrun &>/dev/null; then
    log_error "torchrun not found; required for multi-GPU training"
    exit 1
fi

# ── Stage 1: Pseudo-label generation ─────────────────────────────────────────
log_info ""
log_info ">>> STAGE 1: Pseudo-label generation (single GPU)"
log_info "    python $SCRIPT --config $CONFIG --mode prepare $VERBOSE"
log_info ""

if ! python "$SCRIPT" --config "$CONFIG" --mode prepare $VERBOSE; then
    log_error "STAGE 1 (prepare) FAILED — aborting."
    exit 1
fi

log_info ">>> STAGE 1 PASSED ✓"
log_info ""

# ── Stage 2: Sanity check ─────────────────────────────────────────────────────
log_info ">>> STAGE 2: Sanity check (single GPU)"
log_info "    python $SCRIPT --config $CONFIG --mode sanity $VERBOSE"
log_info ""

if ! python "$SCRIPT" --config "$CONFIG" --mode sanity $VERBOSE; then
    log_error "STAGE 2 (sanity) FAILED — aborting."
    exit 1
fi

log_info ">>> STAGE 2 PASSED ✓"
log_info ""

# ── Stage 3: QAD Training ─────────────────────────────────────────────────────
log_info ">>> STAGE 3: QAD Training ($NPROC GPU(s))"

if [[ "$NPROC" -gt 1 ]]; then
    TRAIN_CMD="torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT $SCRIPT --config $CONFIG --mode train $VERBOSE"
else
    # Single GPU: skip torchrun to avoid unnecessary overhead
    TRAIN_CMD="python $SCRIPT --config $CONFIG --mode train $VERBOSE"
fi

log_info "    $TRAIN_CMD"
log_info ""

if ! eval "$TRAIN_CMD"; then
    log_error "STAGE 3 (train) FAILED."
    exit 1
fi

log_info ">>> STAGE 3 PASSED ✓"
log_info ""
log_info "============================================================"
log_info "  QAD Pipeline COMPLETE"
log_info "  Full log: $LOG_FILE"
log_info "============================================================"
