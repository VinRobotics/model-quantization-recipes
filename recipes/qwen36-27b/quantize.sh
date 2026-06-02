#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
#
# quantize.sh — Quantize Qwen3.6-27B on 2× NVIDIA RTX 5090 (32 GiB each)
#
# Usage:
#   ./quantize.sh [--scheme fp8|nvfp4|int8|int4] [--model_path PATH] [--output_path PATH]
#
# All flags are optional; defaults are set below.
# =============================================================================

set -euo pipefail

# ── Defaults (edit here or pass as CLI flags) ─────────────────────────────────
SCHEME="fp8"
MODEL_PATH="./Qwen3.6-27B"
OUTPUT_PATH=""                  # auto-derived from MODEL_PATH + SCHEME if empty
VENV_PATH="./venv"              # Python venv used for compatibility patches

NUM_CALIB_SAMPLES=512
MAX_SEQ_LEN=1024
MAX_MEMORY_PER_GPU=28           # RTX 5090 = 32 GiB; 4 GiB headroom
DATASET_ID="abisee/cnn_dailymail"
DATASET_CONFIG="3.0.0"
DATASET_SPLIT="train"
DATASET_TEXT_FIELD="article"

# ── Parse CLI flags ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scheme)             SCHEME="$2";             shift 2 ;;
        --model_path)         MODEL_PATH="$2";         shift 2 ;;
        --output_path)        OUTPUT_PATH="$2";        shift 2 ;;
        --num_calib_samples)  NUM_CALIB_SAMPLES="$2";  shift 2 ;;
        --max_seq_len)        MAX_SEQ_LEN="$2";        shift 2 ;;
        --max_memory_per_gpu) MAX_MEMORY_PER_GPU="$2"; shift 2 ;;
        --venv_path)          VENV_PATH="$2";          shift 2 ;;
        *) echo "[ERROR] Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Logging helpers ───────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Validate GPU ──────────────────────────────────────────────────────────────
log "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
    || die "nvidia-smi not found — is the NVIDIA driver installed?"
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
log "Detected ${GPU_COUNT} GPU(s)."

# ── Compatibility patches ─────────────────────────────────────────────────────
# Fix 1 — TORCH_INIT_FUNCTIONS import removed in transformers ≥ 5.6
DEV_PY=$(find "${VENV_PATH}" -path "*/llmcompressor/utils/dev.py" 2>/dev/null | head -1 || true)
if [[ -n "${DEV_PY}" ]]; then
    if grep -q "from transformers.modeling_utils import TORCH_INIT_FUNCTIONS" "${DEV_PY}"; then
        log "Applying patch 1: TORCH_INIT_FUNCTIONS → empty dict in ${DEV_PY}"
        sed -i \
            's/from transformers\.modeling_utils import TORCH_INIT_FUNCTIONS/TORCH_INIT_FUNCTIONS = {}/' \
            "${DEV_PY}"
        log "Patch 1 applied."
    else
        warn "Patch 1 already applied or not needed — skipping."
    fi
else
    warn "llmcompressor/utils/dev.py not found in ${VENV_PATH} — skipping patch 1."
fi

# Fix 2 — _get_no_split_modules API removed in newer transformers
MODULE_PY=$(find "${VENV_PATH}" -path "*/llmcompressor/utils/pytorch/module.py" 2>/dev/null | head -1 || true)
if [[ -n "${MODULE_PY}" ]]; then
    if grep -q '_get_no_split_modules("auto")' "${MODULE_PY}"; then
        log "Applying patch 2: _get_no_split_modules → _no_split_modules in ${MODULE_PY}"
        sed -i \
            's/model\._get_no_split_modules("auto")/model._no_split_modules/' \
            "${MODULE_PY}"
        log "Patch 2 applied."
    else
        warn "Patch 2 already applied or not needed — skipping."
    fi
else
    warn "llmcompressor/utils/pytorch/module.py not found in ${VENV_PATH} — skipping patch 2."
fi

# ── Validate model path ───────────────────────────────────────────────────────
[[ -d "${MODEL_PATH}" ]] || die "Model path not found: ${MODEL_PATH}"

# ── Run quantization ──────────────────────────────────────────────────────────
log "Starting ${SCHEME^^} quantisation..."
log "  Model:            ${MODEL_PATH}"
log "  Output:           ${OUTPUT_PATH:-<auto-derived>}"
log "  Scheme:           ${SCHEME}"
log "  Samples:          ${NUM_CALIB_SAMPLES}  |  SeqLen: ${MAX_SEQ_LEN}"
log "  VRAM cap/GPU:     ${MAX_MEMORY_PER_GPU} GiB  |  GPUs: ${GPU_COUNT}"
echo ""

PYTHON_ARGS=(
    --model_path         "${MODEL_PATH}"
    --scheme             "${SCHEME}"
    --num_calib_samples  "${NUM_CALIB_SAMPLES}"
    --max_seq_len        "${MAX_SEQ_LEN}"
    --dataset_id         "${DATASET_ID}"
    --dataset_config     "${DATASET_CONFIG}"
    --dataset_split      "${DATASET_SPLIT}"
    --dataset_text_field "${DATASET_TEXT_FIELD}"
    --max_memory_per_gpu "${MAX_MEMORY_PER_GPU}"
)

if [[ -n "${OUTPUT_PATH}" ]]; then
    PYTHON_ARGS+=(--output_path "${OUTPUT_PATH}")
fi

python quantize.py "${PYTHON_ARGS[@]}"

log "Quantisation complete."
