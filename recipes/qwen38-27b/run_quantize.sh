#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
#
# run_quantize.sh — build one quantized Qwen3.8-27B checkpoint.
#
# Runs inline, or under SLURM with the header below. Everything is driven by
# environment variables so the script itself carries no site-specific paths.
#
#   MODEL_PATH=/path/to/Qwen3.8-27B ./run_quantize.sh
#   MODEL_PATH=... STRATEGY=nvfp4 ./run_quantize.sh
#   MODEL_PATH=... sbatch run_quantize.sh
#
# The SLURM directives below match the machine this recipe was validated on
# (1x H100 80 GB, 256 GB host RAM). Adjust the partition and GPU type for your
# cluster before submitting; they are ignored when the script is run inline.
# =============================================================================

#SBATCH --job-name=qwen38_ptq
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Paths — MODEL_PATH is the only required input
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the base Qwen3.8-27B checkpoint}"
OUT_ROOT="${OUT_ROOT:-${RECIPE_DIR}/outputs}"
STRATEGY="${STRATEGY:-fp8}"                  # fp8 | fp8-dynamic | fp8-block | nvfp4 | nvfp4a16
OUTPUT_DIR="${OUTPUT_DIR:-${OUT_ROOT}/qwen38-27b-${STRATEGY}}"

# Activate a virtualenv if one is named; otherwise use whatever python3 is live.
if [ -n "${QUANT_VENV:-}" ]; then
    # shellcheck disable=SC1091
    source "${QUANT_VENV}/bin/activate"
fi

# ---------------------------------------------------------------------------
# Calibration — CNN/DailyMail, text-only
# ---------------------------------------------------------------------------
NUM_CALIB_SAMPLES="${NUM_CALIB_SAMPLES:-512}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
DATASET_ID="${DATASET_ID:-abisee/cnn_dailymail}"
DATASET_CONFIG="${DATASET_CONFIG:-3.0.0}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
TEXT_FIELD="${TEXT_FIELD:-article,highlights,text}"

# ---------------------------------------------------------------------------
# Memory. 27B bf16 is ~54 GiB; cap the card below its total so accelerate leaves
# headroom for the calibration activations instead of filling it with weights.
# ---------------------------------------------------------------------------
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-74}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-160}"

# Fragmentation is the main OOM driver during sequential calibration.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# NVFP4 kernels need SM100+ to run fast. On older cards the checkpoint is still
# numerically valid, so allow the build; set ALLOW_UNSUPPORTED="" to enforce.
ALLOW_UNSUPPORTED="${ALLOW_UNSUPPORTED:---allow_unsupported_gpu}"

cd "${RECIPE_DIR}"
mkdir -p "${OUTPUT_DIR}" logs

echo "========================================"
echo "Job         : ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-inline})"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null)"
echo "Model       : ${MODEL_PATH}"
echo "Output      : ${OUTPUT_DIR}"
echo "Strategy    : ${STRATEGY}"
echo "Calibration : ${DATASET_ID}/${DATASET_CONFIG} split=${DATASET_SPLIT}"
echo "              ${NUM_CALIB_SAMPLES} samples @ max_seq_len ${MAX_SEQ_LEN}"
echo "========================================"

python3 quantize.py \
    --model_path         "${MODEL_PATH}"          \
    --output_path        "${OUTPUT_DIR}"          \
    --strategy           "${STRATEGY}"            \
    --dataset_id         "${DATASET_ID}"          \
    --dataset_config     "${DATASET_CONFIG}"      \
    --dataset_split      "${DATASET_SPLIT}"       \
    --text_field         "${TEXT_FIELD}"          \
    --num_calib_samples  "${NUM_CALIB_SAMPLES}"   \
    --max_seq_len        "${MAX_SEQ_LEN}"         \
    --max_memory_per_gpu "${MAX_MEMORY_PER_GPU}"  \
    --cpu_offload_gb     "${CPU_OFFLOAD_GB}"      \
    ${ALLOW_UNSUPPORTED}

EXIT_CODE=$?

echo "========================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "SUCCESS — checkpoint: ${OUTPUT_DIR}"
    du -sh "${OUTPUT_DIR}"
else
    echo "FAILED  — exit code: ${EXIT_CODE}"
fi
echo "========================================"

exit ${EXIT_CODE}
