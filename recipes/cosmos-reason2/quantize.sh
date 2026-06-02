#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/Cosmos-Reason2-2B}"
OUTPUT_PATH="${OUTPUT_PATH:-./cosmos-reason2-nvfp4}"
NUM_CALIB_SAMPLES="${NUM_CALIB_SAMPLES:-512}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-30}"

python "$(dirname "$0")/quantize_cosmos_reason2.py" \
  --model_path "$MODEL_PATH" \
  --output_path "$OUTPUT_PATH" \
  --num_calib_samples "$NUM_CALIB_SAMPLES" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --max_memory_per_gpu "$MAX_MEMORY_PER_GPU"
