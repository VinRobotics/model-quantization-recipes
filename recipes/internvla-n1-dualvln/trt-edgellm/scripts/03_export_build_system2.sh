#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
#
# Export a System-2 checkpoint to ONNX and build the TensorRT-Edge-LLM engines.
#
# Works on both a quantized checkpoint and the unquantized one; pass --no_quantization
# for the latter to get the FP16 fidelity reference.
#
# Two details here are load-bearing and must not be "cleaned up":
#
#   * CausalLM.emit_hidden_states = True before the export. The System 2 -> System 1
#     bridge reads the last-layer hidden states of the trajectory tokens; without this the
#     engine emits logits only and the bridge cannot be evaluated at all.
#
#   * __LUNOWUD=-peep:fc_h_fusion=off on the engine build. TensorRT 10.13 miscompiles
#     Myelin's horizontal fusion of the gate/up projections on sm_110 at batch 1, and an
#     FP16 engine built without this emits fluent-looking gibberish. TensorRT-Edge-LLM
#     already disables the fusion, but only for TensorRT >= 10.15, so 10.13 falls through
#     the gap. FP8 happens to dodge the same bug because its Q/DQ nodes break the fusion
#     pattern, which is why FP8 once looked mandatory -- it is not.
#
# The visual encoder is sized for multi-image VLN prompts (~1764 image tokens across 9-10
# frames). The single-image demo default of 512 max image tokens cannot hold one.

set -euo pipefail

MODEL_PATH=""
ENGINE_DIR=""
ONNX_DIR=""
TRT_EDGELLM_DIR="${TRT_EDGELLM_DIR:-$HOME/modelopt/TensorRT-Edge-LLM}"
NO_QUANTIZATION=0
SKIP_VISUAL=0
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-3072}"
MAX_KV_CACHE="${MAX_KV_CACHE:-4096}"
LUNOWUD_WAR="${LUNOWUD_WAR:--peep:fc_h_fusion=off}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)       MODEL_PATH="$2"; shift 2 ;;
        --engine_dir)       ENGINE_DIR="$2"; shift 2 ;;
        --onnx_dir)         ONNX_DIR="$2"; shift 2 ;;
        --trt_edgellm_dir)  TRT_EDGELLM_DIR="$2"; shift 2 ;;
        --no_quantization)  NO_QUANTIZATION=1; shift ;;
        --skip_visual)      SKIP_VISUAL=1; shift ;;
        --max_batch_size)   MAX_BATCH_SIZE="$2"; shift 2 ;;
        --max_input_len)    MAX_INPUT_LEN="$2"; shift 2 ;;
        --max_kv_cache)     MAX_KV_CACHE="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,25p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "[ERROR] unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$MODEL_PATH" ]] || { echo "[ERROR] --model_path is required" >&2; exit 1; }
[[ -n "$ENGINE_DIR" ]] || { echo "[ERROR] --engine_dir is required" >&2; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo "[ERROR] model path not found: $MODEL_PATH" >&2; exit 1; }

ONNX_DIR="${ONNX_DIR:-${ENGINE_DIR%/}_onnx}"
PLUGIN="$TRT_EDGELLM_DIR/build/libNvInfer_edgellm_plugin.so"
LLM_BUILD="$TRT_EDGELLM_DIR/build/examples/llm/llm_build"
VISUAL_BUILD="$TRT_EDGELLM_DIR/build/examples/multimodal/visual_build"

for f in "$PLUGIN" "$LLM_BUILD"; do
    [[ -f "$f" ]] || { echo "[ERROR] not found: $f" >&2
                       echo "        set TRT_EDGELLM_DIR or build TensorRT-Edge-LLM first" >&2
                       exit 1; }
done

export EDGELLM_PLUGIN_PATH="$PLUGIN"
export __LUNOWUD="${__LUNOWUD:+$__LUNOWUD }$LUNOWUD_WAR"

echo "=============================================================="
echo " InternVLA-N1 System 2 — ONNX export and engine build"
echo "=============================================================="
echo " Model path:      $MODEL_PATH"
echo " Quantized:       $([[ $NO_QUANTIZATION -eq 1 ]] && echo 'no (FP16 reference)' || echo yes)"
echo " ONNX dir:        $ONNX_DIR"
echo " Engine dir:      $ENGINE_DIR"
echo " TensorRT-Edge:   $TRT_EDGELLM_DIR"
echo " maxBatchSize:    $MAX_BATCH_SIZE"
echo " maxInputLen:     $MAX_INPUT_LEN"
echo " maxKVCacheCap:   $MAX_KV_CACHE"
echo " __LUNOWUD:       $__LUNOWUD"
echo "=============================================================="

mkdir -p "$ONNX_DIR" "$ENGINE_DIR"

echo
echo "[1/3] Exporting ONNX (emit_hidden_states=True for the System 1 bridge)..."
python - "$MODEL_PATH" "$ONNX_DIR" <<'PYEOF'
import sys
from tensorrt_edgellm.models.default.modeling_default import CausalLM
# The bridge to System 1 reads the last-layer hidden states, not just logits.
CausalLM.emit_hidden_states = True
from tensorrt_edgellm.scripts.export import main
sys.argv = ["tensorrt-edgellm-export", sys.argv[1], sys.argv[2]]
sys.exit(main())
PYEOF

echo
echo "[2/3] Building the LLM engine..."
mkdir -p "$ENGINE_DIR/llm"
"$LLM_BUILD" \
    --onnxDir "$ONNX_DIR/llm" \
    --engineDir "$ENGINE_DIR/llm" \
    --maxBatchSize "$MAX_BATCH_SIZE" \
    --maxInputLen "$MAX_INPUT_LEN" \
    --maxKVCacheCapacity "$MAX_KV_CACHE"

# llm_bench reads its engine configuration from base_config.json, while llm_build writes
# config.json. Same content, different name; without this copy the benchmark cannot open
# the engine it was just handed.
cp "$ENGINE_DIR/llm/config.json" "$ENGINE_DIR/llm/base_config.json"

if [[ $SKIP_VISUAL -eq 0 && -d "$ONNX_DIR/visual" ]]; then
    echo
    echo "[3/3] Building the visual engine..."
    [[ -f "$VISUAL_BUILD" ]] || { echo "[ERROR] not found: $VISUAL_BUILD" >&2; exit 1; }
    # visual_build appends its own "visual" component directory under --engineDir, so
    # pass the parent: giving it $ENGINE_DIR/visual yields $ENGINE_DIR/visual/visual and
    # every consumer that expects $ENGINE_DIR/visual/visual.engine then misses it.
    #
    # A VLN prompt carries 9-10 frames and roughly 1764 image tokens. The single-image
    # demo default of 512 cannot hold one.
    "$VISUAL_BUILD" \
        --onnxDir "$ONNX_DIR/visual" \
        --engineDir "$ENGINE_DIR" \
        --minImageTokens 4 \
        --maxImageTokens 4096 \
        --maxImageTokensPerImage 1024
else
    echo
    echo "[3/3] Skipping the visual engine."
fi

echo
echo "=============================================================="
echo " Build complete"
echo "=============================================================="
du -sh "$ENGINE_DIR"/* 2>/dev/null | sed 's/^/  /'
echo " Engines: $ENGINE_DIR"
echo " ONNX kept at $ONNX_DIR (safe to delete once the engines verify)"
