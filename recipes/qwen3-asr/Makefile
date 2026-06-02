# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
# Makefile — Top-level command panel for qwen-asr-optimization
#
# Override paths via environment variables, e.g.:
#   make vllm-quantize-fp8 MODEL_PATH=/data/Qwen3-ASR-1.7B QWEN_ASR_ROOT=./external/qwen3_asr
#
# Targets that run Python/bash assume the virtualenv is already activated.

# ── Configurable paths ─────────────────────────────────────────────────────────
MODEL_PATH      ?= ./Qwen3-ASR-1.7B
QWEN_ASR_ROOT   ?= ./external/qwen3_asr
TRT_EDGELLM_DIR ?= ./external/tensorrt_edge_llm
DEVICE          ?= cuda:0
REPO_ROOT       := $(abspath ../..)

# ── vLLM ──────────────────────────────────────────────────────────────────────
.PHONY: vllm-quantize-fp8 vllm-quantize-nvfp4 vllm-export-fp8 vllm-export-nvfp4

vllm-quantize-fp8:
	bash vllm/scripts/01_quantize.sh \
	    --model_path    $(MODEL_PATH) \
	    --qwen_asr_root $(QWEN_ASR_ROOT) \
	    --format        fp8 \
	    --algorithm     max \
	    --output_pt     ./qwen3asr_thinker_fp8.pt

vllm-quantize-nvfp4:
	bash vllm/scripts/01_quantize.sh \
	    --model_path    $(MODEL_PATH) \
	    --qwen_asr_root $(QWEN_ASR_ROOT) \
	    --format        nvfp4 \
	    --algorithm     max \
	    --output_pt     ./qwen3asr_thinker_nvfp4.pt

vllm-export-fp8:
	bash vllm/scripts/02_export.sh \
	    --model_path    $(MODEL_PATH) \
	    --qwen_asr_root $(QWEN_ASR_ROOT) \
	    --quant_pt      ./qwen3asr_thinker_fp8.pt \
	    --tmp_dir       ./qwen3asr_fp8_hf \
	    --export_dir    ./qwen3asr_fp8_vllm

vllm-export-nvfp4:
	bash vllm/scripts/02_export.sh \
	    --model_path    $(MODEL_PATH) \
	    --qwen_asr_root $(QWEN_ASR_ROOT) \
	    --quant_pt      ./qwen3asr_thinker_nvfp4.pt \
	    --tmp_dir       ./qwen3asr_nvfp4_hf \
	    --export_dir    ./qwen3asr_nvfp4_vllm

# ── TRT-Edge-LLM ──────────────────────────────────────────────────────────────
.PHONY: trt-quantize-int8 trt-quantize-int4

trt-quantize-int8:
	bash trt-edgellm/scripts/01_quantize.sh \
	    --model_path    $(MODEL_PATH) \
	    --qwen_asr_root $(QWEN_ASR_ROOT) \
	    --format        int8 \
	    --output_dir    ./Qwen3-ASR-1.7B-int8

trt-quantize-int4:
	bash trt-edgellm/scripts/01_quantize.sh \
	    --model_path    $(MODEL_PATH) \
	    --qwen_asr_root $(QWEN_ASR_ROOT) \
	    --format        int4 \
	    --output_dir    ./Qwen3-ASR-1.7B-int4

# ── Setup ─────────────────────────────────────────────────────────────────────
.PHONY: setup setup-submodules

setup:
	cd $(REPO_ROOT) && uv sync --extra qwen3-asr

setup-submodules:
	bash external/init_submodules.sh

# ── Help ──────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  Setup:"
	@echo "    make setup                  Install Python requirements with uv"
	@echo "    make setup-submodules       Clone Qwen3-ASR + TRT-Edge-LLM"
	@echo ""
	@echo "  vLLM path (RTX 5090 / x86):"
	@echo "    make vllm-quantize-fp8      PTQ → FP8"
	@echo "    make vllm-quantize-nvfp4    PTQ → NVFP4"
	@echo "    make vllm-export-fp8        Export FP8 checkpoint for vLLM"
	@echo "    make vllm-export-nvfp4      Export NVFP4 checkpoint for vLLM"
	@echo ""
	@echo "  TRT-Edge-LLM path (Jetson Orin):"
	@echo "    make trt-quantize-int8      PTQ → INT8 SmoothQuant"
	@echo "    make trt-quantize-int4      PTQ → INT4 AWQ"
	@echo ""
	@echo "  Override defaults:"
	@echo "    make <target> MODEL_PATH=/path/to/model QWEN_ASR_ROOT=/path/to/qwen"
	@echo ""
