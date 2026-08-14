# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
# Makefile — Top-level command panel for internvla-n1-dualvln
#
# Override paths via environment variables, e.g.:
#   make quantize-fp8 INTERNVLA_CKPT=/data/InternVLA-N1-DualVLN WORK_DIR=/mnt/scratch
#
# Targets that run Python/bash assume the virtualenv is already activated.
#
# Targets are grouped by dependency boundary. Everything under "System 2" runs
# without InternNav; System 1 and the agent-level checks require INTERNNAV_PATH.

# ── Configurable paths ─────────────────────────────────────────────────────────
INTERNVLA_CKPT  ?= $(HOME)/InternNav/checkpoints/InternVLA-N1-DualVLN
INTERNNAV_PATH  ?= $(HOME)/InternNav
TRT_EDGELLM_DIR ?= $(HOME)/modelopt/TensorRT-Edge-LLM
WORK_DIR        ?= $(HOME)/vln-opt-work
ENGINE_DIR      ?= $(WORK_DIR)/engines
CALIB_DATA_ROOT ?= $(WORK_DIR)/calib_scenes
SCHEME          ?= fp8_default
STRATEGY        ?= s1
DEVICE          ?= cuda
REPO_ROOT       := $(abspath ../..)

REPKG_CKPT := $(WORK_DIR)/qwen25vl_system2
QUANT_CKPT := $(WORK_DIR)/qwen25vl_$(STRATEGY)_$(SCHEME)

# ── System 2: quantize (no InternNav required) ────────────────────────────────
.PHONY: fetch-calib repackage quantize quantize-fp8 quantize-nvfp4

fetch-calib:
	bash quantize/scripts/00_fetch_calib_scenes.sh \
	    --output_path $(CALIB_DATA_ROOT)

repackage:
	bash quantize/scripts/01_repackage.sh \
	    --model_path  $(INTERNVLA_CKPT) \
	    --output_path $(REPKG_CKPT)

quantize:
	bash quantize/scripts/02_quantize.sh \
	    --model_path  $(REPKG_CKPT) \
	    --output_path $(QUANT_CKPT) \
	    --scheme      $(SCHEME) \
	    --strategy    $(STRATEGY) \
	    --device      $(DEVICE)

quantize-fp8:
	$(MAKE) quantize SCHEME=fp8_default STRATEGY=s1

quantize-nvfp4:
	$(MAKE) quantize SCHEME=nvfp4_default STRATEGY=s1

# ── System 2: export, build, verify (no InternNav required) ────────────────────
.PHONY: export-build export-build-base-fp16 verify-latents benchmark

export-build:
	bash trt-edgellm/scripts/03_export_build_system2.sh \
	    --model_path      $(QUANT_CKPT) \
	    --engine_dir      $(ENGINE_DIR)/$(STRATEGY)_$(SCHEME) \
	    --trt_edgellm_dir $(TRT_EDGELLM_DIR)

export-build-base-fp16:
	bash trt-edgellm/scripts/03_export_build_system2.sh \
	    --model_path      $(REPKG_CKPT) \
	    --engine_dir      $(ENGINE_DIR)/base_fp16 \
	    --trt_edgellm_dir $(TRT_EDGELLM_DIR) \
	    --no_quantization

verify-latents:
	bash trt-edgellm/scripts/05_verify.sh \
	    --engine_dir $(ENGINE_DIR)/$(STRATEGY)_$(SCHEME) \
	    --repkg_ckpt $(REPKG_CKPT) \
	    --calib_data_root $(CALIB_DATA_ROOT)

# 06_measure.sh sweeps every engine under ENGINE_DIR, so it takes the parent directory
# rather than one engine, and is configured by environment rather than flags.
benchmark:
	WORK_DIR=$(WORK_DIR) ENGINE_DIR=$(ENGINE_DIR) TRT_EDGELLM_DIR=$(TRT_EDGELLM_DIR) \
	    bash trt-edgellm/scripts/06_measure.sh

# ── System 1: BF16 engines (requires INTERNNAV_PATH) ──────────────────────────
.PHONY: export-system1 verify-system1 quantize-system1-fp8

export-system1:
	bash trt-edgellm/scripts/04_export_system1.sh \
	    --internvla_ckpt  $(INTERNVLA_CKPT) \
	    --internnav_path  $(INTERNNAV_PATH) \
	    --engine_dir      $(ENGINE_DIR)/system1

# Measured and deliberately not shipped: FP8 costs 6x the waypoint deviation for 1.7% of
# the planning step. Kept so the finding stays reproducible. See README.
quantize-system1-fp8:
	python trt-edgellm/dump_system1_calib.py \
	    --calib_data_root $(CALIB_DATA_ROOT) \
	    --output_path     $(WORK_DIR)/system1_calib.pt
	python trt-edgellm/quantize_system1.py \
	    --calib_path $(WORK_DIR)/system1_calib.pt \
	    --out_dir    $(ENGINE_DIR)/system1_fp8

verify-system1:
	bash trt-edgellm/scripts/05_verify.sh \
	    --system1 \
	    --internvla_ckpt $(INTERNVLA_CKPT) \
	    --internnav_path $(INTERNNAV_PATH) \
	    --engine_dir     $(ENGINE_DIR)/system1

# ── NVFP4 investigation ───────────────────────────────────────────────────────
.PHONY: investigate-nvfp4

investigate-nvfp4:
	python trt-edgellm/investigate_nvfp4.py \
	    --repkg_ckpt      $(REPKG_CKPT) \
	    --calib_data_root $(CALIB_DATA_ROOT) \
	    --work_dir        $(WORK_DIR)

# ── Housekeeping ──────────────────────────────────────────────────────────────
.PHONY: clean-onnx help

# ONNX is a pure intermediate and the largest reclaimable artifact (9-16 GB).
clean-onnx:
	rm -rf $(WORK_DIR)/onnx
	@echo "Removed $(WORK_DIR)/onnx"

help:
	@echo "internvla-n1-dualvln — targets"
	@echo ""
	@echo "  System 2 (no InternNav needed):"
	@echo "    fetch-calib             download VLN calibration scenes"
	@echo "    repackage               InternVLA checkpoint -> stock Qwen2.5-VL System 2"
	@echo "    quantize                quantize with SCHEME/STRATEGY (default fp8_default/s1)"
	@echo "    quantize-fp8            shorthand for SCHEME=fp8_default STRATEGY=s1"
	@echo "    quantize-nvfp4          shorthand for SCHEME=nvfp4_default STRATEGY=s1 (experimental)"
	@echo "    export-build            ONNX export + LLM and visual engines"
	@echo "    export-build-base-fp16  unquantized FP16 reference engine"
	@echo "    verify-latents          acceptance gate: z_latents cosine vs FP32"
	@echo "    benchmark               latency and memory"
	@echo ""
	@echo "  System 1 (requires INTERNNAV_PATH):"
	@echo "    export-system1          traj_dit + memory block -> BF16 engines"
	@echo "    verify-system1          trajectory parity vs PyTorch"
	@echo "    quantize-system1-fp8    System 1 FP8 PTQ (measured, not recommended)"
	@echo ""
	@echo "  Other:"
	@echo "    investigate-nvfp4       why NVFP4 breaks the System2 -> System1 bridge"
	@echo "    clean-onnx              remove the ONNX intermediate (9-16 GB)"
	@echo ""
	@echo "  Variables: INTERNVLA_CKPT INTERNNAV_PATH TRT_EDGELLM_DIR WORK_DIR"
	@echo "             ENGINE_DIR CALIB_DATA_ROOT SCHEME STRATEGY DEVICE"
