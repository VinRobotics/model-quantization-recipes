# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
# Makefile — internvla-n1-dualvln
#
# Builds the navigation-domain calibration set this recipe is responsible for. Quantize,
# export, and build are the standard TensorRT-Edge-LLM CLI, run in that repo's own
# environment -- see README.md for why they are not wrapped here.
#
# Override paths via environment variables, e.g.:
#   make build-calib CALIB_DATA_ROOT=/mnt/scratch/calib

INTERNVLA_CKPT   ?= $(HOME)/InternNav/checkpoints/InternVLA-N1-DualVLN
CALIB_DATA_ROOT  ?= $(HOME)/vln-opt-work/calib
TRAIN_JSON       := $(CALIB_DATA_ROOT)/vln_ce/raw_data/r2r/train/train.json.gz
NAV_CALIB_JSONL  := $(CALIB_DATA_ROOT)/nav_calib.jsonl
NUM_SAMPLES      ?= 512

.PHONY: fetch-calib-source build-calib all clean help

fetch-calib-source:
	huggingface-cli download InternRobotics/InternData-N1 \
	    vln_ce/raw_data/r2r/train/train.json.gz --repo-type dataset \
	    --local-dir $(CALIB_DATA_ROOT)

build-calib: $(TRAIN_JSON)
	python quantize/build_calib_jsonl.py \
	    --train_json $(TRAIN_JSON) \
	    --output     $(NAV_CALIB_JSONL) \
	    --num_samples $(NUM_SAMPLES)

all: fetch-calib-source build-calib
	@echo ""
	@echo "Calibration set ready: $(NAV_CALIB_JSONL)"
	@echo "Quantize/export/build with the TensorRT-Edge-LLM CLI -- see README.md:"
	@echo "  export EDGELLM_QUANT_DATASET_CNN_DAILYMAIL=$(NAV_CALIB_JSONL)"
	@echo "  tensorrt-edgellm-quantize llm --model_dir $(INTERNVLA_CKPT) \\"
	@echo "      --output_dir <out> --quantization {fp8,nvfp4}"

clean:
	rm -rf $(CALIB_DATA_ROOT)

help:
	@echo "internvla-n1-dualvln — targets"
	@echo ""
	@echo "  fetch-calib-source   download InternData-N1's R2R train split (2.4 MB, gated)"
	@echo "  build-calib          build the navigation-prompt calibration JSONL"
	@echo "  all                  both of the above, then print the CLI commands to run next"
	@echo "  clean                remove CALIB_DATA_ROOT"
	@echo ""
	@echo "  Variables: INTERNVLA_CKPT CALIB_DATA_ROOT NUM_SAMPLES"
	@echo ""
	@echo "  Quantize / export / build / run are the standard TensorRT-Edge-LLM CLI,"
	@echo "  not wrapped here -- see README.md."
